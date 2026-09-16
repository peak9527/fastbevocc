# FastBEV 多模态融合（图像 + 激光雷达）实现文档

## 1. 概述

本文档记录了在 FastBEV 基础上添加激光雷达（LiDAR）输入，实现多模态融合（图像 + 激光）3D 目标检测和语义占用估计的完整实现方案。

### 1.1 核心目标
- 在原有纯视觉 FastBEV 模型基础上，添加激光雷达分支
- 实现图像 BEV 特征与激光 BEV 特征的融合
- 同时支持 3D 目标检测（pts_bbox）和语义占用估计（occ）任务

### 1.2 涉及文件
| 文件路径 | 说明 |
|---------|------|
| `mmdet3d/models/detectors/fastbev.py` | FastBEVOCC 类，新增激光分支和多模态融合逻辑 |
| `configs/fastbev/paper/fastbev-occ-r50-cbgs-fusion.py` | 多模态融合训练配置文件 |
| `work_dirs/fastbev-occ-r50-cbgs-fusion/` | 训练输出目录 |

---

## 2. 技术架构

### 2.1 整体流程

```
                         输入数据
                            │
              ┌─────────────┴─────────────┐
              │                           │
         图像输入 (img)              激光点云 (points)
              │                           │
              ▼                           ▼
    ┌─────────────────┐         ┌─────────────────┐
    │  Image Backbone │         │   Voxelization  │
    │  + Image Neck   │         │  (PillarFeature) │
    └────────┬────────┘         └────────┬────────┘
             │                           │
             ▼                           ▼
    ┌─────────────────┐         ┌─────────────────┐
    │ FastrayTransform│         │  Scatter (BEV)  │
    │  (View Transform)│        └────────┬────────┘
    └────────┬────────┘                 │
             │                         ▼
             ▼               ┌─────────────────┐
    ┌─────────────────┐      │  激光 BEV 特征  │
    │  图像 BEV 特征   │      │   (64ch, 200x200) │
    └────────┬────────┘      └────────┬────────┘
             │                         │
             └──────────┬──────────────┘
                        ▼
              ┌─────────────────┐
              │  特征融合模块    │
              │ (concat + 1x1) │
              └────────┬────────┘
                       ▼
        ┌──────────────────────────────┐
        │      融合 BEV 特征 (256ch)    │
        └──────────────┬───────────────┘
                        │
         ┌──────────────┴──────────────┐
         ▼                              ▼
  ┌─────────────────┐          ┌─────────────────┐
  │  检测头 (CenterHead)│        │  OCC头 (FastBEVOCCHead) │
  │  pts_bbox_head   │          │  occ_head       │
  └─────────────────┘          └─────────────────┘
```

### 2.2 模块详解

#### 2.2.1 图像分支
- **Backbone**: ResNet-50，输出多尺度特征
- **Neck**: CustomFPN，输出 256 通道特征
- **View Transformer**: FastrayTransformer，将透视图特征转换为 BEV 特征
- **BEV Encoder**: ResNet + FPN_LSS，输出 200×200×256 的 BEV 特征

#### 2.2.2 激光雷达分支
| 模块 | 类型 | 说明 |
|------|------|------|
| `pts_voxel_layer` | Voxelization | 体素化：0.4m×0.4m×8.0m，最大 16000/40000 体素 |
| `pts_voxel_encoder` | PillarFeatureNet | pillar 特征提取：4→64 通道 |
| `pts_middle_encoder` | PointPillarsScatter | 散射到 BEV 视图：200×200×64 |
| `pts_bev_encoder_*` | 可选 | 额外的 BEV 特征编码器 |

#### 2.2.3 融合模块
- **融合方式**: Concat + 1×1 Conv
- **输入通道**: 图像 256 + 激光 64 = 320
- **输出通道**: 256

---

## 3. 代码实现

### 3.1 FastBEVOCC 类新增参数

```python
# mmdet3d/models/detectors/fastbev.py

def __init__(self,
             occ_head=None,
             det_downsample=None,
             pts_voxel_layer=None,        # 新增：体素化层
             pts_voxel_encoder=None,      # 新增：体素编码器
             pts_middle_encoder=None,      # 新增：中间编码器
             pts_bev_encoder_backbone=None,# 新增：BEV backbone（可选）
             pts_bev_encoder_neck=None,    # 新增：BEV neck（可选）
             fusion=None,                  # 新增：融合配置
             **kwargs):
```

### 3.2 激光雷达特征提取

```python
@torch.no_grad()
@force_fp32()
def voxelize(self, points):
    """对点云做体素化"""
    voxels, coors, num_points = [], [], []
    for res in points:
        res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
        voxels.append(res_voxels)
        coors.append(res_coors)
        num_points.append(res_num_points)
    voxels = torch.cat(voxels, dim=0)
    num_points = torch.cat(num_points, dim=0)
    # 构建 batch 坐标
    coors_batch = []
    for i, coor in enumerate(coors):
        coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
        coors_batch.append(coor_pad)
    coors_batch = torch.cat(coors_batch, dim=0)
    return voxels, num_points, coors_batch

def extract_pts_feat(self, points):
    """提取激光 BEV 特征"""
    voxels, num_points, coors = self.voxelize(points)
    voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
    batch_size = coors[-1, 0].item() + 1
    x = self.pts_middle_encoder(voxel_features, coors, batch_size)
    if self.pts_bev_encoder_backbone is not None:
        x = self.pts_bev_encoder_backbone(x)
    if self.pts_bev_encoder_neck is not None:
        x = self.pts_bev_encoder_neck(x)
        if isinstance(x, (list, tuple)):
            x = x[0]
    return x
```

### 3.3 特征融合

```python
def fuse_bev_feats(self, img_bev, pts_bev):
    """BEV 特征融合"""
    if self.fusion_type == 'concat':
        x = torch.cat([img_bev, pts_bev], dim=1)
        x = self.fusion(x)
    elif self.fusion_type == 'add':
        x = img_bev + pts_bev
    else:
        raise ValueError(f"Unsupported fusion type: {self.fusion_type}")
    return x
```

### 3.4 前向传播

#### 训练阶段 (`forward_train`)
```python
def forward_train(self, points=None, img_inputs=None, ...):
    # 1. 提取图像 BEV 特征
    img_feats, _, depth = self.extract_feat(
        points, img=img_inputs, img_metas=img_metas, **kwargs)

    # 2. 多模态融合
    if self.with_lidar:
        pts_bev = self.extract_pts_feat(points)
        fused_bev = self.fuse_bev_feats(img_feats[0], pts_bev)
        bev_feats = [fused_bev]
    else:
        bev_feats = img_feats

    # 3. 检测损失
    losses_pts = self.forward_pts_train(
        self._det_feats(bev_feats), gt_bboxes_3d, gt_labels_3d, ...)

    # 4. OCC 损失
    if self.occ_head is not None:
        occ_pred = self.occ_head(bev_feats[0])
        losses_occ = self.occ_head.loss(occ_pred, voxel_semantics, mask_camera)
        losses.update(losses_occ)

    return losses
```

#### 测试阶段 (`simple_test`)
```python
def simple_test(self, points, img_metas, img=None, ...):
    # 1. 提取图像 BEV 特征
    img_feats, _, _ = self.extract_feat(points, img=img, ...)

    # 2. 多模态融合
    if self.with_lidar:
        pts_bev = self.extract_pts_feat(points)
        fused_bev = self.fuse_bev_feats(img_feats[0], pts_bev)
        bev_feats = [fused_bev]
    else:
        bev_feats = img_feats

    # 3. 检测推理
    bbox_pts = self.simple_test_pts(self._det_feats(bev_feats), img_metas, ...)
    result_dict['pts_bbox'] = pts_bbox

    # 4. OCC 推理
    if self.occ_head is not None:
        occ_pred = self.occ_head(bev_feats[0])
        occ_res = self.occ_head.get_occ(occ_pred)
        result_dict['occ'] = occ_res

    return bbox_list
```

---

## 4. 配置文件

### 4.1 输入模态配置

```python
# configs/fastbev/paper/fastbev-occ-r50-cbgs-fusion.py

input_modality = dict(
    use_lidar=True,    # 启用激光雷达
    use_camera=True,    # 启用相机
    use_radar=False,
    use_map=False,
    use_external=False)
```

### 4.2 激光体素化配置

```python
pts_voxel_layer=dict(
    max_num_points=32,
    point_cloud_range=[-40.0, -40.0, -5.0, 40.0, 40.0, 3.0],
    voxel_size=[0.4, 0.4, 8.0],       # xy=0.4m 与图像 BEV 对齐
    max_voxels=(16000, 40000),         # (train, test)
    deterministic=False),
```

### 4.3 Pillar 特征网络配置

```python
pts_voxel_encoder=dict(
    type='PillarFeatureNet',
    in_channels=4,                     # x, y, z, intensity
    feat_channels=[64],
    with_distance=False,
    voxel_size=[0.4, 0.4, 8.0],
    point_cloud_range=[-40.0, -40.0, -5.0, 40.0, 40.0, 3.0],
    norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
    legacy=False),
```

### 4.4 Scatter 配置

```python
pts_middle_encoder=dict(
    type='PointPillarsScatter',
    in_channels=64,
    output_shape=[200, 200]),  # 与图像 BEV 完全对齐
```

### 4.5 融合配置

```python
fusion=dict(
    type='concat',
    in_channels=256 + 64,  # 图像 256ch + 激光 64ch
    out_channels=256),      # 保持与下游模块输入维度一致
```

### 4.6 数据 Pipeline 配置

```python
train_pipeline = [
    dict(type='PrepareImageInputs', is_train=True, data_config={{ _base_.data_config }}),
    dict(type='LoadAnnotations'),
    dict(type='LoadOccGTFromFile'),
    dict(type='BEVAug', ...),
    dict(
        type='LoadPointsFromFile',    # 加载点云
        coord_type='LIDAR',
        load_dim=6,
        use_dim=4,                   # 使用 x, y, z, intensity
        file_client_args={{ _base_.file_client_args }}),
    dict(type='PointToMultiViewDepth', ...),
    dict(type='ObjectRangeFilter', ...),
    dict(type='ObjectNameFilter', classes={{ _base_.class_names }}),
    dict(type='DefaultFormatBundle3D', class_names={{ _base_.class_names }}),
    dict(
        type='Collect3D',
        keys=[
            'points',                 # 新增：点云输入模型
            'img_inputs',
            'gt_bboxes_3d',
            'gt_labels_3d',
            'gt_depth',
            'voxel_semantics',
            'mask_lidar',
            'mask_camera'
        ])
]
```

---

## 5. 关键技术细节

### 5.1 空间对齐

| 分支 | 体素/网格大小 | 说明 |
|------|-------------|------|
| 图像分支 (检测) | 0.4m × 0.4m × 0.4m | grid_config，Z方向 16层 |
| 图像分支 (OCC) | 0.4m × 0.4m × 0.4m | grid_config，Z方向 16层 |
| 激光分支 (PointPillars) | 0.4m × 0.4m × **8.0m** | Pillar 不分层Z |

**对齐方式**：
- X/Y 方向：三支路均为 0.4m，保证空间对齐后可以融合
- Z 方向：图像用分层网格（0.4m × 16层），激光用 pillar（单一高度柱体 8.0m）

| 参数 | 图像分支 | 激光分支 | 对齐方式 |
|------|---------|---------|---------|
| X/Y 体素大小 | 0.4m | 0.4m | 相同 |
| BEV 尺寸 | 200×200 | 200×200 | 相同 |
| 物理范围 | [-40, 40]m | [-40, 40]m | 相同 |
| 输出通道 | 256 | 64 | concat 融合 |

### 5.2 PointPillars 处理流程

1. **体素化**: 将点云划分为 0.4m×0.4m×8.0m 的 pillar
2. **特征编码**: PillarFeatureNet 从 (N, 4) 点特征生成 (N, 64) pillar 特征
3. **散射**: PointPillarsScatter 将 pillar 特征散射到 200×200 的 BEV 网格

### 5.3 融合策略

- **Concat + 1×1 Conv**: 将图像 256ch 和激光 64ch 在通道维拼接，然后通过 1×1 卷积压缩到 256ch
- **优势**: 保留两支路的全部信息，融合后下游模块无需修改

### 5.4 训练策略

- **预训练权重**: 从纯视觉模型 (`epoch_30_ema_custom.pth`) 加载
- **激光分支初始化**: 从零开始训练
- **多任务联合训练**: 同时优化检测损失 + OCC 损失 + 深度损失

---

## 6. 训练日志

### 6.1 训练环境
```
GPU: NVIDIA GeForce RTX 3090 × 2
PyTorch: 1.13.1+cu116
MMDetection3D: 1.0.0rc4+6a7c146
分布式训练: True (NCCL)
```

### 6.2 损失收敛情况

| Epoch | Iter | loss_occ | loss_occ_lovasz | task0.loss_heatmap | loss |
|-------|------|----------|-----------------|-------------------|------|
| 1 | 50 | 3.41 | 0.97 | 8.05 | 32.15 |
| 1 | 200 | 0.24 | 0.88 | 3.09 | 11.64 |
| 1 | 500 | 0.18 | 0.87 | 2.51 | 9.04 |
| 1 | 1000 | 0.15 | 0.86 | 2.29 | 8.13 |
| 1 | 1500 | 0.15 | 0.85 | 2.04 | 7.55 |
| 2 | 50 | 0.14 | 0.85 | 1.93 | 7.09 |

**观察**:
- 所有损失函数均正常收敛
- OCC 损失 (loss_occ) 从 3.41 降至 0.14
- 检测热图损失从 8.05 降至 1.93
- 梯度稳定，grad_norm 在 11-15 范围

---

## 7. 使用方法

### 7.1 训练

```bash
cd /home/server/workspace/nvidia/fastbev_py
bash tools/dist_train.sh configs/fastbev/paper/fastbev-occ-r50-cbgs-fusion.py 2
```

### 7.2 测试

```bash
bash tools/dist_test.sh configs/fastbev/paper/fastbev-occ-r50-cbgs-fusion.py \
    work_dirs/fastbev-occ-r50-cbgs-fusion/epoch_1_ema.pth 2
```

### 7.3 显存优化

如遇 OOM，可将 batch size 从 4 降至 2：
```python
# 在配置文件中取消注释
# data['samples_per_gpu'] = 1
```

---

## 8. 多模态融合的提升点

### 8.1 航向角估计的提升

| 方面 | 纯视觉 | 图像+激光融合 |
|------|--------|--------------|
| **深度来源** | 通过图像估算（存在累计误差） | LiDAR 直接测量（毫米级精度） |
| **目标3D形状** | 从2D图像反推，不准确 | 直接获取精确3D点云 |
| **航向角计算** | 依赖时序或假设，转弯时误差大 | 直接拟合点云方向，稳定准确 |
| **遮挡场景** | 容易被遮挡，丢失目标 | 激光穿透部分遮挡，保留目标点云 |

**转弯场景特别受益的原因**：
```
纯视觉航向估计：
  - 时序法：依赖连续帧间的位置变化 → 转弯时位移方向和朝向不一致，误差累积
  - 单目法：假设目标2D框的方向 → 相机角度变化时误差大

激光+视觉融合：
  - 激光提供精确的局部3D结构
  - 可以直接拟合点云的朝向（如主成分分析PCA）
  - 视觉提供语义信息辅助分类和关联
```

### 8.2 其他方面的提升

| 场景 | 纯视觉 | 图像+激光融合 |
|------|--------|--------------|
| **恶劣天气** | 雨天/雾天图像质量差 | 激光受天气影响相对小 |
| **光照变化** | 夜晚/逆光检测退化 | 激光主动发光，不依赖光照 |
| **小目标检测** | 远处目标像素少，难以检测 | 激光不缩放，可靠检测远处目标 |
| **Occ 占用估计** | 仅靠图像特征推断 | 激光提供真实几何结构 |
| **定位精度** | 视觉定位有漂移 | 激光SLAM精度更高 |

### 8.3 本项目的预期提升

```
激光分支输出的 64ch BEV 特征包含：
  - 精确的 XYZ 位置信息
  - intensity（反射率）信息
  - 目标的边缘和轮廓细节

融合后的 256ch 特征同时包含：
  - 视觉语义信息（物体类别、外观）
  - 激光几何信息（精确位置、朝向、形状）
```

### 8.4 注意事项

| 因素 | 影响 |
|------|------|
| **数据配准精度** | 激光和相机外参标定不准确会引入噪声 |
| **点云密度** | 远处点云稀疏，优势减弱 |
| **融合策略** | concat/add 是否是最优方案（可尝试 attention 融合） |
| **任务权衡** | 融合可能带来检测-OCC任务竞争，需调参 |

---

## 9. 总结

本次实现成功在 FastBEV 基础上添加了激光雷达输入，通过 PointPillars 风格的体素化、特征编码和 BEV 散射，将激光点云转换为 200×200×64 的 BEV 特征。通过 Concat + 1×1 Conv 的融合方式，将图像 BEV 特征（256ch）与激光 BEV 特征（64ch）融合为统一的 256ch BEV 特征，同时供给检测头和 OCC 头使用。训练日志显示所有损失函数正常收敛，多模态融合链路工作正常。

**核心价值**：
- 目标转弯时航向估计更准确——激光直接提供目标的空间结构
- 恶劣天气和光照变化场景下更鲁棒
- Occ 占用估计获得真实几何结构约束

---

## 10. v3 版本：多层级门控融合 + 增强体素特征（2026-09-16）

v1（`fastbev-occ-r50-cbgs-fusion.py`）仅做「BEV 末端 concat 融合」，激光信息进入太晚。
v3（`fastbev-occ-r50-cbgs-fusion-v3.py`）针对该问题做了 4 项改造。

### 10.1 四项优化

| # | 优化 | 配置位置 | 说明 |
|---|------|---------|------|
| 1 | 增强体素特征 | `pts_voxel_encoder` | `PillarFeatureNet` 新增 `with_height_stat`(+4: z_max/z_min/z_mean/z_std)、`with_density`(+1)、`with_intensity_stat`(+2)，共 +7 维；两层 MLP 输出 128ch |
| 2 | 多层级融合 | `multi_level_fusion=True` | 在 BEV backbone 的每个 stage 输出后即做门控融合，再送入 FPN neck；激光信息更早参与特征提取 |
| 3 | OCC 专用激光头 | `pts_occ_head` + `occ_fusion_weight` | 从激光 BEV 特征直接预测 OCC，与主 OCC 通过可学习权重 `alpha=sigmoid(occ_fuse_logit)` 融合；训练时双路监督（激光路 0.5 权重） |
| 4 | 深度引导增强 | `img_view_transformer.loss_depth_weight` | 0.125 → 0.25（实测 1.0 会过度扰动预训练主干、拖慢检测/OCC 收敛） |

### 10.2 新增代码接口

```python
# mmdet3d/models/detectors/fastbev.py

# FastBEV：支持返回多尺度特征
def bev_encoder(self, x, return_multi_level=False)
def extract_img_feat(self, img, img_metas, return_multi_level=False, **kwargs)
def multi_level_fuse(self, img_multi, pts_multi)   # 逐 stage 门控融合

# FastBEVOCC：激光分支
def extract_pts_feat(self, points, return_multi_level=False)  # 新增多尺度返回
def fuse_bev_feats(self, img_bev, pts_bev)         # 支持 concat / add / gated
```

门控融合公式（逐 level / 末端各一次）：

```
gate = Sigmoid(Conv3x3(ReLU(BN(Conv3x3([F_img, F_pts])))))   # → 2 通道
F_fused = gate[:, 0:1] * F_img + gate[:, 1:2] * F_pts
```

OCC 双路融合（可学习权重）：

```python
alpha = torch.sigmoid(self.occ_fuse_logit)          # 初始 0.7（主 OCC 占 70%）
occ_pred = alpha * occ_pred_main + (1 - alpha) * occ_pred_lidar
losses_occ = self.occ_head.loss(occ_pred, ...)      # 梯度回流到 occ_fuse_logit
losses['pts_loss_occ'] = 0.5 * pts_occ_head.loss(occ_pred_lidar, ...)
```

### 10.3 关键实现细节

- **只对 neck 消费的层级建融合模块**：`FPN_LSS` 默认 `input_feature_index=(0, 2)`，
  未被消费的层级若也建融合单元，其参数永远拿不到梯度，DDP（`find_unused_parameters=False`）
  会报 `Expected to have finished reduction` 崩溃。实现中从**已构建的 neck 实例**读取
  `input_feature_index`（而非配置 dict，其有类默认值会误判），只对 `fusion_level_ids` 建融合，
  其余层级原样透传。
- **`backbone_output_ids=[0,1,2]`**：激光 BEV backbone（`CustomResNet`，`num_layer=[2,2,2]`，
  `num_channels=[128,256,512]`）输出全部 3 层以配合多层级融合；`with_cp=False` 规避
  DDP + 多路径消费的兼容性问题。
- **新分支 5x 学习率**：`paramwise_cfg` 给 `pts_*`、`img_proj`、`pts_proj`、`gate_net`、
  `level_fusions`、`occ_fuse_logit` 设 `lr_mult=5.0`（1e-4 → 5e-4），因其为随机初始化，
  而视觉主干与 `occ_head` 来自基线 `epoch_24_ema.pth` 预训练权重。

### 10.4 训练配置

| 项目 | 值 |
|------|-----|
| 配置文件 | `configs/fastbev/paper/fastbev-occ-r50-cbgs-fusion-v3.py` |
| 输出目录 | `work_dirs/fastbev-occ-r50-cbgs-fusion-v3/` |
| 预训练权重 | `work_dirs/fastbev-occ-r50-cbgs-custom/epoch_24_ema.pth`（`load_from`） |
| batch / 优化器 | 2 GPU × `samples_per_gpu=4`（总 batch 8）、AdamW lr 1e-4、warmup 300 iter、step [18, 21]、24 epoch |
| 显存 | 每卡约 19.6 GB / 24 GB |

启动命令：

```bash
cd /home/server/workspace/nvidia/fastbev_py
source /home/server/miniconda3/etc/profile.d/conda.sh && conda activate bev
nohup bash tools/dist_train.sh configs/fastbev/paper/fastbev-occ-r50-cbgs-fusion-v3.py 2 \
    > work_dirs/fastbev-occ-r50-cbgs-fusion-v3/train_nohup_r2.log 2>&1 &
```

### 10.5 训练状态

| 时间 | 事件 |
|------|------|
| 2026-09-16 13:53–16:11 | 多轮调试重启（`samples_per_gpu`、GPU 数调整） |
| 2026-09-16 16:12–17:04 | 训练至 epoch1/iter1050 被中断（无 checkpoint） |
| 2026-09-16 17:07 | **重新启动训练（本次）**，双卡 100% 利用率、显存约 19.6 GB/卡，epoch1 损失正常收敛 |

首个 epoch 损失（本次运行）：

| Iter | loss_depth | task0.loss_heatmap | pts_loss_occ | loss_occ | loss |
|------|-----------|-------------------|--------------|----------|------|
| 50 | 18.10 | 5.71 | 1.55 | 3.34 | 39.65 |
| 100 | 7.12 | 3.77 | 0.21 | 0.83 | 21.47 |

**观察**：深度损失与 OCC 损失均快速下降；`pts_loss_occ` 收敛到同一量级，说明激光分支
OCC 监督生效。预计 24 epoch 约需 2 天，checkpoint 按 epoch 保存在输出目录。

> 待办：训练完成后运行检测（mAP/NDS）与 OCC（mIoU）评估，与纯视觉基线
> `fastbev-occ-r50-cbgs-custom` 及 v1 融合结果对比，并把最终指标补入本文档。

---

## 11. v3.1 修正与增强（2026-09-16，重新启动训练）

v3 首轮训练过程中发现一个实质性缺陷，并在此基础上补齐了激光数据增强与双向融合。

### 11.1 【Bug 修复】点云未参与 bda 增强（最高优先级）

**现象**：`train_pipeline` 顺序是 `BEVAug → LoadPointsFromFile`，而 `BEVAug`
内的点云增强分支写作 `if 'points' in results:` —— 执行 BEVAug 时点云尚未加载，
该分支永远不执行。

**后果**：图像 BEV 处于 bda 坐标系（旋转 ±22.5°、缩放 0.95~1.05、50% 翻转），
激光 BEV 停留在原始 lidar 坐标系，**两个分支空间错位**，融合时激光信息落到
错误的空间位置。

**修复**（对齐官方多模态参考实现 `configs/dal/dal-base.py`）：

```
修复前： PrepareImageInputs → LoadAnnotations → LoadOccGTFromFile
        → BEVAug → LoadPointsFromFile → PointToMultiViewDepth → ...

修复后： PrepareImageInputs → LoadAnnotations → LoadOccGTFromFile
        → LoadPointsFromFile            # 点云先加载
        → BEVAug                        # bda 同步作用到点云
        → PointToMultiViewDepthFusion   # 深度图（内部先做 bda 逆变换）
        → RandomPointDrop → RandomIntensityJitter → PointShuffle
        → ObjectRangeFilter → ...
```

两处必须成对修改：点云已被 bda 增强后，深度图必须改用
`PointToMultiViewDepthFusion`（内部 `p_orig = R^-1 (p_aug - t)` 还原回原始
lidar 坐标系再投影），否则深度监督同样错位。

**验证**：用手工 bda 变换 `p @ R.T + t` 与 pipeline 输出点云逐点比对，
最大误差 `0.000000`。

### 11.2 新增激光数据增强

新增两个 transform（`mmdet3d/datasets/pipelines/transforms_3d.py`）：

| Transform | 作用 | 配置 |
|-----------|------|------|
| `RandomPointDrop` | 逐点伯努利采样丢点，模拟点云稀疏/遮挡；带 `min_points` 下限保护 | `drop_ratio=0.15, min_points=1000` |
| `RandomIntensityJitter` | 加性高斯噪声 + 乘性缩放 + 整帧置零（模拟强度缺失）；保证非负、不改 xyz、不就地修改 | `jitter_std=2.0, clip_range=6.0, scale_range=(0.9,1.1), zero_ratio=0.02` |

参数依据实测 intensity 分布（中位数 7 / 均值 15.3 / 99 分位 81）确定：噪声取
绝对刻度且偏小，避免淹没暗点信号。另启用 `PointShuffle` 打乱点序，避免体素内
`max_num_points=32` 截断偏向采集顺序靠前的点。

实测：原始 122038 点 → 增强后 103764 点（丢点率 0.150，与配置一致）。

### 11.3 方案 A：双向融合（图像语义反哺激光分支）

**动机**：v3 的融合是单向的 —— 激光特征流入图像分支（`multi_level_fuse` 与末端
`fuse_bev_feats`），而激光自己的 `pts_occ_head` 消费的是**纯激光**特征
`pts_bev_final`，从未获得图像语义。激光有几何但缺语义、图像有语义但缺几何。

**实现**：在激光 neck 输出后增加一组与正向结构对称的门控，用融合特征（已含
图像语义）调制激光 BEV 特征，再送入 `pts_occ_head`：

```python
# fastbev.py
def rev_fuse_bev_feats(self, pts_bev, fused_bev):
    pts_feat = self.rev_pts_proj(pts_bev)
    img_feat = self.rev_img_proj(fused_bev)
    gate = self.rev_gate_net(torch.cat([pts_feat, img_feat], dim=1))
    return gate[:, 0:1] * pts_feat + gate[:, 1:2] * img_feat
```

```python
# forward_train / simple_test
pts_bev_for_occ = self.rev_fuse_bev_feats(pts_bev_for_occ, bev_feats[0])
pts_occ_pred = self.pts_occ_head(pts_bev_for_occ)
```

配置（`hidden_channels=64`，为正向 gate 的一半以控制算力）：

```python
reverse_fusion=dict(
    type='gated',
    in_channels_pts=256, in_channels_img=256,
    out_channels=256, hidden_channels=64),
```

**算力代价**（实测）：

| 指标 | 增量 |
|------|------|
| 参数量 | ≈ 0.15 M（`rev_pts_proj` + `rev_img_proj` + `rev_gate_net`，共 7 个张量） |
| 理论 FLOPs | ≈ 6 GMACs/样本 |
| 显存 | torch 统计 19509 → 20403 MB（约 +0.9 GB，含激光增强侧开销） |
| 迭代耗时 | ≈ 2.96 → 3.03 s/iter（约 +2%） |

结论：方案 A 的算力开销很小，适合先落地验证收益。

### 11.4 当前训练状态

| 时间 | 事件 |
|------|------|
| 17:07–17:37 | v3 首轮（旧配置：点云未对齐、无激光增强、单向融合），至 iter 400+ 主动停止 |
| 17:37 | 应用上述三项改动并重启训练（PID 1454269） |

首 100 iter 损失（r3，含全部改动）：

| Iter | loss_depth | loss_heatmap | pts_loss_occ | loss_occ | loss |
|------|-----------|--------------|--------------|----------|------|
| 50 | 18.80 | 4.85 | 1.59 | 3.36 | 39.45 |
| 100 | 7.21 | 2.95 | 0.22 | 0.82 | 20.40 |

> 注意：本次重启同时引入了三类变化（对齐修复 / 激光增强 / 双向融合）。
> 若要单独归因某一项的收益，需各自单独训练一次做消融。

---

*文档生成时间: 2026-09-15*
*最后更新时间: 2026-09-16*
