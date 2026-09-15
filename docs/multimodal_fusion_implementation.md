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

| 参数 | 图像分支 | 激光分支 | 对齐方式 |
|------|---------|---------|---------|
| 体素大小 | 0.4m | 0.4m | 相同 |
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

## 8. 总结

本次实现成功在 FastBEV 基础上添加了激光雷达输入，通过 PointPillars 风格的体素化、特征编码和 BEV 散射，将激光点云转换为 200×200×64 的 BEV 特征。通过 Concat + 1×1 Conv 的融合方式，将图像 BEV 特征（256ch）与激光 BEV 特征（64ch）融合为统一的 256ch BEV 特征，同时供给检测头和 OCC 头使用。训练日志显示所有损失函数正常收敛，多模态融合链路工作正常。

---

*文档生成时间: 2026-09-15*
