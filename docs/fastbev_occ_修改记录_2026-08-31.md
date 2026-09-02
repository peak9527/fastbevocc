# FastBEV-OCC 双任务模型修改记录

- **日期**：2026-08-31
- **项目**：`/home/server/workspace/nvidia/fastbev_py`（分支 `urban-dev`）
- **运行环境**：conda 环境 `bev`（`/home/server/miniconda3/envs/bev`，Python 3.8）
- **目标配置**：`configs/fastbev/paper/fastbev-occ-r50-cbgs-custom.py`
- **涉及模型**：FastBEVOCC（3D 目标检测 + 占据栅格 occ 双任务，自定义 yk_c1 数据集）

---

## 1. 背景与问题

### 1.1 检测分支变差（本次核心问题）

FastBEVOCC 双任务模型中，`BEVAug`（BEV 数据增强）只对 occ GT 做 flip（水平/垂直翻转），**不支持旋转/缩放同步变换**。因此官方 occ 配置把 BEV 旋转/缩放增强关闭：

```python
# 官方 fastbev-occ-r50-cbgs.py 中的做法
bda_aug_conf = dict(
    rot_lim=(0., 0.),      # 旋转关闭
    scale_lim=(1., 1.),    # 缩放关闭
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)
```

后果：检测分支缺少航向角（旋转）与目标尺度（缩放）增强，导致 **3D 目标检测变差**（尤其是航向回归与尺度泛化）。

### 1.2 类别不平衡（CBGS）

yk_c1 数据集中 `car` 等大类占比高，稀有类（`motorcycle/bicycle/construction_vehicle` 等）样本少，检测分支小类 AP 偏低。

---

## 2. 修改文件清单

| 文件 | 修改类型 | 内容 |
|---|---|---|
| `mmdet3d/datasets/pipelines/loading.py` | 功能扩展 | `BEVAug` 支持用 bda 矩阵对 occ GT 做最近邻重采样 |
| `configs/fastbev/paper/fastbev-occ-r50-cbgs-custom.py` | 配置修改 | CBGS 采样、60e 配方、恢复 BDA 增强、传入 occ 栅格参数 |

---

## 3. 详细改动

### 3.1 `mmdet3d/datasets/pipelines/loading.py` — `BEVAug` 类

#### 3.1.1 `__init__` 新增可选参数

```python
def __init__(self,
             bda_aug_conf,
             classes,
             is_train=True,
             point_cloud_range=None,   # 新增：occ GT 栅格世界坐标范围（如 [-40,-40,-5,40,40,3]）
             occ_voxel_size=None):     # 新增：occ 栅格分辨率（如 [0.4, 0.4]）
```

- 两个参数**可选**，不传时保持旧行为（仅 flip），**向后兼容**，不影响其他配置（bevdet-occ 等）。
- 传入后，`BEVAug` 会把 bda 的旋转/缩放/平移通过最近邻重采样同步作用于 occ GT。

#### 3.1.2 `__call__` 处理分支

```python
if 'voxel_semantics' in results:
    if self.point_cloud_range is not None and \
            self.occ_voxel_size is not None:
        self._aug_occ_by_bda(results, bda_rot, tran_bda)   # 新路径：完整 bda 重采样
    else:
        # 旧路径：仅 flip（[::-1] / [:, ::-1]），保持向后兼容
        ...
```

#### 3.1.3 新增 `_aug_occ_by_bda()` 方法（核心）

**算法**：用与检测框**完全相同的 bda 矩阵**（`p_aug = R @ p_orig + t`）对 `voxel_semantics / mask_lidar / mask_camera` 做最近邻重采样。

- **布局**：`voxel_semantics` 为 `[x, y, z]`（如 200×200×16 @ 0.4m），第一维 = x，第二维 = y。
- **逆变换**：`p_orig = R⁻¹(p_aug − t)` 映射回原栅格索引。
- **仅作用于 xy 平面**：bda 只旋转/缩放 xy（z 分量为 1），z 层保持不变。
- **越界处理**：重采样越界区域填 0（ignore，不参与 loss，与 `mask=0` 语义一致）。
- **浮点鲁棒性修复**：`valid` 判定区间用与 `np.round` 舍入一致的 `[-0.5, N-0.5)`，避免浮点误差（如 `199.00000000000003`）把边界栅格误判为越界而清零。

### 3.2 `configs/fastbev/paper/fastbev-occ-r50-cbgs-custom.py`

#### 3.2.1 train 启用 CBGSDataset 类平衡采样

```python
train=dict(
    type='CBGSDataset',                    # 外层：类平衡采样包装
    dataset=dict(
        type='NuScenesDatasetOccpancy',    # 内层：occ 数据集
        data_root=data_root,
        ann_file=ann_file,
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR')),
```

- CBGS 按类别密度过采样稀有类（`motorcycle/bicycle/construction_vehicle` 等），提升检测分支小类 AP，**不影响 occ 训练**。
- 验证：`build_dataset` 输出 `CBGSDataset -> NuScenesDatasetOccpancy`（`mmdet3d/datasets/builder.py` 有专门的 `CBGSDataset` 分支，按字符串匹配，无需额外注册）。

#### 3.2.2 `share_data_config` 合并逻辑（关键坑点）

```python
# 注意：train 是 CBGSDataset 包装，share_data_config 需 update 到内层 dataset，
# 否则会覆盖外层 type='CBGSDataset'，导致 CBGS 失效。
for key in ['val', 'test']:
    data[key].update(share_data_config)
    data[key]['data_root'] = data_root
data['train']['dataset'].update(share_data_config)
data['train']['dataset']['data_root'] = data_root
```

- 只把 `share_data_config` 合并到 **val/test 外层** + **train 内层 `dataset`**。
- 若合并到 train 外层，会把 `type='CBGSDataset'` 覆盖掉 → CBGS 失效；若 val/test 也变成 CBGSDataset 会因缺少 `dataset` 字段而 `KeyError`。

#### 3.2.3 优化器配方 50e → 60e（当前临时 30e 快速验证）

正式配方（50e → 60e）：

```python
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
lr_config = dict(policy='step', warmup='linear', warmup_iters=200,
                 warmup_ratio=0.001, step=[48, 55])   # 50e 时是 [40, 46]
runner = dict(type='EpochBasedRunner', max_epochs=60)  # 50e → 60e
```

> **临时改动（快速验证）**：为快速出结果，当前配置临时改为 30e，两次衰减按 60e 配方等比换算为 `[24, 27]`：
> ```python
> runner = dict(type='EpochBasedRunner', max_epochs=30)
> lr_config = dict(..., step=[24, 27])
> ```
> 验证完成后恢复正式配方 `max_epochs=60, step=[48, 55]`。

#### 3.2.4 恢复 BEVAug 旋转/缩放增强（官方 FastBEV 幅度）

```python
bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),     # 官方 FastBEV 检测幅度（±22.5°）
    scale_lim=(0.95, 1.05),    # 官方 FastBEV 检测幅度
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)
```

- 官方 FastBEV 检测配置（`fastbev-r50-cbgs.py` 等）统一使用该幅度；mmdet3d 官方 nuScenes 检测（`GlobalRotScaleTrans`，`configs/_base_/datasets/nus-3d.py`）同为 `±22.5°/0.95~1.05`。
- 曾短暂使用更大幅度 `±45°/0.9~1.1`（对应 DAL 配置），最终按用户要求改为官方幅度，对 occ 干扰最小。

#### 3.2.5 BEVAug 传入 occ 栅格参数（启用 occ 重采样）

```python
dict(
    type='BEVAug',
    bda_aug_conf=bda_aug_conf,
    classes=class_names,
    point_cloud_range=point_cloud_range,   # occ GT 世界坐标范围
    occ_voxel_size=[0.4, 0.4]),            # occ 栅格分辨率 0.4m
```

- test pipeline 的 `BEVAug` 保持 `is_train=False`（无增强），不传 occ 参数。

---

## 4. 验证结果

### 4.1 配置解析与数据集构建（`bev` 环境实测）

```
train.type          = CBGSDataset
train.dataset.type  = NuScenesDatasetOccpancy
val.type            = NuScenesDatasetOccpancy
max_epochs          = 60 | lr step = [48, 55] | samples_per_gpu = 4   # 验证时配方；当前临时为 30e / [24, 27]
build_dataset OK    = CBGSDataset -> NuScenesDatasetOccpancy
```

### 4.2 BEVAug occ 重采样单元测试

| 测试项 | 结果 |
|---|---|
| flip_dx 与旧 `[::-1]` 完全等价 | PASS |
| flip_dy 与旧 `[:, ::-1]` 完全等价 | PASS |
| 旋转 30° + 缩放 0.9 形状/内容保留 | PASS |
| 90° 旋转中心目标保持 | PASS |
| 真实 train pipeline（含图像加载 + occ 重采样） | PASS（`voxel_semantics` 200×200×16 正常输出，mask 有效） |

### 4.3 occ 有效监督区域量化（官方幅度）

| 场景 | occ GT 有效监督区域 |
|---|---|
| 纯 flip（rot=0, scale=1） | 100% |
| 官方幅度平均（1000 次随机采样） | 91.9% |
| 最坏组合（rot=22.5° + scale=0.95） | 82.0% |
| （对比）±45°/0.9~1.1 平均 | 87.1% |

---

## 5. 预期影响

- **检测分支**：恢复航向/尺度增强（±22.5° 旋转 + 0.95~1.05 缩放），解决之前检测变差的主因之一；CBGS 进一步提升小类 AP。
- **occ 分支**：平均约 8% 有效监督区域被裁掉（四角填 0 ignore），预计 mIoU 约 -0.3 ~ -1.0（相对纯 flip 版本）。mask 与语义同步重采样，loss 计算无错配，旋转/缩放多样性可部分补偿。

---

## 6. 训练与测试

```bash
# 训练（2 卡，当前临时 30e 快速验证，EMA 权重为 epoch_30_ema.pth）
conda activate bev
bash tools/dist_train.sh configs/fastbev/paper/fastbev-occ-r50-cbgs-custom.py 2 \
    --work-dir work_dirs/fastbev-occ-r50-cbgs-custom-free33

# 正式训练：把 max_epochs 恢复 60、step 恢复 [48, 55] 后，EMA 权重为 epoch_60_ema.pth
# 测试 3D 目标检测：使用对应 EMA 权重
```

---

## 7. 注意事项

1. **文件回退风险**：配置曾因外部操作（`git checkout`、编辑器撤销）整体回退。若再次发生，用 `git diff configs/fastbev/paper/fastbev-occ-r50-cbgs-custom.py` 核对是否丢失改动。
2. **幅度可调**：若训练后 occ 掉得超出预期，可收小幅度（如 `rot_lim=(-11.25, 11.25)`、`scale_lim=(0.97, 1.03)`），或回退纯 flip（occ 无损但检测分支回到变差状态）。
3. **向后兼容**：`BEVAug` 新增参数可选，未传 occ 参数的配置（如 bevdet-occ）行为与原先完全一致。
4. **显存**：`samples_per_gpu=4`（0.4m BEV 200×200 + 6 相机 320×896），24GB 卡峰值约 9-10GB/卡，余量充足。

---

## 8. 变更文件 diff 摘要

```
修改：mmdet3d/datasets/pipelines/loading.py     （BEVAug 扩展 + 浮点修复）
修改：configs/fastbev/paper/fastbev-occ-r50-cbgs-custom.py  （CBGS / 60e / BDA 恢复 / occ 栅格参数）
```

---

## 9. 双任务同步提升建议（后续优化方向）

针对 FastBEVOCC（CenterHead 检测 + FastBEVOCCHead 语义占用，共享 ResNet-50 + FastrayTransformer + BEV 特征），能**同步**提升检测与 occ 的方法按性价比分层：

### 9.1 低成本（只改配置/参数，立即可做）

| 方法 | 说明 | 双任务受益点 |
|---|---|---|
| 恢复长训练 + 余弦退火 | 30e → 60e+，`lr_config policy='cosine'`（或 step 多段） | 双任务都欠拟合，多训 + 平滑退火对两个 head 都涨 |
| 加大相机空间增强 | `resize/rot` 幅度加大（接近官方），`post_rot/post_tran` 同步外参 | 纯图像增强，不破坏 occ 坐标，双任务共同受益（与 BEVAug 互补） |
| occ 类别平衡 loss | `occ_head.class_balance=False` → `True`（配置已暴露该参数） | 稀有语义类 mIoU 直接提升 |
| 测试时增强 TTA | test pipeline 加 `flip_tta`（BEV flip + 检测框翻转 + occ 翻转投票） | 检测 AP/NDS 与 occ mIoU 各涨 1~3%，零训练成本 |
| 确认 EMA hook 生效 | `MEGVIIEMAHook init_updates=10560` | 测试务必用 `epoch_*_ema.pth` |

### 9.2 中成本（小改动代码/加模块）

| 方法 | 说明 |
|---|---|
| 提升深度监督（`use_depth=True` 已有） | DepthNet 的 depth loss 是 BEV 特征质量地基：调 depth loss 权重/监督范围（当前 depth `[1,45,0.5]`），深度准 → 相机到 BEV 投影准 → 检测和 occ 同时涨（双任务共享瓶颈） |
| 多帧时序输入（4d） | 参考官方 `fastbev-r50-cbgs-4d-*.py`（时序融合 + velocity）；提升检测速度估计与 occ 动态栅格，显存增加 |
| 检测 NMS 参数调优 | 当前 `nms_thr=0.2`、`nms_rescale_factor` 较激进，可用验证集扫描 `[0.3~0.5]`；`score_threshold=0.1` 可微调 |
| CBGS 幅度微调 | 已在用；稀有类 AP 提升也间接提升这些场景的 occ 精度 |

### 9.3 高成本（结构升级，收益最大）

| 方法 | 说明 |
|---|---|
| backbone 升级 r50 → r101 | 参考官方 `fastbev-r101-cbgs-4d-longterm.py`；双任务都受益，显存 ↑ |
| 更大输入分辨率 | 320×896 → 512×1408：小目标检测 + 远处 occ 分辨率同时提升；显存显著增加，samples_per_gpu 需降 |
| BEV 分辨率 0.4m → 0.2m | occ 精细度翻倍 + 检测定位更准；显存约 ×4，需重新生成 occ GT |
| 检测-occ 特征交互 | 检测热图/框作为 occ instance 提示，或 occ 语义特征回流辅助检测分组；双任务正向耦合，改动大 |

### 9.4 建议优先级（针对 30e 快速验证阶段）

1. **验证结果后恢复 60e**（或直接 60e + cosine），补齐欠拟合——当前 `warmup_iters=200` 在 30e 下占比偏高
2. **开启 `class_balance=True`**（一行改动，occ 白捡）
3. **加大相机空间增强幅度**（一行改动，双任务共同受益，无 occ 坐标风险，比 BEVAug 更安全）
4. 深度监督调优 + TTA 作为下一轮
