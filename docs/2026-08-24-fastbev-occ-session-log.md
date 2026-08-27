# FastBEV 检测 + Occupancy 联调会话记录（2026-08-24）

> 项目：`/home/server/workspace/nvidia/fastbev_py`（mmdet3d 扩展，FastBEVOCC：检测 + 语义占用联合训练）
> 硬件：2 × RTX 3090；Python：`/home/server/miniconda3/envs/bev/bin/python`
> 训练命令：`bash tools/dist_train.sh configs/fastbev/paper/fastbev-occ-r50-cbgs.py 2`（默认端口 29500）

---

## 0. 会话背景：当前配置（v2 分支解耦版）

当前 `configs/fastbev/paper/fastbev-occ-r50-cbgs.py` 为 v2 分支解耦架构：

| 项 | v1（旧） | v2（当前） |
|---|---|---|
| FastrayTransformer 输出 | `128x128 @ 0.8m`（[-51.2, 51.2]） | `200x200 @ 0.4m`（[-40, 40]） |
| occ 头输入 | 0.8m 特征 → `grid_sample` 插值到 0.4m | 原生 0.4m 特征（`bev_size==occ_size` 时跳过重采样） |
| 检测分支输入 | 0.8m 特征直连 CenterHead | 经 `det_downsample`（stride-2 卷积）到 `100x100 @ 0.8m` |
| 感知范围 | 102.4m | 80m |
| `point_cloud_range` | [-51.2, 51.2, -5, 51.2, 51.2, 3] | [-40, 40, -5, 40, 40, 3] |
| `grid_size` | [1024, 1024, 40] | [800, 800, 40] |
| 训练配方 | 60e / step=[24] | 24e / step=[24]（FlashOCC 对齐快速版） |
| `load_from` | `data/ckpts/init_merged.pth` | `data/ckpts/epoch_50_ema.pth`（本会话改动） |

detach 开关：`det_downsample = dict(in_channels=256, out_channels=256, detach=True)`。
`detach=True` 用于 P2-4 消融（检测梯度不回流 BEV 特征，验证"检测抢占导致 occ 变差"假设）。

---

## 1. 事项一：load_from 预训练权重选择（epoch_50.pth vs epoch_50_ema.pth）

### 1.1 背景

用上次 50e 训练产出的两个权重做 `load_from` 继续训练：

- `work_dirs/fastbev-occ-r50-cbgs/epoch_50.pth`（raw）
- `work_dirs/fastbev-occ-r50-cbgs/epoch_50_ema.pth`（EMA）

### 1.2 文件结构对比（检查结果）

| 维度 | `epoch_50.pth`（raw） | `epoch_50_ema.pth`（EMA） |
|---|---|---|
| 顶层键 | `meta` / `state_dict` / `optimizer` | `epoch` / `state_dict` / `updates` |
| state_dict 键数 | 486 | 485 |
| optimizer | 有（供 `resume_from` 用） | 无 |
| 质量 | 训练末点，波动较大 | EMA 均值，泛化更好 |

### 1.3 关键结论

- **两者都不含 `det_downsample`**（7 个参数键：`det_downsample.0.conv.weight/bias` 等）→ 当前新架构加载时 `strict=False` 自动随机初始化，属预期；
- raw 文件独有 `occ_head.bev2occ_grid` 非浮点 buffer；`voxel_coords` size mismatch（114688 vs 280000）为无害常量网格 buffer；
- **推荐用 `epoch_50_ema.pth`**：EMA 质量更高、纯权重更干净；若用 raw 做 `resume_from` 会因 optimizer 参数个数对不上而报错（resume 场景不适用）；
- 已落地到配置：`load_from = 'data/ckpts/epoch_50_ema.pth'`（实际训练日志显示加载路径为 `data/ckpts/epoch_50_ema.pth`）。

### 1.4 加载时的无害告警清单

- `size mismatch for img_view_transformer.voxel_coords ... [114688,3] vs [280000,3]`：常量网格 buffer，无害；
- `missing keys in source state_dict: det_downsample.0.conv.weight/bias`：新模块随机初始化，预期内。

---

## 2. 事项二：端口 29500 冲突与误杀事故（教训记录）

### 2.1 事件经过

1. 15:13 启动 24e 训练（PID 2979292，work_dir `work_dirs/fastbev-occ-r50-cbgs-24e`）；
2. 15:14 又重复启动相同配置 → 端口 29500 冲突：
   `RuntimeError: The server socket has failed to listen ... errno: 98 - Address already in use`；
3. 排查时**误将 8 个 DataLoader worker（PID 2980411~2980860）判定为残留进程并 kill**，
   导致正在运行的 24e 训练崩溃（仅跑 21 分钟，无 checkpoint 落盘）；
4. 已承认失误，并重启恢复训练。

### 2.2 教训

- 端口冲突时应先确认已有训练进程是否**还在健康运行**（看 `nohup` 日志心跳 / `nvidia-smi`），
  再决定是否 kill；不要直接按 PID 列表批量 kill；
- DataLoader worker 进程是**父训练进程的子进程**，kill 子进程会连带让训练崩溃；
- 判断"残留进程"应看进程完整命令行（`ps -ef | grep`），确认父进程与启动时间。

### 2.3 重启命令（供参考）

```bash
nohup env PYTHONPATH=... python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port=29500 tools/train.py \
  configs/fastbev/paper/fastbev-occ-r50-cbgs.py \
  --seed 0 --launcher pytorch \
  --work-dir work_dirs/fastbev-occ-r50-cbgs-24e \
  > /path/to/train_24e.log 2>&1 &
```

> 遗留：上次启动未指定独立 `work_dir`，checkpoint 会写入 50e 旧目录
> `work_dirs/fastbev-occ-r50-cbgs`（污染 `latest.pth`）。下次务必用独立目录，如
> `--work-dir work_dirs/fastbev-occ-r50-cbgs-24e-c`。

---

## 3. 事项三：DDP 报错根因与 detach 修复（核心修改）

### 3.1 报错现象

```
RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one.
Parameter indices which did not receive grad for rank 0: 259 260
```

触发条件：配置 `detach` 从 `False` 改为 `True` 后复现。

### 3.2 根因

`mmdet3d/models/detectors/fastbev.py` 的 `_det_feats` 中，`detach()` 放在
`det_downsample` **输出之后**，导致：

- `det_downsample` 的输出被 detach → 其自身参数永远收不到梯度；
- 配合 DDP 默认 `find_unused_parameters=False`，报 `Parameter indices ... 259 260` 崩溃。

### 3.3 修复前后对比

```python
# 修复前（错误）：detach 在 det_downsample 之后 —— det_downsample 无梯度
def _det_feats(self, img_feats):
    if self.det_downsample is not None:
        det_feats = [self.det_downsample(img_feats[0])]
    else:
        det_feats = img_feats
    if self.detach_det:
        det_feats = [f.detach() for f in det_feats]   # ← 太晚，阻断自身梯度
    return det_feats

# 修复后（正确）：detach 在 det_downsample 输入之前
def _det_feats(self, img_feats):
    if self.detach_det:
        img_feats = [f.detach() for f in img_feats]   # ← 只切断共享 BEV 特征回流
    if self.det_downsample is not None:
        det_feats = [self.det_downsample(img_feats[0])]
    else:
        det_feats = img_feats
    return det_feats
```

### 3.4 修复后的语义

- `detach=True`：检测 loss 的梯度**不回流** BEV 特征 / 视图变换网络，但 `det_downsample`
  自身照常训练（接收检测梯度）；
- 用于验证"检测任务抢占 BEV 特征导致 occ 变差"（P2-4）；
- 修复后 lint 通过；**训练尚未以修复后代码重启**（见 §5 遗留事项）。

---

## 4. 本会话涉及的代码修改清单（工作区 diff 对比）

`git status` 当前修改：4 个已跟踪文件 + 3 个未跟踪文件。

| 文件 | 变更 | 说明 |
|---|---|---|
| `configs/fastbev/paper/fastbev-occ-r50-cbgs.py` | +94 / 大量修改 | v2 分支解耦注释、`point_cloud_range`→[-40,40]、`grid_size`→[800,800,40]、新增 `det_downsample` 配置、`aux_loss_occ` 开启 CE+Lovász+sem_scal、24e/step=[24]、`load_from`→`epoch_50_ema.pth` |
| `mmdet3d/models/detectors/fastbev.py` | +58 / -11 | 新增 `det_downsample` 构造参数与 `_det_feats` 方法；train/test/aug_test 三处改走 `_det_feats`；**detach 位置修复**（§3） |
| `mmdet3d/models/dense_heads/bev_occ_head.py` | +38 / -2 | 新增 `aux_loss_occ` 参数与 `_calc_aux_losses`；`bev2occ_grid` 在 `bev_size==occ_size` 且范围相同时返回 None 跳过重采样 |
| `mmdet3d/models/losses/__init__.py` | +4 / -1 | 注册 `LovaszSoftmaxLoss` / `SemScalLoss` / `GeoScalLoss` |
| `mmdet3d/models/losses/occ_aux_loss.py`（新） | 245 行 | FlashOCC 风格辅助损失：Lovász（直优化 mIoU）、sem_scal、geo_scal；输入统一扁平化 `(P, C)` logits，适配 mask_camera 加权 |
| `docs/architecture.md`（新） | 737 行 | 全系统 Mermaid 架构图（20 张） |
| `tools/md_to_pdf.py`（新） | — | markdown 转 PDF 工具 |

### 4.1 损失改动对比（occ 分支）

| 版本 | loss 组合 |
|---|---|
| 旧 | 仅 `loss_occ`（CrossEntropyLoss, ignore_index=255, mask_camera 加权） |
| 新 | `loss_occ`（CE）+ `loss_occ_lovasz`（LovaszSoftmaxLoss, w=1.0）+ `loss_occ_sem_scal`（SemScalLoss, w=1.0） |

开关：删掉 `aux_loss_occ` 中对应 dict 或把 `loss_weight` 设为 0 即关闭；可选加
`GeoScalLoss`（`non_empty_idx` 默认 0，对齐 FlashOCC 可改 17）。

---

## 5. 遗留事项与下一步

- [ ] **用修复后代码重启训练**，务必指定独立 `work_dir`（如 `work_dirs/fastbev-occ-r50-cbgs-24e-c`），避免污染 50e 目录 `latest.pth`；
- [ ] 待训练日志确认：`load_from = epoch_50_ema.pth` 正常加载、无 DDP reduction 报错；
- [ ] 可选对照实验：B（`detach=False`）vs C（`detach=True`，验证 P2-4 抢占假设）；
- [ ] 24e 快速验证 mIoU 趋势后，扩回 50e 终训：`max_epochs=50`、`step=[24,40]`；
- [ ] 数据增强对齐 FlashOCC 严格值 `(-0.06,0.11)/(-5.4,5.4)`（当前为折中值）为可选微调项。

---

## 附：关键文件路径

| 用途 | 路径 |
|---|---|
| 检测器（含 detach 修复） | `mmdet3d/models/detectors/fastbev.py` |
| occ 头（含 aux loss） | `mmdet3d/models/dense_heads/bev_occ_head.py` |
| 辅助损失实现 | `mmdet3d/models/losses/occ_aux_loss.py` |
| 训练配置 | `configs/fastbev/paper/fastbev-occ-r50-cbgs.py` |
| 架构文档 | `docs/architecture.md` |

---

# 追加记录：FastBEV vs FlashOCC 效果对比与差异分析（同一会话后续）

## 6. 效果对比结论（同口径）

| 指标 | FastBEV v1（50e 联合训练） | FlashOCC 官方 M1（24e，flashocc-r50-256x704.pth） | 差距 |
|---|---|---|---|
| occ mIoU（17 类不含 free，6019 val） | **27.4** | **32.08** | **-4.68（相对 -14.6%）** |
| 检测 NDS / mAP | 0.388 / 0.288 | 无检测分支 | — |

**结论：相对差点**（不是差不多）。但注意两点：
1. FastBEV 是检测 + occ 联合训练（额外拥有 NDS 0.388 / mAP 0.288 的检测能力），FlashOCC 是纯 occ 模型，本身不是完全对等对比；
2. 27.4 是 **v1 旧架构**（0.8m BEV + grid_sample 插值、无 aux_loss、无 detach）50e 训练的结果（用 v2 配置 + 50e EMA 权重测试，`det_downsample` 随机初始化）；**当前正在跑的 v2（0.4m BEV + Lovász/sem_scal + detach=True）24e 结果尚未出**。

### 6.1 逐类 IoU 对比（FastBEV 27.4 vs FlashOCC 32.08）

| 类别 | FastBEV | FlashOCC | 差值 | 类别 | FastBEV | FlashOCC | 差值 |
|---|---|---|---|---|---|---|---|
| others | 3.33 | 6.74 | -3.4 | driveable_surface | 73.55 | 78.82 | -5.3 |
| barrier | 39.02 | 37.65 | **+1.4** | other_flat | 28.44 | 37.98 | **-9.5** |
| bicycle | 11.23 | 10.26 | **+1.0** | sidewalk | 39.31 | 48.70 | **-9.4** |
| bus | 29.95 | 39.55 | **-9.6** | terrain | 44.30 | 52.50 | **-8.2** |
| car | 45.16 | 44.36 | **+0.8** | manmade | 29.46 | 37.89 | **-8.4** |
| construction_vehicle | 9.63 | 14.88 | -5.3 | vegetation | 22.81 | 32.24 | **-9.4** |
| motorcycle | 13.82 | 13.40 | **+0.4** | trailer | 20.04 | 27.44 | -7.4 |
| pedestrian | 14.35 | 15.79 | -1.4 | truck | 26.99 | 31.73 | -4.7 |
| traffic_cone | 14.38 | 15.38 | -1.0 | | | | |

规律：**车辆小类（barrier/bicycle/car/motorcycle）略好，场景类（路面/人行道/地形/建筑/植被）与 bus/trailer 大车类全面落后 5~9.6 个点**。

## 7. 配置差异对比表（flashocc-r50.py vs fastbev-occ-r50-cbgs.py）

| 维度 | FlashOCC flashocc-r50.py | FastBEV fastbev-occ-r50-cbgs.py (v2) |
|---|---|---|
| 模型类型 | BEVDetOCC（纯 occ，无检测） | FastBEVOCC（检测 + occ 联合） |
| 视图变换 | LSSViewTransformer（`collapse_z=True`，单层 BEV） | FastrayTransformer（z 网格 8 层 sum 融合） |
| z 网格 | `z=[-1, 5.4, 6.4]`（collapse 后 1 层） | `z=[-2.5, 4.5, 1.0]`（8 层） |
| depth 网格 | `depth=[1, 45, 0.5]` → **88 bins**（0.5m 精细） | `depth=[1, 60, 1.0]` → 59 bins（1.0m） |
| BEV/occ 分辨率 | 200x200 @ 0.4m | 200x200 @ 0.4m（v2 已对齐） |
| 检测分支 | 无 | CenterHead（10 类）+ `det_downsample` |
| occ 头 | BEVOCCHead2D（仅 CE loss） | FastBEVOCCHead（CE + Lovász + sem_scal） |
| 数据增强 | `resize(-0.06,0.11)` / `rot(-5.4,5.4)` | `resize(-0.10,0.15)` / `rot(-6,6)`（折中） |
| BEV 增强 | LoadAnnotationsBEVDepth 内置 bda（rot/scale=0 仅 flip） | BEVAug（flip-only） |
| 过滤 | 无 | ObjectRangeFilter / ObjectNameFilter |
| 采样/bs | 普通采样 / samples_per_gpu=4 | 普通采样 / samples_per_gpu=4 |
| optimizer | AdamW 1e-4, wd 1e-2, grad_clip 5 | 同 |
| lr/epoch | warmup 200, step=[24], 24e | 同 |
| EMA | MEGVIIEMAHook init_updates=10560 | 同 |
| fp16 | 关 | 关 |
| load_from | `flashocc-r50-256x704.pth`（BEVDet 检测预训练，深度监督已训好） | `epoch_50_ema.pth`（自己 v1 联合训练 EMA，occ 分支仅 27.4 水平） |
| 评估 | interval=1, start=20, 6019 val | 同 |

## 8. 训练流程对比

**FlashOCC（24e 达 32.08）：**
`ResNet50 → CustomFPN → LSSViewTransformer(深度估计, collapse_z) → CustomResNet → FPN_LSS → BEVOCCHead2D`
- pipeline：PrepareImageInputs → LoadAnnotationsBEVDepth → LoadOccGTFromFile → LoadPointsFromFile → PointToMultiViewDepth → DefaultFormatBundle3D → Collect3D（无检测过滤）
- loss：`loss_depth`（LSS 深度监督）+ `loss_occ`（CE, ignore=255, mask_camera 加权）
- 初始化：从 BEVDet 检测预训练权重加载（含已训好的深度估计网络），纯 occ 微调，**无任务竞争**

**FastBEVOCC（v1 50e 达 27.4 / v2 24e 训练中）：**
`ResNet50 → CustomFPN → FastrayTransformer(8 层 z sum) → CustomResNet → FPN_LSS → 分支：CenterHead(det_downsample) + FastBEVOCCHead`
- pipeline：PrepareImageInputs → LoadAnnotations → LoadOccGTFromFile → BEVAug(flip) → LoadPointsFromFile → PointToMultiViewDepth → ObjectRangeFilter → ObjectNameFilter → DefaultFormatBundle3D → Collect3D
- loss：`loss_depth` + 检测 5 项（heatmap/xy/z/whl/yaw/vel）+ `loss_occ` CE + `loss_occ_lovasz` + `loss_occ_sem_scal`
- 初始化：上一轮 v1 联合训练 EMA（occ 分支 27.4 水平），检测 + occ **共享 BEV 特征、存在任务竞争**（v2 已用 detach=True 消融验证）

## 9. 优化建议（按优先级）

| 优先级 | 方向 | 做法 | 预期 |
|---|---|---|---|
| P0 | **occ 分支初始化对齐 FlashOCC** | 用 `tools/merge_fastbev_occ_weights.py` 生成 init_merged（检测分支=FastBEV 权重，occ_head=FlashOCC 官方权重），`load_from` 改用它 | 把 occ 起点从 27.4 提到 32.08 水平，预计 mIoU +2~4 |
| P1 | **等 v2 24e-c 结果** | 当前训练（detach=True + aux_loss + epoch_50_ema）约 17.5h，训练中评估 start=20，先看 v2 首个结果再定 | — |
| P1 | **深度 bin 细化** | depth 对齐 FlashOCC `[1,45,0.5]`（88 bins） | 深度估计更准 → 场景类几何提升；注意 depth_net 通道数变化，load_from 需重新预训练 |
| P2 | **数据增强严格对齐** | `resize/rot` → `(-0.06,0.11)/(-5.4,5.4)` | 减少对 occ 几何扰动，mIoU 看涨（检测 mAP 可能 -1~2） |
| P2 | **检测/occ loss 平衡** | 检测 5 项回归 loss 权重占大头，可试降低检测 `loss_weight`（0.25→0.1）或提高 occ CE 权重 | 缓解任务抢占，occ 看涨 |
| P2 | **扩训 50e** | v2 结果 mIoU 趋势好则 `max_epochs=50, step=[24,40]` | FlashOCC 24e 即 32.08，说明差距主要不在时长 |
| P3 | **加 GeoScalLoss** | aux 中加 `dict(type='GeoScalLoss', loss_weight=1.0)`（non_empty_idx=0） | 几何一致性辅助，mIoU +0.5~1 |
| P3 | **多任务梯度策略** | 若 detach=True 验证了抢占假设，可进一步试梯度缩放/分阶段训练（先 occ 后 det） | 长期方向 |

## 10. 当前训练状态（2026-08-24 15:44 起）

- 修复 detach 后已重启：`work_dirs/fastbev-occ-r50-cbgs-24e-c`（独立 work_dir ✓）
- 加载 `epoch_50_ema.pth` 正常：仅预期内告警（`voxel_coords` size mismatch 无害；`det_downsample` missing keys 随机初始化）
- 日志正常：loss 平稳下降（Epoch1 中段 loss 5.8~5.9，grad_norm ~20），无 DDP reduction 报错
- ETA 约 17.5h，24 个 epoch；evaluation start=20 起每 epoch 出 occ mIoU + NDS/mAP
