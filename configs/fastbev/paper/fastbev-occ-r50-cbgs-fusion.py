# Copyright (c) OpenMMLab. All rights reserved.
# FastBEVOCC 多模态融合（图像 + 激光雷达）训练配置
#
# 基于 fastbev-occ-r50-cbgs-custom.py，新增：
#   - 激光分支：PointPillars (PillarFeatureNet + PointPillarsScatter)
#   - BEV 融合：concat + 1x1 conv 降维（图像 256 + 激光 64 → 256）
#   - 检测头 / OCC 头均消费融合后的 BEV 特征
#
# 训练（2 GPU；若 OOM 把 samples_per_gpu 从 2 降到 1）：
#   bash tools/dist_train.sh configs/fastbev/paper/fastbev-occ-r50-cbgs-fusion.py 2
# 测试：
#   bash tools/dist_test.sh configs/fastbev/paper/fastbev-occ-r50-cbgs-fusion.py \
#       work_dirs/fastbev-occ-r50-cbgs-fusion/epoch_30_ema.pth 2

_base_ = ['./fastbev-occ-r50-cbgs-custom.py']

# ========== 输入模态 ==========
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

# ========== 模型：追加激光分支 + BEV 融合 ==========
model = dict(
    type='FastBEVOCC',
    # ---- 激光体素化 ----
    # voxel_size xy=0.4m 与图像 BEV 对齐（200x200），
    # z 方向取整个高度（pillar 是柱体），scatter 出来的 BEV 与图像分支空间一致
    pts_voxel_layer=dict(
        max_num_points=32,
        point_cloud_range=[-40.0, -40.0, -5.0, 40.0, 40.0, 3.0],
        voxel_size=[0.4, 0.4, 8.0],       # xy=0.4m（200 格），z 整层 = pillar
        max_voxels=(16000, 40000),        # (train, test)
        deterministic=False),
    # ---- 激光体素特征编码 ----
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,                     # x, y, z, intensity
        feat_channels=[64],
        with_distance=False,
        voxel_size=[0.4, 0.4, 8.0],
        point_cloud_range=[-40.0, -40.0, -5.0, 40.0, 40.0, 3.0],
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        legacy=False),
    # ---- 激光 scatter 成 BEV 伪图像 ----
    # output_shape = [200, 200] 与图像 BEV 完全对齐，无需重采样
    pts_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=[200, 200]),
    # ---- BEV 融合：concat + 1x1 conv 降维 ----
    # 图像 256ch + 激光 64ch = 320ch → 256ch（保持下游模块输入维度不变）
    fusion=dict(
        type='concat',
        in_channels=256 + 64,
        out_channels=256),
)

# ========== 数据 pipeline：把 points 加入 Collect3D ==========
# 原配置已用 LoadPointsFromFile 加载点云（用于深度监督），
# 但 train_pipeline 的 Collect3D 没包含 'points'，需要补上，
# 否则模型 forward 拿不到 points 参数。
train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config={{ _base_.data_config }}),
    dict(type='LoadAnnotations'),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='BEVAug',
        bda_aug_conf={{ _base_.bda_aug_conf }},
        classes={{ _base_.class_names }},
        point_cloud_range={{ _base_.point_cloud_range }},
        occ_voxel_size=[0.4, 0.4]),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=4,
        file_client_args={{ _base_.file_client_args }}),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config={{ _base_.grid_config }}),
    dict(type='ObjectRangeFilter', point_cloud_range={{ _base_.point_cloud_range }}),
    dict(type='ObjectNameFilter', classes={{ _base_.class_names }}),
    dict(type='DefaultFormatBundle3D', class_names={{ _base_.class_names }}),
    dict(
        type='Collect3D',
        keys=[
            'points',                        # 新增：点云输入模型
            'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_depth',
            'voxel_semantics', 'mask_lidar', 'mask_camera'
        ])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config={{ _base_.data_config }}),
    dict(type='LoadAnnotations'),
    dict(type='BEVAug',
         bda_aug_conf={{ _base_.bda_aug_conf }},
         classes={{ _base_.class_names }},
         is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=4,
        file_client_args={{ _base_.file_client_args }}),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names={{ _base_.class_names }},
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]

# ========== 数据配置 ==========
data = dict(
    train=dict(
        dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)

# 多模态融合显存占用更高，若 OOM 可取消下行注释把 batch 减半
# data['samples_per_gpu'] = 1
# custom_hooks[0]['init_updates'] = 21120
# lr_config['warmup_iters'] = 400
