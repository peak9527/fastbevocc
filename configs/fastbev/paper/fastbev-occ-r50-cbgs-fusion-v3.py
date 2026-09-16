# Copyright (c) OpenMMLab. All rights reserved.
# FastBEVOCC 多模态融合 v3（图像 + 激光雷达）训练配置
#
# 相比 v2 的改进（4 项优化）：
#   1. 增强体素特征：PillarFeatureNet 增加高度统计（z_max/z_min/z_mean/z_std）、
#      密度（点数归一化）、强度统计（i_mean/i_std），共 +7 维输入特征
#   2. 多层级融合（multi_level_fusion）：在 BEV backbone 的 3 个 stage 输出后
#      都做门控融合，再送入 FPN neck，激光信息更早参与特征提取
#   3. OCC 专用激光头（pts_occ_head）：从激光 BEV 特征直接预测 OCC，
#      与主 OCC head 可学习加权融合，训练时双路监督
#   4. 深度引导增强：深度 loss 权重 0.125 → 0.25（温和加权；
#      实测 1.0 会过度扰动预训练主干、拖慢检测/occ 收敛）
#
# 新分支用 paramwise_cfg 给 5x 学习率（1e-4 → 5e-4）：激光分支/融合/OCC 激光头
# 均为随机初始化，与来自基线预训练权重的视觉主干分开训练。
#
# 训练（2 GPU，samples_per_gpu=4 → 总 batch=8，每卡峰值约 19GB/24GB；若 OOM 降到 2）：
#   bash tools/dist_train.sh configs/fastbev/paper/fastbev-occ-r50-cbgs-fusion-v3.py 2
# 测试：
#   bash tools/dist_test.sh configs/fastbev/paper/fastbev-occ-r50-cbgs-fusion-v3.py \
#       work_dirs/fastbev-occ-r50-cbgs-fusion-v3/epoch_24_ema.pth 2

_base_ = ['./fastbev-occ-r50-cbgs-custom.py']

# ========== 输入模态 ==========
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

# ========== 模型：4 项优化全量集成 ==========
model = dict(
    type='FastBEVOCC',
    # ---- 优化 4：深度引导增强，提升深度 loss 权重 ----
    img_view_transformer=dict(
        type='FastrayTransformer',
        loss_depth_weight=0.25,              # 1.0 过强会扰动预训练主干 → 0.25（默认 0.125 的 2 倍）
    ),
    # ---- 激光体素化 ----
    pts_voxel_layer=dict(
        max_num_points=32,
        point_cloud_range=[-40.0, -40.0, -5.0, 40.0, 40.0, 3.0],
        voxel_size=[0.4, 0.4, 8.0],          # xy=0.4m（200 格），z 整层 = pillar
        max_voxels=(30000, 60000),
        deterministic=False),
    # ---- 优化 1：增强体素特征 ----
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,                        # x, y, z, intensity
        feat_channels=[64, 128],              # 两层 MLP，输出 128ch
        with_distance=False,
        with_cluster_center=True,
        with_voxel_center=True,
        with_height_stat=True,                # +4 维：z_max/z_min/z_mean/z_std
        with_density=True,                    # +1 维：点数归一化密度
        with_intensity_stat=True,             # +2 维：i_mean/i_std
        voxel_size=[0.4, 0.4, 8.0],
        point_cloud_range=[-40.0, -40.0, -5.0, 40.0, 40.0, 3.0],
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        legacy=False),
    # ---- 激光 scatter 成 BEV 伪图像 ----
    pts_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=128,
        output_shape=[200, 200]),
    # ---- 激光 BEV backbone（3 stage，输出多尺度用于多层级融合） ----
    pts_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=128,
        num_layer=[2, 2, 2],
        num_channels=[128, 256, 512],
        stride=[2, 2, 2],
        backbone_output_ids=[0, 1, 2],       # 输出全部 3 层（多层级融合用）
        norm_cfg=dict(type='BN'),
        with_cp=False),                      # 关闭 checkpoint，避免 DDP + 多路径消费的兼容性问题
    # ---- 激光 BEV neck（FPN 融合多尺度） ----
    pts_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=512 + 128,
        out_channels=256,
        scale_factor=4,
        input_feature_index=(0, 2),
        extra_upsample=2,                     # 200x200 输出
    ),
    # ---- 优化 2：多层级融合（每个 stage 后都做门控融合） ----
    multi_level_fusion=True,
    # ---- 门控融合（最后一层也做，与多层级融合互补） ----
    fusion=dict(
        type='gated',
        in_channels_img=256,
        in_channels_pts=256,
        out_channels=256,
        hidden_channels=128),
    # ---- 方案 A：反向融合（双向融合）----
    # 正向：激光几何 → 图像语义分支；反向：融合特征（含图像语义）→ 调制激光
    # BEV 特征，再送入 pts_occ_head。激光有几何但缺语义、图像有语义但缺几何，
    # 双向互补，重点提升激光 OCC 分支的语义判别能力。
    # hidden_channels=64（正向 gate 为 128）：额外算力约 +6 GMACs/样本。
    reverse_fusion=dict(
        type='gated',
        in_channels_pts=256,       # 激光 BEV 特征 pts_bev_final
        in_channels_img=256,       # 融合 BEV 特征（含图像语义）
        out_channels=256,
        hidden_channels=64),
    # ---- 优化 3：OCC 专用激光头 + 可学习融合权重 ----
    # 训练：主 OCC 与激光 OCC 按 alpha = sigmoid(occ_fuse_logit) 先融合，
    #       融合结果送主 occ_head.loss（CE + aux）→ 梯度回传到 alpha，
    #       由网络自动学习"图像 OCC / 激光 OCC"的最优加权比例；
    #       激光分支另有 0.5 权重辅助监督，保证自身可训练。
    pts_occ_head=dict(
        type='FastBEVOCCHead',
        in_dim=256,
        out_dim=256,
        Dz=16,
        use_mask=True,
        num_classes=33,
        use_predicter=True,
        class_balance=True,
        bev_size=(200, 200),
        bev_range=(-40.0, 40.0),
        occ_size=(200, 200),
        occ_range=(-40.0, 40.0),
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            ignore_index=255,
            loss_weight=1.0),
        aux_loss_occ=None),                  # 激光 OCC 只用 CE，省显存
    # 可学习融合权重 alpha 的初始值（主 OCC 0.7 / 激光 OCC 0.3）
    occ_fusion_weight=0.7,
)

# ========== 数据 pipeline ==========
# 顺序要点（对齐官方多模态参考实现 configs/dal/dal-base.py）：
#   1) LoadPointsFromFile 必须在 BEVAug **之前**。BEVAug 内部是
#      `if 'points' in results:` 才会把 bda（旋转/缩放/flip/平移）同步作用到点云；
#      若点云在 BEVAug 之后才加载，该分支永远不执行 → 图像 BEV 处于 bda 坐标系
#      而激光 BEV 停留在原始 lidar 坐标系，两分支空间错位，融合失效。
#   2) 点云已被 bda 增强 → 深度图必须用 PointToMultiViewDepthFusion：它内部先做
#      bda 逆变换把点云还原回原始 lidar 坐标系再投影到图像；继续用普通版
#      PointToMultiViewDepth 会得到与图像不对齐的深度监督。
#   3) 激光专用增强（丢点/强度扰动）放在深度图生成**之后**：深度真值仍由完整
#      点云产生，丢点与强度扰动只影响送入模型的点云。
train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config={{ _base_.data_config }}),
    dict(type='LoadAnnotations'),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=4,
        file_client_args={{ _base_.file_client_args }}),
    dict(
        type='BEVAug',
        bda_aug_conf={{ _base_.bda_aug_conf }},
        classes={{ _base_.class_names }},
        point_cloud_range={{ _base_.point_cloud_range }},
        occ_voxel_size=[0.4, 0.4]),
    dict(type='PointToMultiViewDepthFusion', downsample=1,
         grid_config={{ _base_.grid_config }}),
    # ---- 激光专用增强（新增）----
    # 随机丢点：模拟点云稀疏/遮挡，提升激光分支鲁棒性
    dict(type='RandomPointDrop', drop_ratio=0.15, min_points=1000),
    # 强度扰动：加性噪声 + 乘性缩放 + 2% 概率整帧强度缺失。
    # 参数依据本数据集实测 intensity 分布（中位数 7 / 均值 15.3 / 99 分位 81）：
    # 噪声用绝对刻度且取较小值，避免淹没暗点信号；整帧置零概率保持低位。
    dict(
        type='RandomIntensityJitter',
        jitter_std=2.0,
        clip_range=6.0,
        scale_range=(0.9, 1.1),
        zero_ratio=0.02),
    # 打乱点序：避免体素内 max_num_points 截断偏向采集顺序靠前的点
    dict(type='PointShuffle'),
    dict(type='ObjectRangeFilter', point_cloud_range={{ _base_.point_cloud_range }}),
    dict(type='ObjectNameFilter', classes={{ _base_.class_names }}),
    dict(type='DefaultFormatBundle3D', class_names={{ _base_.class_names }}),
    dict(
        type='Collect3D',
        keys=[
            'points',
            'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_depth',
            'voxel_semantics', 'mask_lidar', 'mask_camera'
        ])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config={{ _base_.data_config }}),
    dict(type='LoadAnnotations'),
    # 与 train_pipeline 保持一致：点云先加载，BEVAug 内的点云分支才会执行。
    # 测试时 is_train=False → bda 恒为单位矩阵，点云数值不变。
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=4,
        file_client_args={{ _base_.file_client_args }}),
    dict(type='BEVAug',
         bda_aug_conf={{ _base_.bda_aug_conf }},
         classes={{ _base_.class_names }},
         is_train=False),
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
# 注意：基础配置 fastbev-occ-r50-cbgs-custom.py 里 share_data_config 已用
# `modality=input_modality`（当时 use_lidar=False）完成 dataset 注入，而配置继承
# 不会回溯更新 share_data_config，因此这里必须重新显式传入本文件的 input_modality
# （use_lidar=True），否则 dataset.modality 仍是 use_lidar=False。
# 说明：对本仓库的 NuScenesDataset 而言，get_data_info() 中 pts_filename 是无条件
# 设置的（只有 use_camera 影响相机分支），LoadPointsFromFile 读取的是 results
# ['pts_filename']，故 use_lidar=False 并不会阻断点云输入；此处的显式传递是为了
# 让 modality 语义与实际输入一致（也影响 nusc_submissions['meta'] 等元信息）。
data = dict(
    samples_per_gpu=4,                     # 实测每卡峰值 ~19GB/24GB（OOM 则降为 2）
    workers_per_gpu=2,
    train=dict(
        dataset=dict(
            pipeline=train_pipeline,
            modality=input_modality)),
    val=dict(
        pipeline=test_pipeline,
        modality=input_modality),
    test=dict(
        pipeline=test_pipeline,
        modality=input_modality),
)

# ========== 训练配置 ==========
# 与基线 fastbev-occ-r50-cbgs-custom.py 对齐（同 batch=8、基础 lr=1e-4、warmup=300、
# 衰减点 [18,21]、24 epoch），但新分支用 5x 学习率（见 paramwise_cfg），
# 因为它们是随机初始化，而视觉主干/occ_head 来自基线预训练权重。
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=1e-2,
    # 新分支 lr_mult=5.0（1e-4 -> 5e-4）：激光体素编码/骨干/neck、跨模态融合、
    # OCC 专用激光头、可学习融合权重全部随机初始化，用基础 lr 会欠拟合。
    # 实测缺失该设置时 pts 分支收敛很慢；其余参数（视觉主干、occ_head）保持 1e-4。
    paramwise_cfg=dict(
        custom_keys={
            'pts_voxel_encoder': dict(lr_mult=5.0),
            'pts_middle_encoder': dict(lr_mult=5.0),
            'pts_bev_encoder_backbone': dict(lr_mult=5.0),
            'pts_bev_encoder_neck': dict(lr_mult=5.0),
            'pts_occ_head': dict(lr_mult=5.0),
            'img_proj': dict(lr_mult=5.0),
            'pts_proj': dict(lr_mult=5.0),
            'gate_net': dict(lr_mult=5.0),
            'level_fusions': dict(lr_mult=5.0),
            # 方案 A 反向融合（随机初始化，同样给 5x）
            'rev_pts_proj': dict(lr_mult=5.0),
            'rev_img_proj': dict(lr_mult=5.0),
            'rev_gate_net': dict(lr_mult=5.0),
            'occ_fuse_logit': dict(lr_mult=5.0),
        }))
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=300,                       # epoch1 共 2519 iter，warmup 占 ~12%
    warmup_ratio=0.001,
    step=[18, 21])
runner = dict(type='EpochBasedRunner', max_epochs=24)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=5280,
        priority='NORMAL',
    ),
]

evaluation = dict(interval=1, start=20, pipeline=test_pipeline)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)

# 用基线模型权重初始化（激光分支随机初始化）
load_from = 'work_dirs/fastbev-occ-r50-cbgs-custom/epoch_24_ema.pth'
resume_from = None
