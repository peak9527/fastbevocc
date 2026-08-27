# Copyright (c) OpenMMLab. All rights reserved.
# FastBEV (3D 目标检测) + FlashOCC 风格语义占用头：训练/测试时同时输出
# BEV 3D 目标检测结果与 200x200x16 语义占用栅格。
#
# [v2 分支解耦版] 与 FlashOCC flashocc-r50.py 对齐 BEV/occ 分辨率：
#   - FastrayTransformer 直接生成 200x200 @ 0.4m 的 BEV 特征（覆盖 [-40,40]），
#     occ 头（FastBEVOCCHead）原生消费 0.4m 分辨率，不再做 0.8m->0.4m 插值。
#   - 检测分支通过 det_downsample（stride-2 卷积）下采样到 100x100
#     （0.8m/像素，与原检测物理分辨率一致；感知范围由 102.4m 缩至 80m）。
#   - 保留 z/depth 原值：改 z 会放大 voxel 数，改 depth 会改变 depth_net 输出
#     通道（D+out_channels），两者都能兼容加载 init_merged.pth 预训练权重。
#
# 依赖：
#   - data/nuscenes/bevdetv3-nuscenes_infos_{train,val}.pkl（含 occ_path 字段）
#   - data/nuscenes/gts/.../labels.npz（occ 标注，由 FlashOCC 数据生成）
#
# 训练（超参对齐 FlashOCC flashocc-r50.py：全量微调 / 24e / lr=1e-4 / step=[24] /
#       EMA / 普通采样 / ignore_index=255 / 关 fp16）：
#   bash tools/dist_train.sh configs/fastbev/paper/fastbev-occ-r50-cbgs.py 2
#   注意：0.4m BEV 计算/显存约为 0.8m 的 2.4 倍，每步时间约翻倍；
#   建议先按 24e 快速验证 mIoU 趋势，再决定是否扩训到 50e。
#
# ==== P2 低成本消融速查表（改动点均已做成配置开关） ====
#   P2-3 数据增强对齐：data_config.resize/rot 已改折中值（(-0.10,0.15)/(-6,6)），
#        想严格对齐 FlashOCC 改为 (-0.06,0.11)/(-5.4,5.4)；想恢复检测配方改为
#        (-0.15,0.25)/(-8.0,8.0)。影响：检测 mAP ∓1~2，occ mIoU 与之相反。
#   P2-4 任务竞争消融：det_downsample.detach 设 True（检测梯度不回流 BEV 特征）。
#        预期：occ mIoU 看涨（若抢占假设成立），检测 mAP 基本不变。
#   P2-5 loss 改进：occ_head.aux_loss_occ 已默认开启 CE+Lovász+sem_scal；
#        可加 GeoScalLoss 或删项对比。预期 occ mIoU +1~3，检测几乎无感。
#   P2-6 训练配方：下面对齐 FlashOCC 的 24e 快速版（max_epochs=24, step=[24]）；
#        验证 mIoU 真实差距后扩回 50e（step=[24,40]）。
# ======================================================================
# 测试（同时输出检测 + occ；推荐用 EMA 权重）：
#   bash tools/dist_test.sh configs/fastbev/paper/fastbev-occ-r50-cbgs.py \
#       work_dirs/fastbev-occ-r50-cbgs/epoch_24_ema.pth 2

_base_ = ['../../_base_/default_runtime.py']
# Global
# 检测范围对齐 0.4m BEV（[-40,40]）；检测 heatmap 经 det_downsample 后为
# 100x100（0.8m/像素，与原检测分辨率一致，仅感知范围 102.4m -> 80m）。
point_cloud_range = [-40, -40, -5.0, 40, 40, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation（P2-3 折中版：取检测配方与 FlashOCC 之间的中间值，
    # 保检测鲁棒性的同时减少对 occ 几何对齐的扰动；纯相机空间增强，
    # 通过 post_rot/post_tran 同步外参，不影响 occ/检测坐标一致性）
    #   检测配方(原):  resize (-0.15, 0.25), rot (-8.0, 8.0)
    #   FlashOCC 值:   resize (-0.06, 0.11), rot (-5.4, 5.4)
    'resize': (-0.10, 0.15),
    'rot': (-6.0, 6.0),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
# BEV 0.4m 分辨率（200x200，覆盖 [-40,40]），与 FlashOCC 的 occ 栅格对齐；
# FastBEVOCCHead 直接消费该分辨率（bev_size=occ_size，不再插值）。
grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-2.5, 4.5, 1.0],
    'depth': [1.0, 60.0, 1.0],
}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 64

model = dict(
    type='FastBEVOCC',
    use_depth=True,     # 深度监督（fastbev-r50-cbgs-custom.py 调试后检测效果提升的关键）
    # 分支解耦：BEV 为 200x200@0.4m 时，检测分支先经 stride-2 卷积下采样到
    # 100x100（0.8m/像素），CenterHead 的 heatmap/target 编码保持兼容；
    # occ 分支直接用 200x200 原生 0.4m 特征（不再插值）。
    # detach=False：检测梯度正常回流 BEV 特征（默认）。
    # detach=True ：检测特征 detach()，检测 loss 梯度不回流 BEV 特征——
    #               用于消融"检测任务抢占 BEV 特征导致 occ 变差"（P2-4），
    #               此时检测分支照常训练，仅共享特征不共享梯度。
    det_downsample=dict(
        in_channels=256, out_channels=256, detach=False),
    # 若显存紧张可改为 False；pipeline 始终生成 gt_depth，仅开关损失
    img_backbone=dict(
        pretrained='data/ckpts/resnet50-19c8e357.pth',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='FastrayTransformer',
        grid_config=grid_config,
        in_channels=256,
        out_channels=numC_Trans,
        image_size=[256, 704],
        feature_size=[16, 44],
        downsample=1,
        stride=16,
        fuse=dict(type='sum')),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=256,
        tasks=[
            dict(num_class=10, class_names=['car', 'truck',
                                            'construction_vehicle',
                                            'bus', 'trailer',
                                            'barrier',
                                            'motorcycle', 'bicycle',
                                            'pedestrian', 'traffic_cone',
                                            ]),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-50, -50, -10.0, 50, 50, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # FlashOCC 风格语义占用头
    occ_head=dict(
        type='FastBEVOCCHead',
        in_dim=256,
        out_dim=256,
        Dz=16,
        use_mask=True,
        num_classes=18,
        use_predicter=True,
        class_balance=False,
        # BEV 特征 200x200 (x/y: [-40, 40], 0.4m)，与 occ 输出同分辨率；
        # FastBEVOCCHead 内部检测到 bev_size==occ_size 后跳过 grid_sample。
        bev_size=(200, 200),
        bev_range=(-40.0, 40.0),
        # occ 输出 200x200 (x/y: [-40, 40], 0.4m)，与 nuScenes occ 标注对齐
        occ_size=(200, 200),
        occ_range=(-40.0, 40.0),
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            ignore_index=255,   # FlashOCC 相同：255(ignore) 体素不参与 loss 计数
            loss_weight=1.0),
        # P2-5 FlashOCC V2 风格辅助 loss（CE + Lovász + sem_scal），
        # 直接优化 mIoU，对车辆小目标/场景类边界友好。
        # 消融开关：删掉对应 dict 或把 loss_weight 设为 0 即关闭；完整组合：
        #   aux_loss_occ=[
        #       dict(type='LovaszSoftmaxLoss', loss_weight=1.0),
        #       dict(type='SemScalLoss', loss_weight=1.0),
        #       dict(type='GeoScalLoss', loss_weight=1.0, non_empty_idx=0),
        #   ]
        # 注：GeoScal 默认 non_empty_idx=0（nuScenes empty 类）；若想严格对齐
        # FlashOCC BEVOCCHead2D_V2（其传 non_empty_idx=17）可改为 17。
        aux_loss_occ=[
            dict(type='LovaszSoftmaxLoss', loss_weight=1.0),
            dict(type='SemScalLoss', loss_weight=1.0),
        ]),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[800, 800, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            # code=[dx,dy,z,w,l,h,sin,cos,vx,vy]
            # 参考 fastbev-r50-cbgs-custom.py：旋转(sin/cos)权重 3x 优先拟合航向，
            # 速度(vx/vy)权重 0.5 帮助动态障碍物
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                          3.0, 3.0, 0.5, 0.5])),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-50, -50, -10.0, 50, 50, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pre_max_size=1000,
            post_max_size=500,

            # Scale-NMS
            nms_type=['rotate'],
            nms_thr=[0.2],
            nms_rescale_factor=[[1.0, 0.7, 0.7, 0.4, 0.55,
                                 1.1, 1.0, 1.0, 1.5, 3.5]]
        )
    )
)

# Data
dataset_type = 'NuScenesDatasetOccpancy'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

# 注意：必须与 FlashOCC 原版一致，仅做 flip（rot/scale 保持默认）！
# BEVAug 只对 occ GT（voxel_semantics/mask_camera）同步 flip，
# 而 bda 的旋转/缩放会通过 FastrayTransformer 的 lidar_aug_matrix
# 变换 BEV 特征空间；检测 GT 在 BEVAug 中同步了旋转缩放，
# 但 occ GT 没有，若开启 rot/scale 会导致 occ 监督坐标系与预测不一致。
bda_aug_conf = dict(
    rot_lim=(0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config),
    dict(type='LoadAnnotations'),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='BEVAug',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_depth',
            'voxel_semantics', 'mask_lidar', 'mask_camera'
        ])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config),
    dict(type='LoadAnnotations'),
    dict(type='BEVAug',
         bda_aug_conf=bda_aug_conf,
         classes=class_names,
         is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet',
)

test_data_config = dict(
    pipeline=test_pipeline,
    data_root=data_root,
    ann_file=data_root + 'bevdetv3-nuscenes_infos_val.pkl')

data = dict(
    # 与 FlashOCC 完全一致：普通采样（非 CBGS），samples_per_gpu=4/GPU
    #（2 GPU 总 batch=8，EMA init_updates=10560 ≈ 3 epoch 与其训练步数一致）
    # 注意：0.4m BEV（200x200）显存约为 0.8m 的 2.4 倍，若 OOM 请降为 2。
    samples_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
        type='NuScenesDatasetOccpancy',
        data_root=data_root,
        ann_file=data_root + 'bevdetv3-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['train', 'val', 'test']:
    data[key].update(share_data_config)

# Optimizer（超参对齐 FlashOCC flashocc-r50.py）
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    # P2-6 / D 消融：对齐 FlashOCC 基准配方（24e 单次衰减）：
    #   lr=1e-4 直到 epoch 24 后降 10 倍，与 FlashOCC 32.08 同口径对比。
    # 验证完扩回 [24,40] + max_epochs=50 终训。
    step=[24])
runner = dict(type='EpochBasedRunner', max_epochs=24)  # D: 24e 快速版

# EMA（FlashOCC 相同；评估/测试请用 epoch_*_ema.pth）
custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=5280,
        priority='NORMAL',
    ),
]

# 与 FlashOCC 一致：不开 fp16（occ 训练对精度敏感）
# fp16 = dict(loss_scale='dynamic')

# 训练中评估 occ mIoU（FlashOCC: evaluation=dict(interval=1, start=20, ...)）
evaluation = dict(interval=1, start=20, pipeline=test_pipeline)
# 只保留最近 5 个 checkpoint（FlashOCC 相同，每个约 190MB，防止占满磁盘）
checkpoint_config = dict(interval=1, max_keep_ckpts=5)

# 不使用 fastbev-r50-cbgs.py 的 resume，防止占用头初始化错位。
resume_from = None
# resume_from = 'work_dirs/fastbev-occ-r50-cbgs/latest.pth'
load_from = 'data/ckpts/epoch_24_ema.pth'
# ---------------------------------------------------------------------------
# 预训练权重加载（推荐方式：合并 FastBEV 检测 + FlashOCC occ 两个权重，
# 让检测分支与 occ 分支都被初始化）：
#   python tools/merge_fastbev_occ_weights.py \
#       --fastbev-ckpt work_dirs/fastbev-r50-cbgs/epoch_20_ema.pth \
#       --flashocc-ckpt data/ckpts/flashocc-r50-256x704.pth \
#       --out work_dirs/fastbev-occ-r50-cbgs/init_merged.pth
# load_from = 'work_dirs/fastbev-occ-r50-cbgs/init_merged.pth'
# 合并权重包含 FastBEVOCC 的全部参数（检测分支 + occ_head），
# 键一一对应，无需 strict=False，也无需额外参数。
# ---------------------------------------------------------------------------
# 备选方案 1：只加载 FastBEV 检测权重（occ_head 随机初始化，从头学占用分支）：
#   load_from = 'work_dirs/fastbev-r50-cbgs/epoch_20_ema.pth'
#   （strict=False 自动忽略不存在的 occ_head 键；检测分支 479 键全部匹配，安全）
#
# 备选方案 2：只初始化 occ_head（检测分支随机）——不能用 FlashOCC 完整权重直接 load！
#   原因：FlashOCC 的 img_view_transformer.depth_net 与 FastBEV 的 FastrayTransformer
#   键名相同但 shape 不同（152 vs 123），load_state_dict 会直接 raise size mismatch。
#   必须先提取 occ_head 再加载：
#   python tools/merge_fastbev_occ_weights.py --only-occ-head \
#       --flashocc-ckpt data/ckpts/flashocc-r50-256x704.pth \
#       --out work_dirs/fastbev-occ-r50-cbgs/occ_head_init.pth
#   load_from = 'work_dirs/fastbev-occ-r50-cbgs/occ_head_init.pth'
