# Copyright (c) OpenMMLab. All rights reserved.
# FastBEV (3D 目标检测) + FlashOCC 风格语义占用头：训练/测试时同时输出
# BEV 3D 目标检测结果与 200x200x16 语义占用栅格。
#
# 依赖：
#   - data/nuscenes/bevdetv3-nuscenes_infos_{train,val}.pkl（含 occ_path 字段）
#   - data/nuscenes/gts/.../labels.npz（occ 标注，由 FlashOCC 数据生成）
#
# 训练：
#   bash tools/dist_train.sh configs/fastbev/paper/fastbev-occ-r50-cbgs.py 8
# 测试（同时输出检测 + occ）：
#   bash tools/dist_test.sh configs/fastbev/paper/fastbev-occ-r50-cbgs.py \
#       work_dirs/fastbev-occ-r50-cbgs/latest.pth 8

_base_ = ['../../_base_/default_runtime.py']
# Global
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
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

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-2.5, 4.5, 1.0],
    'depth': [1.0, 60.0, 1.0],
}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 64

model = dict(
    type='FastBEVOCC',
    use_depth=False,
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
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
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
        # BEV 特征 128x128 (x/y: [-51.2, 51.2], 0.8m)
        bev_size=(128, 128),
        bev_range=(-51.2, 51.2),
        # occ 输出 200x200 (x/y: [-40, 40], 0.4m)，与 nuScenes occ 标注对齐
        occ_size=(200, 200),
        occ_range=(-40.0, 40.0),
        loss_occ=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
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
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv3-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR')),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'test']:
    data[key].update(share_data_config)
data['train']['dataset'].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[20,])
runner = dict(type='EpochBasedRunner', max_epochs=100)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

# fp16 = dict(loss_scale='dynamic')

# 不使用 fastbev-r50-cbgs.py 的 resume，防止占用头初始化错位。
resume_from = None
# resume_from = 'work_dirs/fastbev-occ-r50-cbgs/latest.pth'
load_from = 'data/ckpts/init_merged.pth'
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
