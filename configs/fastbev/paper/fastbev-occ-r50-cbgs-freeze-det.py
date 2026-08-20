# Copyright (c) OpenMMLab. All rights reserved.
# FastBEV (3D 目标检测) + FlashOCC 风格语义占用头 —— 方案 B：冻结检测分支。
#
# 与 fastbev-occ-r50-cbgs.py（方案 A，全量 fine-tune）的推理流程完全相同，
# 区别仅在训练时的参数更新范围：
#   - 方案 A：所有参数（backbone/neck/view_transformer/bev_encoder/检测头/occ_head）都更新
#   - 方案 B：冻结检测分支（img_backbone/img_neck/img_view_transformer/
#             img_bev_encoder_*/pts_bbox_head，requires_grad=False），
#             只训练 occ_head（final_conv + predicter）去适配 FastBEV 的 BEV 特征空间
#
# 方案 B 的实现：
#   - model.freeze_det_branch=True 触发 FastBEVOCC 冻结逻辑：
#       * 除 occ_head 外所有参数 requires_grad=False
#       * 训练时检测分支在 torch.no_grad() 下前向（不保存中间激活，省显存/加速）
#       * 只计算 occ 损失，不再计算检测损失
#       * 冻结分支内的 BN 保持 eval 模式（running stats 不更新）
#   - optimizer.constructor='RequiresGradOptimizerConstructor'：
#       优化器只包含 requires_grad=True 的参数（即 occ_head 的参数）
#
# 优势：显存占用少、反向传播更快，可尝试更大的 batch size（samples_per_gpu）。
#
# 依赖：
#   - data/nuscenes/bevdetv3-nuscenes_infos_{train,val}.pkl（含 occ_path 字段）
#   - data/nuscenes/gts/.../labels.npz（occ 标注，由 FlashOCC 数据生成）
#   - data/ckpts/init_merged.pth（FastBEV 检测 + FlashOCC occ 合并权重，
#     检测分支权重作为冻结初始值，occ_head 权重作为微调初始值）
#
# 训练：
#   bash tools/dist_train.sh configs/fastbev/paper/fastbev-occ-r50-cbgs-freeze-det.py 8
# 测试（同时输出检测 + occ，与方案 A 相同）：
#   bash tools/dist_test.sh configs/fastbev/paper/fastbev-occ-r50-cbgs-freeze-det.py \
#       work_dirs/fastbev-occ-r50-cbgs-freeze-det/latest.pth 8

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
    # 方案 B：冻结检测分支，只训练 occ_head
    freeze_det_branch=True,
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
    # FlashOCC 风格语义占用头（方案 B 中唯一可训练的模块）
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
    # 方案 B 显存占用更少，可尝试加大 batch size（如 6/8）
    samples_per_gpu=16,
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
# 方案 B：RequiresGradOptimizerConstructor 只把 requires_grad=True 的参数
# （即 occ_head 的 final_conv + predicter）放进优化器。
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=1e-2,
    constructor='RequiresGradOptimizerConstructor')
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[20,])
runner = dict(type='EpochBasedRunner', max_epochs=100)



fp16 = dict(loss_scale='dynamic')

# 加载 FastBEV 检测 + FlashOCC occ 合并权重：
# 检测分支权重作为冻结初始值，occ_head 权重作为微调初始值。
# resume_from = None
resume_from = 'work_dirs/fastbev-occ-r50-cbgs-freeze-det/latest.pth'
# load_from = 'data/ckpts/init_merged.pth'
# 若未生成合并权重，可退化为只加载 FastBEV 检测权重（occ_head 随机初始化）：
#   load_from = 'work_dirs/fastbev-r50-cbgs/epoch_20_ema.pth'
# 合并权重生成方式见 configs/fastbev/paper/fastbev-occ-r50-cbgs.py 末尾注释。
