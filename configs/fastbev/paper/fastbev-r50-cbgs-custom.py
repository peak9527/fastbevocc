# Copyright (c) Phigent Robotics. All rights reserved.
#
# 自定义数据集训练配置
# 基于 fastbev-r50-cbgs.py，适配 SUSTechPOINTS 数据格式
#
# 用法:
#   ./tools/dist_train.sh configs/fastbev/paper/fastbev-r50-cbgs-custom-train.py 2

_base_ = ['./fastbev-r50-cbgs.py']

# ========== 覆盖数据集路径 ==========
# pkl 中路径为相对路径 (data/custom_data/...)，所以 data_root 设为本项目根目录
data_root = './'

# ========== 覆盖相机配置 ==========
# 注意: 不同相机分辨率不同 (frontmain=3840x2160, 其余≈1920x1536)
# src_size 设为最大分辨率，PrepareImageInputs 会自动按实际尺寸计算 resize 比例
data_config = dict(
    cams=['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
          'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    Ncams=6,
    input_size=(256, 704),
    src_size=(2160, 3840),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.00,
)

# ========== BEV 增强（与 nuScenes 一致） ==========
bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

# ========== 训练 pipeline ==========
train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config),
    dict(type='LoadAnnotations'),
    dict(
        type='BEVAug',
        bda_aug_conf=bda_aug_conf,
        classes=[
            'car', 'truck', 'cyclist', 'bus', 'cone',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,            # yk_c1 点云: x, y, z, intensity, ring, padding
        use_dim=4,             # 取前4维: x, y, z, intensity
        file_client_args=dict(backend='disk')),
    dict(
        type='PointToMultiViewDepth',
        downsample=1,
        grid_config=dict(
            x=[-51.2, 51.2, 0.8],
            y=[-51.2, 51.2, 0.8],
            z=[-2.5, 4.5, 1.0],
            depth=[1.0, 60.0, 1.0])),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'cyclist', 'bus', 'cone',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'cyclist', 'bus', 'cone',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='Collect3D',
        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_depth'])
]

# ========== 测试/验证 pipeline ==========
test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config),
    dict(type='LoadAnnotations'),
    dict(type='BEVAug',
         bda_aug_conf=bda_aug_conf,
         classes=[
            'car', 'truck', 'cyclist', 'bus', 'cone',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
         ],
         is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=4,
        file_client_args=dict(backend='disk')),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'cyclist', 'bus', 'cone',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
                ],
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]

# ========== 数据集配置 ==========
dataset_type = 'NuScenesDataset'
class_names = [
    'car', 'truck', 'cyclist', 'bus', 'cone',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
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
    ego_cam='CAM_FRONT',
)

# 训练 / 验证共用一个 pkl（数据量小，建议后续分 train/val）
ann_file = 'yk_c1_infos_val_3.pkl'

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=ann_file)

data = dict(
    samples_per_gpu=1,          # 自定义数据集帧数少，batch size 用 1
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            use_valid_flag=True,
            box_type_3d='LiDAR')),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'test']:
    data[key].update(share_data_config)
    data[key]['data_root'] = data_root

data['train']['dataset'].update(share_data_config)
data['train']['dataset']['data_root'] = data_root
# ========== 覆盖 pts_bbox_head 的 tasks，匹配自定义类别 ==========
# 注意: 必须覆盖 base config 的 pts_bbox_head.tasks，
# 否则 head 的 class_names 与自定义数据不匹配
model = dict(
    pts_bbox_head=dict(
        tasks=[
            dict(num_class=10, class_names=[
                'car', 'truck', 'cyclist', 'bus', 'cone',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
            ]),
        ],
    ),
    # ========== Backbone 训练模式 ==========
    # 全量微调: 注释下面 block 即可
    # img_backbone=dict(
    #     frozen_stages=4,
    #     norm_eval=True,
    #     norm_cfg=dict(type='BN', requires_grad=False),
    # ),
)

# ========== 训练设置 ==========
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
runner = dict(type='EpochBasedRunner', max_epochs=35)

# 禁用继承来的 resume_from，确保从头训练并仅加载预训练权重
resume_from = None
# 加载预训练权重 (transfer learning)
# resume_from = 'work_dirs/fastbev-r50-cbgs-custom/latest.pth'
load_from = 'work_dirs/fastbev-r50-cbgs-custom/epoch_20_nusenes_ema.pth'
