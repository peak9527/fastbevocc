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
    input_size=(320, 896),          # 提高分辨率，(256,704)→(320,896)，保留更多细节
    src_size=(2160, 3840),
    resize=(-0.15, 0.25),           # 增大缩放增强范围
    rot=(-8.0, 8.0),                # 增大旋转增强范围
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.00,
)

# ========== BEV 增强（与 nuScenes 一致） ==========
bda_aug_conf = dict(
    rot_lim=(-45.0, 45.0),          # 增强 BEV 旋转，提升方位泛化
    scale_lim=(0.9, 1.1),           # 增大 BEV 缩放范围
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

# class_names 顺序必须与 base config (fastbev-r50-cbgs.py) 严格一致
# 注意: mmcv 的 _base_ 仅合并 dict，不会注入 Python 变量，
# 所以 class_names 必须在本文件内定义（且必须在 pipeline 之前，否则 NameError）。
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]

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
        classes=class_names),
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
        classes=class_names),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names),
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
         classes=class_names,
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
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]

# ========== 数据集配置 ==========
dataset_type = 'NuScenesDataset'

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
ann_file = 'yk_c1_infos_train.pkl'

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=ann_file)

data = dict(
    samples_per_gpu=4,          # 3090 24GB + FP16，batch=4 足够
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
# pts_bbox_head 继承自 base config，class_names 与 base 一致，无需覆盖
# 如需冻结 backbone，取消注释下面两行:
# model = dict(img_backbone=dict(frozen_stages=4, norm_eval=True,
#                                norm_cfg=dict(type='BN', requires_grad=False)))

# ========== 模型增强 ==========
model = dict(
    use_depth=True,                 # 开启深度监督，提升定位精度
    img_view_transformer=dict(
        image_size=[320, 896],      # 与 input_size 对齐，修复投影坐标偏差
        feature_size=[20, 56]),     # 320/16=20, 896/16=56
    # 增大旋转 loss 权重：code=[dx,dy,z,w,l,h,sin,cos,vx,vy]
    # 第 6、7 位是旋转 (sin/cos)，放大 3 倍让模型优先拟合航向
    # 第 8、9 位是速度 (vx/vy)，提高到 0.5 帮助弯道动态障碍物
    train_cfg=dict(
        pts=dict(
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 0.5, 0.5])),
)

# ========== 训练设置 ==========
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(                   # 新增学习率衰减策略
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[30, 40],)                 # epoch 30、40 各降 lr 到 1/10
runner = dict(type='EpochBasedRunner', max_epochs=100)  # 25→50 epochs

# FP16 混合精度（3090 原生支持，省一半显存、速度更快）
# fp16 = dict(loss_scale='dynamic')

# 验证与最佳模型保存
evaluation = dict(interval=10, save_best='pts/mAP')  # 每10 epoch验证，按mAP保存best
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=3,                                 # 只保留最近3个checkpoint
)

# EMA 指数滑动平均（稳定训练，提升泛化）
custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

# 禁用继承来的 resume_from，确保从头训练并仅加载预训练权重
resume_from = None
# 加载预训练权重 (transfer learning)
# resume_from = 'work_dirs/yk_c1_exp1/latest.pth'
load_from = 'data/ckpts/epoch_20_ema.pth'
