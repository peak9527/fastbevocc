# Copyright (c) OpenMMLab. All rights reserved.
# FastBEVOCC 自定义数据（yk_c1）训练配置
#
# 融合两个参考配置：
#   - fastbev-occ-r50-cbgs.py   ：nuScenes FastBEVOCC 训练配方（0.4m BEV + occ 头）
#   - fastbev-r50-cbgs-custom.py：yk_c1 自定义数据检测训练（相机/输入/增强）
#
# 自定义数据要点：
#   - occ 真值：tools/create_occ_gt_custom.py 用固定 label_map 全量生成 labels.npz
#     （data/custom_data/yk_c1/label_map_zhito31.json，恒等映射 1~31），
#     semantics 值域 1..31（0=无标注/忽略；mask_lidar/mask_camera 评估+训练 mask），
#     已通过 --write-back 写回 yk_c1_infos_train.pkl 的 occ_path 字段。
#     注意：--label-map auto 是按升序重编号（数据缺 22 号会导致错位），正式数据
#     必须用固定映射表，不能再用 auto。
#   - 类别：zhito 标注规范 v1.2 = A 类目标 12 类（编号 1~12）+ B 类语义 19 类
#     （编号 13~31），共 31 类；num_classes=32（索引 0 为 ignore/无标注，1~31 为类别）。
#   - 检测标注：pkl 含 10 类 nuScenes 类别 3D 框，检测分支沿用 CenterHead。
#   - 评估：NuScenesDatasetOccpancy 新增 occ_num_classes / occ_class_names 参数
#     （默认 18 类），自定义数据必须传 33 与 33 个类别名（0=ignore、1~31 语义、32=free），
#     否则 Metric_mIoU 混淆矩阵 reshape 与 count_miou 打印会越界。
#
# 训练（2 GPU；samples_per_gpu=4，总 batch=8，每卡约 10GB）：
#   bash tools/dist_train.sh configs/fastbev/paper/fastbev-occ-r50-cbgs-custom.py 2
# 测试（推荐 EMA 权重）：
#   bash tools/dist_test.sh configs/fastbev/paper/fastbev-occ-r50-cbgs-custom.py \
#       work_dirs/fastbev-occ-r50-cbgs-custom/epoch_24_ema.pth 2
#
# 24e 快速验证：训练配置与 fastbev-occ-r50-cbgs-fusion-v3.py 完全对齐
# （同 batch=8、同 lr=1e-4、同 warmup=300、同衰减点 [18,21]、同 EMA 起点），
# 使「是否引入激光融合」成为两者唯一变量，便于公平对比检测与 OCC 指标。

_base_ = ['../../_base_/default_runtime.py']

# ========== 全局 ==========
# 检测范围对齐 0.4m BEV（[-40,40]），与 occ 栅格一致；occ 真值即 200x200x16@0.4m
point_cloud_range = [-40, -40, -5.0, 40, 40, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]

# ========== 相机（与 fastbev-r50-cbgs-custom.py 一致） ==========
data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (320, 896),
    'src_size': (2160, 3840),

    # 增强：resize 对齐官方 FastBEV fastbev-r50-cbgs.py（(-0.06,0.11)），
    # rot 保持 (-8,8)（官方 (-5.4,5.4)，取较大值增强旋转泛化）；
    # 纯相机空间增强，经 post_rot/post_tran 同步外参，不影响 occ 坐标一致性
    'resize': (-0.06, 0.11),
    'rot': (-8.0, 8.0),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# ========== BEV 栅格（0.4m，200x200，与 occ 真值对齐） ==========
# grid_config = {
#     'x': [-40, 40, 0.4],
#     'y': [-40, 40, 0.4],
#     'z': [-2.5, 4.5, 1.0],
#     'depth': [1.0, 60.0, 1.0],
# }
grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1.0, 5.4, 0.4],      # 与 create_occ_gt_custom.py 的 OCC_Z 对齐
    'depth': [1.0, 45.0, 0.5],  # 可保持 1m 粒度，或改成 [1.0, 45.0, 0.5]
}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 64

# ========== 模型 ==========
model = dict(
    type='FastBEVOCC',
    use_depth=True,
    # 分支解耦：BEV 200x200@0.4m，检测分支 stride-2 下采样到 100x100@0.8m；
    # occ 分支直接消费 0.4m 特征（bev_size==occ_size，head 内跳过 grid_sample）
    det_downsample=dict(
        in_channels=256, out_channels=256, detach=False),
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
        image_size=[320, 896],
        feature_size=[20, 56],
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
            dict(num_class=10, class_names=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone',
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
        # 方案 C：检测权重提升（loss_cls 1.0->1.5，loss_bbox 0.25->0.5），
        # 与 SemScal 0.5 组合，双方向微调、最稳。
        loss_cls=dict(
            type='GaussianFocalLoss', reduction='mean', loss_weight=1.5),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.5),
        norm_bbox=True),
    # 语义占用头：num_classes=33（仿照 nuScenes FlashOCC 18 类 = 0~16 语义 + 17=free）：
    #   索引 0=ignore/无标注(mask_camera 外)，1~31=zhito 31 类语义，32=free(空)。
    #   free 由 create_occ_gt_custom.py 自动填充（mask_camera 内无标注体素），
    #   参与训练；mIoU 评估排除 free（只统计前 num_classes-1 类）。
    occ_head=dict(
        type='FastBEVOCCHead',
        in_dim=256,
        out_dim=256,
        Dz=16,
        use_mask=True,
        num_classes=33,
        use_predicter=True,
        # 类别平衡 loss：对稀有语义类（摩托车/自行车等）加权，提升小类 mIoU
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
        # 辅助 loss（Lovasz/SemScal）对 33 类 mIoU 有明显提升，默认保留；
        # 若 samples_per_gpu=2 仍 OOM，把 aux_loss_occ 改为 []（纯 CE 训通再开）。
        aux_loss_occ=[
            dict(type='LovaszSoftmaxLoss', loss_weight=1.0),
            # 方案 C：SemScal 1.0->0.5，给检测让出梯度占比
            dict(type='SemScalLoss', loss_weight=0.5),
        ]),
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
            # code=[dx,dy,z,w,l,h,sin,cos,vx,vy]：旋转 3x、速度 0.5x
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
            nms_type=['rotate'],
            nms_thr=[0.2],
            nms_rescale_factor=[[1.0, 0.7, 0.7, 0.4, 0.55,
                                 1.1, 1.0, 1.0, 1.5, 3.5]]
        )
    )
)

# ========== 数据 ==========
dataset_type = 'NuScenesDatasetOccpancy'
data_root = './'  # pkl 中 lidar_path/occ_path 是相对项目根目录的路径
ann_file = 'yk_c1_infos_train.pkl'
file_client_args = dict(backend='disk')

# BEVAug 旋转/缩放增强：mmdet3d BEVAug 已扩展为用 bda 矩阵对 occ GT
# （voxel_semantics/mask_*）做最近邻重采样，与检测框变换严格一致，
# 因此可恢复 BDA 增强，提升检测分支的航向/尺度泛化，且不破坏 occ 坐标一致性。
# 采用官方 FastBEV 检测幅度（rot ±22.5°、scale 0.95~1.05），对 occ 干扰最小。
bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
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
        classes=class_names,
        point_cloud_range=point_cloud_range,
        occ_voxel_size=[0.4, 0.4]),
    # yk_c1 点云 6 列 bin: x,y,z,intensity,语义,实例；深度监督只用前 4 列
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=4,
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
        load_dim=6,
        use_dim=4,
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
    ego_cam='CAM_FRONT',
    # 自定义 occ 类别数 = 31 类语义 + ignore(0) + free(32) = 33；评估 Metric_mIoU 必须与之匹配。
    # occ_class_names 索引 0 = ignore（无标注/mask_camera 外），1~31 对应 zhito 标注规范 v1.2：
    # A 类 1~12：truck/tram/car/special_vehicle/cyclist/bicycle/tricycle/
    #            other_non_motor_vehicle/pedestrian/cone/other_object/barrier
    # B 类 13~31：road/other_ground/curb/vegetation/pole/trunk/manmade/fence/
    #            lifting_lever/noise_dust/noise_underground/noise_other/shadow/
    #            misc/background/helmet/carton/bottle/speed_bump
    # 索引 32=free(空)：仿照 nuScenes，mIoU 只统计前 32 类，free 不参与评估。
    occ_num_classes=33,
    occ_class_names=[
        'ignore', 'truck', 'tram', 'car', 'special_vehicle', 'cyclist',
        'bicycle', 'tricycle', 'other_non_motor_vehicle', 'pedestrian',
        'cone', 'other_object', 'barrier', 'road', 'other_ground', 'curb',
        'vegetation', 'pole', 'trunk', 'manmade', 'fence', 'lifting_lever',
        'noise_dust', 'noise_underground', 'noise_other', 'shadow', 'misc',
        'background', 'helmet', 'carton', 'bottle', 'speed_bump', 'free',
    ],
)

# 目前只有 train pkl；val 先复用同一 pkl 评估 occ mIoU（后续可换 val pkl）
test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=ann_file)

data = dict(
    # 0.4m BEV（200x200）+ 6 相机 320x896，显存约为 0.8m 的 2.4 倍。
    # 24GB 卡实测 samples_per_gpu=8 会 OOM（前向即占 ~20.7GB，backward 需额外分配），
    # samples_per_gpu=4 时每卡峰值约 9-10GB，余量充足。
    # 若换更大显存卡（如 40/80GB）可上调；仍 OOM 时把 occ_head 的 aux_loss_occ 置空。
    samples_per_gpu=4,
    workers_per_gpu=4,
    # CBGSDataset 类平衡采样：yk_c1 中 car 等大类占比高，CBGS 按类别密度
    # 过采样稀有类（motorcycle/bicycle/construction_vehicle 等），
    # 对检测分支小类 AP 提升显著，且不影响 occ 训练。
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type='NuScenesDatasetOccpancy',
            data_root=data_root,
            ann_file=ann_file,
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            use_valid_flag=True,
            box_type_3d='LiDAR')),
    val=test_data_config,
    test=test_data_config)

# 注意：train 是 CBGSDataset 包装，share_data_config 需 update 到内层 dataset，
# 否则会覆盖外层 type='CBGSDataset'，导致 CBGS 失效。
for key in ['val', 'test']:
    data[key].update(share_data_config)
    data[key]['data_root'] = data_root
data['train']['dataset'].update(share_data_config)
data['train']['dataset']['data_root'] = data_root

# ========== 优化器（24e 快速验证，epoch 18/21 两次衰减，各 0.1x） ==========
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=300,
    warmup_ratio=0.001,
    step=[18, 21])
runner = dict(type='EpochBasedRunner', max_epochs=24)

# EMA（评估/测试请用 epoch_*_ema.pth）
custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=5280,
        priority='NORMAL',
    ),
]

# 与 FlashOCC 一致：不开 fp16（occ 训练对精度敏感）
# fp16 = dict(loss_scale='dynamic')

# 每 epoch 评估 occ mIoU（start=20 后开始，前 20 epoch 省时）
evaluation = dict(interval=1, start=20, pipeline=test_pipeline)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)

resume_from = None
# 检测分支用已训练的自定义检测权重初始化（occ_head 随机初始化，
# mmcv load_checkpoint strict=False 自动忽略缺失的 occ_head 键）
load_from = 'data/ckpts/epoch_30_ema_custom.pth'
