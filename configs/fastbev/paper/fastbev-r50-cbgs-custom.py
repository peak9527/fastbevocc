# Copyright (c) Phigent Robotics. All rights reserved.
#
# 基于 fastbev-r50-cbgs.py 的自定义 6 相机数据集推理配置
# 相机命名与 nuScenes 一致: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT,
#                         CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
#
# 用法:
#   ./tools/dist_test.sh configs/fastbev/paper/fastbev-r50-cbgs-custom.py \
#       work_dirs/fastbev-r50-cbgs/epoch_20_ema.pth 1 \
#       --format-only \
#       --eval-options 'jsonfile_prefix=work_dirs/fastbev-r50-cbgs/results_custom'

# inherit base
_base_ = ['./fastbev-r50-cbgs.py']

# ========== 覆盖相机配置 ==========
# 使用 nuScenes 6 相机命名
data_config = dict(
    cams=['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
          'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    Ncams=6,
    input_size=(256, 704),
    src_size=(3840, 2160),       # 实际图像分辨率: 3840x2160
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.00,
)

# ========== 覆盖数据集路径 ==========
data_root = 'data/custom_data/'

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

# 测试 pipeline
# BEVAug 必须保留: 负责将 img_inputs 从 6 元组扩展为 7 元组 (加入 bda_mat)
# LoadAnnotations 必须保留: BEVAug 依赖 gt_bboxes_3d (测试模式下 bda=恒等矩阵)
test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config),
    dict(type='LoadAnnotations'),
    dict(type='BEVAug',
         bda_aug_conf=bda_aug_conf,
         classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
         ],
         is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,      # 自定义点云 4 字段 (x,y,z,intensity)
        use_dim=4,
        file_client_args=dict(backend='disk')),
    # sensor2lidar 在 pkl 中已是 RFU 等价形式，无需额外坐标转换
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
                ],
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file='./custom_infos_val.pkl',
)

# 重建 data dict (因为 _base_ 的 data dict 会被此文件的 data 完全替换)
# train 部分保留原样（不使用），test/val 用自定义
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    val=dict(
        pipeline=test_pipeline,
        ann_file='./custom_infos_val.pkl',
    ),
    test=dict(
        pipeline=test_pipeline,
        ann_file='./custom_infos_val.pkl',
    ),
)

# 共享的 dataset 元信息
share_data_config = dict(
    type='NuScenesDataset',
    ego_cam='CAM_FRONT',     # 必须与 pkl 中 cams 的 key 一致, format_results 用此相机做坐标变换
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    img_info_prototype='bevdet',
)

for key in ['val', 'test']:
    data[key].update(share_data_config)
    data[key]['data_root'] = data_root  # 覆盖 base config 继承的 'data/nuscenes/'

# 临时：把 score 阈值降到 0，验证模型是否产出任何预测
# 注意：_base_ 中 model 通过 deep-merge 合并，不能写 model.test_cfg.pts.xxx
# CenterPoint 有两层 score filter，必须同时覆盖！
#   bbox_coder.score_threshold → decode 阶段过滤 (此处是真正的过滤点)
#   test_cfg.pts.score_threshold → NMS 后处理过滤
model = dict(
    pts_bbox_head=dict(
        bbox_coder=dict(score_threshold=0.0)
    ),
    test_cfg=dict(
        pts=dict(score_threshold=0.0)
    )
)
