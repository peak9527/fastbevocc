#!/usr/bin/env python3
"""
可视化自定义数据（yk_c1，zhito 标注规范 v1.2 共 31 类）FastBEVOCC 推理结果。

本脚本融合两个参考脚本：
  - 显示方式 / 坐标系转换：参考 tools/vis_fastbev_occ_gt_custom_3d.py
      * occ 维度排列 dim0=X(前向)、dim1=Y(左向)、dim2=Z，与
        create_occ_gt_custom.py 的 OCC_SHAPE=(Dx,Dy,Dz) 一致；
      * occ 真值/预测均在 ego 系栅格生成，用 occ_centers_lidar() 逆变换
        回激光(lidar)系，与 lidar 点云 / GT 框同系显示（col=-y、row=-x，
        x 前朝上、y 左朝左）；
      * zhito 31 类颜色表（32 行，索引 0=ignore 无标注/空，FREE_ID=0）；
      * lidar bin 为 6 列 (x,y,z,intensity,semantic,instance) 取前 3 列；
      * 相机分辨率不统一，统一 resize 到 1600x900 后拼接。
  - 显示因素 / 布局：参考 tools/vis_fastbev_occ_results_3d.py
      * 输入 test.py --out 的 results_occ.pkl + infos pkl；
      * 预测框从 ego 系转回 lidar 系（FastBEV 检测输出在 ego 系）；
      * 布局：上方 FRONT_LEFT/FRONT/FRONT_RIGHT 三相机拼接，
        中间 BEV 左右分块（左=GT occ + GT 框，右=Pred occ + 全部框），
        下方 BACK_LEFT/BACK/BACK_RIGHT 三相机拼接（水平镜像）；
      * 颜色：预测框=(0,255,255) 青，GT 框=(255,255,0) 黄；
      * occ BEV color_mode：top（默认，俯视投影）/ bottom / majority；
      * Open3D 真 3D：occ 体素网格(ply) + lidar 点云 + 3D 框，
        支持 --occ-3d-max-z 截断高空误检、png/ply 离屏导出。

用法（conda bev 环境，需在项目根目录运行）：
    source /home/server/miniconda3/etc/profile.d/conda.sh && conda activate bev

    # 生成结果 pkl（test.py --out）
    python tools/test.py configs/fastbev/paper/fastbev-occ-r50-cbgs-custom.py \
        work_dirs/fastbev-occ-r50-cbgs-custom/epoch_30_ema.pth \
        --eval bbox --out work_dirs/fastbev-occ-r50-cbgs-custom/results_occ.pkl

    # 主流程图片：前 10 帧（GT/Pred occ 左右分块对比）
    python tools/vis_fastbev_occ_results_custom_3d.py \
        work_dirs/fastbev-occ-r50-cbgs-custom/results_occ.pkl \
        --pkl yk_c1_infos_train.pkl \
        --frames 0-9 --draw-gt \
        --save-path work_dirs/vis_results_yk

    # Open3D 真 3D：离屏导出 png + occ 体素 ply（headless 可用）
    python tools/vis_fastbev_occ_results_custom_3d.py \
        work_dirs/fastbev-occ-r50-cbgs-custom/results_occ.pkl \
        --pkl yk_c1_infos_train.pkl \
        --frames 0-199 --draw-gt --occ-3d-export --occ-3d-lidar --occ-3d-boxes \
        --occ-3d-save-path work_dirs/vis_results_3d_1

    # 输出视频
    python tools/vis_fastbev_occ_results_custom_3d.py \
        work_dirs/fastbev-occ-r50-cbgs-custom/results_occ.pkl \
        --pkl yk_c1_infos_train.pkl \
        --frames 0-199 --draw-gt --format video --fps 10 \
        --save-path work_dirs/vis_results_yk
"""

import argparse
import os
import pickle

import cv2
import numpy as np
import open3d as o3d
from pyquaternion.quaternion import Quaternion

from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes as LB

# ------------------------- 常量 -------------------------

# zhito 标注规范 v1.2：0=无标注/ignore(mask_camera 外)，1~12 A 类目标，13~31 B 类语义，
# 32=free(空，仿照 nuScenes FlashOCC 18 类中的 free 类)
OCC_CLASS_NAMES = [
    'ignore',                       # 0
    'truck', 'tram', 'car', 'special_vehicle', 'cyclist', 'bicycle',
    'tricycle', 'other_non_motor_vehicle', 'pedestrian', 'cone',
    'other_object', 'barrier',      # 1~12 A 类
    'road', 'other_ground', 'curb', 'vegetation', 'pole', 'trunk',
    'manmade', 'fence', 'lifting_lever', 'noise_dust', 'noise_underground',
    'noise_other', 'shadow', 'misc', 'background', 'helmet', 'carton',
    'bottle', 'speed_bump',         # 13~31 B 类
    'free',                         # 32 空/自由空间（深灰，参与训练、不参与 mIoU）
]
assert len(OCC_CLASS_NAMES) == 33, len(OCC_CLASS_NAMES)

# 33 类语义颜色（0=无标注黑，1~12 目标用鲜艳色，13~31 语义色，32=free 深灰），
# 参照 nuScenes/Cityscapes 语义配色惯例（与 vis_fastbev_occ_gt_custom_3d.py 一致）
COLOR_MAP_RAW = np.array([
    [0, 0, 0],              # 0  ignore（无标注）
    [160, 32, 240],         # 1  truck
    [0, 192, 255],          # 2  tram
    [0, 150, 245],          # 3  car
    [0, 255, 255],          # 4  special_vehicle
    [140, 140, 140],        # 5  cyclist
    [255, 192, 203],        # 6  bicycle
    [180, 120, 200],        # 7  tricycle
    [200, 180, 0],          # 8  other_non_motor_vehicle
    [255, 0, 0],            # 9  pedestrian
    [255, 240, 150],        # 10 cone
    [135, 60, 0],           # 11 other_object
    [255, 120, 50],         # 12 barrier
    [128, 64, 128],         # 13 road
    [200, 60, 60],          # 14 other_ground
    [244, 35, 232],         # 15 curb
    [107, 142, 35],         # 16 vegetation
    [153, 153, 153],        # 17 pole
    [220, 220, 0],          # 18 trunk
    [230, 230, 250],        # 19 manmade
    [190, 153, 153],        # 20 fence
    [255, 165, 0],          # 21 lifting_lever
    [180, 180, 180],        # 22 noise_dust
    [100, 100, 100],        # 23 noise_underground
    [150, 120, 90],         # 24 noise_other
    [50, 50, 80],           # 25 shadow
    [250, 170, 30],         # 26 misc
    [70, 130, 180],         # 27 background
    [255, 215, 0],          # 28 helmet
    [210, 180, 140],        # 29 carton
    [0, 255, 127],          # 30 bottle
    [255, 255, 255],        # 31 speed_bump
    [60, 60, 60],           # 32 free（深灰底，与 nuScenes 的 free 配色一致）
], dtype=np.uint8)


def _brighten_color(rgb, amount=0.25, min_lumi=90):
    """保色调提亮 0-255 RGB 颜色：与白色线性混合 + 保证最低明度。

    Open3D 光照与渲染会压暗颜色，统一提亮后 3D 网格、ply 与
    伪 3D 渲染中各类语义颜色更鲜明。
    """
    f = rgb.astype(np.float32)
    out = f + (255.0 - f) * amount
    lumi = out.max()
    if lumi < min_lumi:
        out = out + (min_lumi - lumi)
    return np.clip(out, 0, 255).astype(np.uint8)


# 渲染用的提亮版：各类颜色更亮，适配深色背景
COLOR_MAP = _brighten_color(COLOR_MAP_RAW)
# 索引 0 = 无标注/ignore，强制保持纯黑，使 BEV 贴图时能被 mask 过滤、3D 中不显示
COLOR_MAP[0] = 0

# 33 类体系：FREE_ID=32=free(空/自由空间)，0=ignore(无标注/mask 外)。
# 渲染时 free 与 ignore 均视为"空"（不生成 3D 网格、BEV 置透明），
# 但 free 是显式可学习类别（深灰），ignore 为不可观测区域（纯黑）。
FREE_ID = 32
EMPTY_ID = 0  # ignore：不可观测体素，与 free 一样不渲染


def _occ_empty(occ):
    """0=ignore(不可观测) 与 32=free(空) 均视为空体素（不显示/不生成网格）。"""
    return (occ == EMPTY_ID) | (occ == FREE_ID)
# occ 栅格几何常量；栅格尺寸由 occ 数组形状自适应
# （当前 0.2m 方案: 400x400x16，z ∈ [-0.5, 2.7]m）
OCC_RANGE = 40.0  # occ 栅格覆盖 [-40, 40]m
# 层 k 中心 z = k*res + res/2 + OCC_Z_BASE（与 create_occ_gt_custom.py 的
# OCC_Z=[-0.5,2.7,0.2] 一致；res 为各向同性 0.2m 栅格分辨率）
OCC_Z_BASE = -0.5


def _occ_res(occ):
    """occ (Dx,Dy,Dz) 的栅格分辨率：x/y 跨度 2*OCC_RANGE / Dx
    （0.2m @ Dx=400；0.4m @ Dx=200）。"""
    return 2.0 * OCC_RANGE / occ.shape[0]

# 相机显示顺序（与 vis.py 完全一致）
CAM_ORDER = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
]

# 颜色映射：0=GT(黄), 1=Pred(青)  —— 与 vis.py 的 color_map 一致
color_map = {0: (255, 255, 0), 1: (0, 255, 255)}

# BEV 框绘制边索引（与 vis.py 一致）
draw_boxes_indexes_bev = [(0, 1), (1, 2), (2, 3), (3, 0)]
draw_boxes_indexes_img_view = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5),
                               (5, 6), (6, 7), (7, 4), (0, 4), (1, 5),
                               (2, 6), (3, 7)]


# ------------------------- 工具函数 -------------------------

def depth2color(depth):
    gray = max(0, min((depth + 2.5) / 3.0, 1.0))
    max_lumi = 200
    colors = np.array(
        [[max_lumi, 0, max_lumi], [max_lumi, 0, 0], [max_lumi, max_lumi, 0],
         [0, max_lumi, 0], [0, max_lumi, max_lumi], [0, 0, max_lumi]],
        dtype=np.float32)
    if gray == 1:
        return tuple(colors[-1].tolist())
    num_rank = len(colors) - 1
    rank = np.floor(gray * num_rank).astype(int)
    diff = (gray - rank / num_rank) * num_rank
    return tuple(
        (colors[rank] + (colors[rank + 1] - colors[rank]) * diff).tolist())


def check_point_in_img(points, height, width):
    valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
    valid = np.logical_and(
        valid, np.logical_and(points[:, 0] < width, points[:, 1] < height))
    return valid


def lidar2img(points_lidar, camrera_info):
    points_lidar_homogeneous = \
        np.concatenate([points_lidar,
                        np.ones((points_lidar.shape[0], 1),
                                dtype=points_lidar.dtype)], axis=1)
    camera2lidar = np.eye(4, dtype=np.float32)
    camera2lidar[:3, :3] = camrera_info['sensor2lidar_rotation']
    camera2lidar[:3, 3] = camrera_info['sensor2lidar_translation']
    lidar2camera = np.linalg.inv(camera2lidar)
    points_camera_homogeneous = points_lidar_homogeneous @ lidar2camera.T
    points_camera = points_camera_homogeneous[:, :3]
    valid = np.ones((points_camera.shape[0]), dtype=bool)
    valid = np.logical_and(points_camera[:, -1] > 0.5, valid)
    points_camera = points_camera / points_camera[:, 2:3]
    camera2img = camrera_info['cam_intrinsic']
    points_img = points_camera @ camera2img.T
    points_img = points_img[:, :2]
    return points_img, valid


def occ_centers_lidar(infos, occ_shape):
    """occ 体素中心在激光(lidar)系的坐标 (Dx,Dy,Dz,3)。

    create_occ_gt_custom.py 生成 occ 时先把点云从 lidar 系转到 ego 系
    （lidar2ego）再体素化，因此 labels.npz 的体素坐标在 ego 系；
    FastBEV 的 occ 预测与 GT 同栅格（同为 ego 系），这里用 lidar2ego 的
    逆变换统一转回 lidar 系，与 GT 框 / lidar 点云同系显示。

    体素中心 (gx=X, gy=Y, gz=Z) 的 ego 系坐标（与 OCC_SHAPE=(Dx,Dy,Dz) 对齐）：
        x_ego = (gx + 0.5) * res - OCC_RANGE
        y_ego = (gy + 0.5) * res - OCC_RANGE
        z_ego = (gz + 0.5) * res + OCC_Z_BASE    # OCC_Z_BASE=-0.5（z∈[-0.5,2.7]）
    lidar2ego: p_ego = R @ p_lidar + t  →  p_lidar = R.T @ (p_ego - t) = (p_ego - t) @ R

    Args:
        infos: 单帧 info dict（含 lidar2ego_rotation/translation）。
        occ_shape: (Dx, Dy, Dz)，与预测/GT occ 数组形状一致
                   （当前 0.2m 方案: (400, 400, 16)）。
    """
    D0, D1, Dz = occ_shape
    res = 2 * OCC_RANGE / D0
    g = (np.arange(D0) + 0.5) * res - OCC_RANGE
    zc = (np.arange(Dz) + 0.5) * res + OCC_Z_BASE
    gxx, gyy, gzz = np.meshgrid(g, g, zc, indexing='ij')
    ego = np.stack([gxx, gyy, gzz], axis=-1)           # (D0,D1,Dz,3)
    rot = Quaternion(infos['lidar2ego_rotation']).rotation_matrix
    t = np.array(infos['lidar2ego_translation'], dtype=np.float32)
    return (ego - t) @ rot


def _truncate_occ_z(occ, max_z):
    """把层中心 z > max_z 的层全部置为 FREE_ID，返回副本。

    png 与 ply 共用同一截断 occ（main 中统一 _truncate_occ_z 后传入），
    保证 png 的 BEV 体素集与 ply 严格一致；max_z 为 None 时原样返回。
    """
    if occ is None or max_z is None:
        return occ
    res = _occ_res(occ)
    # 层中心 z = k*res + res/2 + OCC_Z_BASE，只保留中心不超过 max_z 的层
    keep = np.arange(occ.shape[2]) <= (max_z - OCC_Z_BASE - res / 2) / res
    occ2 = occ.copy()
    occ2[..., ~keep] = FREE_ID
    return occ2


def occ_to_bev_rgb(occ, height_mode='none', color_mode='bottom'):
    """occ (Dx,Dy,Dz) [X,Y,Z] -> BEV 俯视 RGB 图 (200,200,3)。

    occ 维度排列与 create_occ_gt_custom.py 的 OCC_SHAPE=(Dx,Dy,Dz) 一致：
    dim0=X(前向), dim1=Y(左向), dim2=Z(高度)，即 occ[gx, gy, gz]。

    自定义 31 类体系：0=无标注/忽略（FREE_ID），视为空体素。

    height_mode:
      - 'none':   不调亮度。
      - 'height': 保持语义颜色，但按该列最高非空体素高度调整亮度，
                  z 越高越亮，从而在 BEV 俯视图中体现立体高度信息。

    color_mode（每列语义颜色取哪一层）：
      - 'bottom': 最低非空层（与 gt_custom 一致，GT 列内单一类别时与
                  top 相同；pred 列内混合时贴近地面结构）。
      - 'top':    最高非空层（预测 occ 默认，即 ply 体素集合的正俯视
                  投影，高层遮挡低层，竖直绿植不被下层地面截断）。
      - 'majority': 该列出现次数最多的类别（排除无标注 0）。
    """
    valid = (~_occ_empty(occ)).astype(np.int32)          # (Dx,Dy,Dz)
    # 最低非空层（用于语义颜色）
    z_bottom = np.argmax(valid, axis=2)                # (D0,D1)
    # 最高非空层（用于高度编码与 color_mode='top'）
    z_top = (occ.shape[2] - 1) - np.argmax(valid[..., ::-1], axis=2)
    no_occ = valid.sum(axis=2) == 0
    z_top = np.where(no_occ, 0, z_top)

    if color_mode == 'top':
        sem_bev = occ[np.arange(occ.shape[0])[:, None],
                      np.arange(occ.shape[1])[None, :], z_top]
    elif color_mode == 'majority':
        n_cls = COLOR_MAP.shape[0]
        counts = np.stack([(occ == c).sum(axis=2)
                           for c in range(n_cls)], axis=2)  # (200,200,n_cls)
        counts[..., EMPTY_ID] = -1  # 排除 ignore（不可观测）
        counts[..., FREE_ID] = -1  # 排除 free（空），使多数类别只在有效语义类中选
        sem_bev = counts.argmax(axis=2)
    else:
        sem_bev = occ[np.arange(occ.shape[0])[:, None],
                      np.arange(occ.shape[1])[None, :], z_bottom]
    # 无有效体素的列统一显示 0（黑，绘制时由语义 mask 置透明）
    sem_bev = np.where(no_occ, FREE_ID, sem_bev)       # (D0,D1)
    rgb = COLOR_MAP[sem_bev].astype(np.float32)        # (200,200,3)

    if height_mode == 'height':
        # z 越高越亮，factor 范围 [0.35, 1.0]
        factor = 0.35 + 0.65 * (z_top.astype(np.float32) / (occ.shape[2] - 1))
        rgb = rgb * factor[..., None]
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        rgb = rgb.astype(np.uint8)
    return rgb


def occ_to_extrusion(occ, canva_size, show_range, centers_lidar=None,
                     z_scale=1.2, cell_pad=0.9):
    """把 occ (Dx,Dy,Dz) 渲染为 BEV 上的挤压柱画布 (N,N,3)，伪 3D 立体。

    occ 维度排列 dim0=X(前向)、dim1=Y(左向)，与 create_occ_gt_custom.py 一致；
    坐标以激光(lidar)系显示（传 centers_lidar，内部已做 ego→lidar 逆变换）。

    坐标系与 vis_custom_results.py 一致（地图视角）：
      col 小 = y 左，row 小 = x 前（即 x 前朝画布上方、y 左朝画布左方）。
    高度 z 沿画布向上（row 减小）延伸，顶面向右上偏移形成菱形面。
    每个非空 (x,y) 栅格按最高 occupied 高度竖一根柱：
      顶面 = 语义原色(最亮)，右侧面 = 语义色*0.75，上侧面(后侧) = 语义色*0.55。
    绘制顺序：按柱顶深度分桶（远→近），桶内先上侧面 -> 右侧面 -> 顶面。
    """
    N = int(canva_size)
    canvas = np.zeros((N, N, 3), dtype=np.uint8)
    occ_h = int(canva_size * OCC_RANGE / show_range)
    cell = occ_h / float(occ.shape[0])
    margin = int((canva_size - occ_h) / 2)
    half = cell * cell_pad / 2.0

    valid = (~_occ_empty(occ)).astype(np.int32)          # (Dx,Dy,Dz)
    z_bottom = np.argmax(valid, axis=2)                # 最低非空层(语义)
    z_top = (occ.shape[2] - 1) - np.argmax(valid[..., ::-1], axis=2)
    no_occ = valid.sum(axis=2) == 0
    ys, xs = np.nonzero(~no_occ)                        # ys=dim0=X, xs=dim1=Y
    if ys.size == 0:
        return canvas
    sem = occ[ys, xs, z_bottom[ys, xs]]                # (M,) 语义类别

    if centers_lidar is not None:
        # occ 体素中心已在 lidar 系：按 col=-y、row=-x 投影到全画布
        c = centers_lidar[ys, xs, z_bottom[ys, xs]]    # (M,3) lidar 系
        scale = N / (2 * show_range)
        cols = (-c[:, 1] + show_range) * scale         # y 左 → col 左
        rows = (-c[:, 0] + show_range) * scale         # x 前 → row 上
    else:
        # 退化：假设 occ 已在 lidar 系，按 dim0=X、dim1=Y 的栅格映射
        rows = margin + (occ.shape[0] - 1 - ys) * cell + half  # X 大 → row 小(上)
        cols = margin + (occ.shape[1] - 1 - xs) * cell + half  # Y 大 → col 小(左)
    hpx = (z_top[ys, xs].astype(np.float32) + 1) * cell * z_scale
    ddx = hpx * 0.45                                   # 顶面右偏移
    ddy = -hpx                                         # 顶面上偏移(row减小)

    w = half
    # 按柱顶深度分桶：深度 = 顶面上边缘 row（越小越远），远→近绘制
    depth = rows - hpx - w
    bucket_size = max(1.0, N / 32.0)
    buckets = {}
    for i in np.argsort(depth):
        b = int(np.clip(depth[i] // bucket_size, 0, 31))
        buckets.setdefault(b, []).append(i)

    n_cls = COLOR_MAP.shape[0]
    for b in sorted(buckets):
        idx = np.array(buckets[b])
        rr = rows[idx]
        cc = cols[idx]
        dx = ddx[idx]
        dy = ddy[idx]
        sub_sem = sem[idx]
        for c in range(n_cls):
            sel = np.nonzero(sub_sem == c)[0]
            if sel.size == 0:
                continue
            r2 = rr[sel]
            c2 = cc[sel]
            x2 = dx[sel]
            y2 = dy[sel]
            # 上侧面(后侧, 最暗)
            side_u = np.stack([
                np.stack([c2 - w, r2 - w], axis=1),
                np.stack([c2 + w, r2 - w], axis=1),
                np.stack([c2 + w + x2, r2 - w + y2], axis=1),
                np.stack([c2 - w + x2, r2 - w + y2], axis=1),
            ], axis=1).astype(np.int32)
            # 右侧面(中暗)
            side_r = np.stack([
                np.stack([c2 + w, r2 - w], axis=1),
                np.stack([c2 + w, r2 + w], axis=1),
                np.stack([c2 + w + x2, r2 + w + y2], axis=1),
                np.stack([c2 + w + x2, r2 - w + y2], axis=1),
            ], axis=1).astype(np.int32)
            # 顶面(语义原色)
            top = np.stack([
                np.stack([c2 - w + x2, r2 - w + y2], axis=1),
                np.stack([c2 + w + x2, r2 - w + y2], axis=1),
                np.stack([c2 + w + x2, r2 + w + y2], axis=1),
                np.stack([c2 - w + x2, r2 + w + y2], axis=1),
            ], axis=1).astype(np.int32)
            base = COLOR_MAP[c]
            cv2.fillPoly(canvas, list(side_u), tuple(int(v * 0.55) for v in base))
            cv2.fillPoly(canvas, list(side_r), tuple(int(v * 0.75) for v in base))
            cv2.fillPoly(canvas, list(top), tuple(int(v) for v in base))
    return canvas


def occ_to_isometric(occ, canva_size, show_range, centers_lidar=None,
                     z_scale=1.2, cell_pad=0.9):
    """把 occ (Dx,Dy,Dz) 渲染为等距斜视画布 (N,N,3)，伪 3D 立体。

    occ 维度排列 dim0=X(前向)、dim1=Y(左向)，与 create_occ_gt_custom.py 一致；
    坐标以激光(lidar)系显示（传 centers_lidar，内部已做 ego→lidar 逆变换）。

    坐标系与 vis_custom_results.py 一致（地图视角）：
      col 小 = y 左，row 小 = x 前（x 前朝画布上方、y 左朝画布左方）。

    等距投影（2:1 菱形，30° 轴）：
        u = (cols - rows) * cos30      屏幕水平（x 前/上方侧 → 右下斜）
        v = (cols + rows) * sin30 - hpx 屏幕垂直（向下为正，高度 z 使 v 减小）
    观察者位于南侧俯视，x+（前）与 y+（左）两个斜面可见：
        顶面 = 语义原色，右斜面(x+) = 语义色*0.75，左斜面(y+) = 语义色*0.55。
    绘制顺序：按底面深度 v 分桶（远→近），桶内先左斜面 -> 右斜面 -> 顶面。

    返回 (canvas, proj)：proj(cols, rows) 把 BEV 俯视画布坐标(无高度)映射到
    等距画布坐标，供 lidar 点与 3D 框叠加使用。
    """
    N = int(canva_size)
    canvas = np.zeros((N, N, 3), dtype=np.uint8)
    occ_h = int(canva_size * OCC_RANGE / show_range)
    cell = occ_h / float(occ.shape[0])
    margin = int((canva_size - occ_h) / 2)
    half = cell * cell_pad / 2.0
    pad = 0.02 * N

    valid = (~_occ_empty(occ)).astype(np.int32)          # (Dx,Dy,Dz)
    z_bottom = np.argmax(valid, axis=2)                # 最低非空层(语义)
    z_top = (occ.shape[2] - 1) - np.argmax(valid[..., ::-1], axis=2)
    no_occ = valid.sum(axis=2) == 0
    ys, xs = np.nonzero(~no_occ)                        # ys=dim0=X, xs=dim1=Y
    if ys.size == 0:
        return canvas, None
    sem = occ[ys, xs, z_bottom[ys, xs]]                # (M,) 语义类别

    if centers_lidar is not None:
        # occ 体素中心已在 lidar 系：按 col=-y、row=-x 投影到全画布
        c = centers_lidar[ys, xs, z_bottom[ys, xs]]    # (M,3) lidar 系
        scale = N / (2 * show_range)
        cols = (-c[:, 1] + show_range) * scale         # y 左 → col 左
        rows = (-c[:, 0] + show_range) * scale         # x 前 → row 上
    else:
        # 退化：假设 occ 已在 lidar 系，按 dim0=X、dim1=Y 的栅格映射
        rows = margin + (occ.shape[0] - 1 - ys) * cell + half  # X 大 → row 小(上)
        cols = margin + (occ.shape[1] - 1 - xs) * cell + half  # Y 大 → col 小(左)
    hpx = (z_top[ys, xs].astype(np.float32) + 1) * cell * z_scale

    cos30 = np.cos(np.pi / 6.0)
    sin30 = 0.5
    # 等距平面：x 轴沿 row 方向（x_lidar 前向 → 上方），y 轴沿 col 方向（y_lidar 左向 → 左方）
    x0, x1 = rows - half, rows + half
    y0, y1 = cols - half, cols + half

    # 所有角点 u/v 范围（用于 fit 到画布）
    us = np.concatenate([
        (x0 + y0 - N) * cos30, (x1 + y0 - N) * cos30,
        (x1 + y1 - N) * cos30, (x0 + y1 - N) * cos30])
    vs_b = np.concatenate([
        (x0 - y0) * sin30, (x1 - y0) * sin30,
        (x1 - y1) * sin30, (x0 - y1) * sin30])
    u_min, u_max = us.min(), us.max()
    v_min = vs_b.min() - hpx.max()
    v_max = vs_b.max()
    s = min(1.0, (N - 2 * pad) / max(u_max - u_min, v_max - v_min))

    def proj(cols, rows):
        u = (cols + rows - N) * cos30
        v = (cols - rows) * sin30
        return (u - u_min) * s + pad, (v - v_min) * s + pad

    def to_poly(u, v):
        u = (u - u_min) * s + pad
        v = (v - v_min) * s + pad
        return np.stack([u, v], axis=2).astype(np.int32)

    # 按底面深度分桶（depth 大=近），远→近绘制保证遮挡
    depth = (cols - rows) * sin30
    dmin = depth.min()
    bucket_size = max((depth.max() - dmin) / 32.0, 1e-6)
    buckets = {}
    for i in np.argsort(depth):
        b = int(np.clip((depth[i] - dmin) / bucket_size, 0, 31))
        buckets.setdefault(b, []).append(i)

    n_cls = COLOR_MAP.shape[0]
    for b in sorted(buckets):
        idx = np.array(buckets[b])
        x0s, x1s = x0[idx], x1[idx]
        y0s, y1s = y0[idx], y1[idx]
        hs = hpx[idx]
        sub_sem = sem[idx]
        for c in range(n_cls):
            sel = np.nonzero(sub_sem == c)[0]
            if sel.size == 0:
                continue
            X0, X1 = x0s[sel], x1s[sel]
            Y0, Y1 = y0s[sel], y1s[sel]
            H = hs[sel]
            # 左斜面(y+ 后侧, 最暗)
            side_l = to_poly(
                np.stack([X0 + Y1 - N, X1 + Y1 - N, X1 + Y1 - N, X0 + Y1 - N],
                         axis=1),
                np.stack([X0 - Y1, X1 - Y1, X1 - Y1 - H, X0 - Y1 - H], axis=1))
            # 右斜面(x+ 前侧, 中暗)
            side_r = to_poly(
                np.stack([X1 + Y0 - N, X1 + Y1 - N, X1 + Y1 - N, X1 + Y0 - N],
                         axis=1),
                np.stack([X1 - Y0, X1 - Y1, X1 - Y1 - H, X1 - Y0 - H], axis=1))
            # 顶面(语义原色)
            top = to_poly(
                np.stack([X0 + Y0 - N, X1 + Y0 - N, X1 + Y1 - N, X0 + Y1 - N],
                         axis=1),
                np.stack([X0 - Y0 - H, X1 - Y0 - H, X1 - Y1 - H, X0 - Y1 - H],
                         axis=1))
            base = COLOR_MAP[c]
            cv2.fillPoly(canvas, list(side_l), tuple(int(v * 0.55) for v in base))
            cv2.fillPoly(canvas, list(side_r), tuple(int(v * 0.75) for v in base))
            cv2.fillPoly(canvas, list(top), tuple(int(v) for v in base))
    return canvas, proj


# ------------------------- Open3D 真 3D -------------------------

def _parse_vec3(s, default=(0, 0, 0)):
    """解析 'x,y,z'（兼容空格/括号/分号）为 np.ndarray(3)。"""
    if s is None:
        return np.array(default, dtype=np.float64)
    parts = s.replace(';', ',').replace('(', '').replace(')', '').split(',')
    if len(parts) == 1:
        parts = s.split()
    try:
        return np.array([float(x) for x in parts[:3]], dtype=np.float64)
    except Exception:
        return np.array(default, dtype=np.float64)


def _depth_to_colors(depths):
    """向量化 depth2color：depth -> RGB(float32, 0-255)。"""
    gray = np.clip((depths + 2.5) / 3.0, 0.0, 1.0)
    max_lumi = 200
    cols = np.array(
        [[max_lumi, 0, max_lumi], [max_lumi, 0, 0], [max_lumi, max_lumi, 0],
         [0, max_lumi, 0], [0, max_lumi, max_lumi], [0, 0, max_lumi]],
        dtype=np.float32)
    num_rank = len(cols) - 1
    rank = np.clip(np.floor(gray * num_rank).astype(int), 0, num_rank - 1)
    diff = (gray - rank / num_rank) * num_rank
    return cols[rank] + (cols[rank + 1] - cols[rank]) * diff[:, None]


def _load_lidar_xyz(lidar_path):
    """读取 lidar bin 为 (N,3) FLU(x前 y左 z上) 点云，自适应列数。

    自定义数据(yk_c1) bin 为 6 列 (x,y,z,intensity,semantic,instance)；
    标准 nuScenes(bevdetv3) 为 5 列 (x,y,z,intensity,timestamp)；
    部分数据为 4 列 (x,y,z,intensity)。统一取前 3 列。
    """
    arr = np.fromfile(lidar_path, dtype=np.float32)
    total = arr.size
    for n_col in (6, 5, 4):
        if total % n_col == 0:
            return arr.reshape(-1, n_col)[:, :3]
    # 兜底：按 4 列读取，丢弃尾部不足一行的点
    usable = total // 4 * 4
    return arr[:usable].reshape(-1, 4)[:, :3]


def occ_to_3d_meshes(occ, voxel_size=0.38, centers_lidar=None, max_z=None):
    """把 occ (Dx,Dy,Dz) [X,Y,Z] 转换为语义着色的立方体网格列表。

    occ 维度排列 dim0=X(前向)、dim1=Y(左向)、dim2=Z，与 create_occ_gt_custom.py
    的 OCC_SHAPE=(Dx,Dy,Dz) 一致；世界坐标采用 FLU lidar 系（x 前、y 左、z 上），
    与 2D BEV 画布、lidar 点云、GT 框同系。传 centers_lidar 时直接使用其
    体素中心（已做 ego→lidar 逆变换，正确处理 lidar 安装高度等偏移）。

    z 映射按当前 occ 栅格 z∈[-0.5,2.7]m：层 k 中心 = k*res + res/2 + OCC_Z_BASE：
        x = gx*res + res/2 - OCC_RANGE   (gx: dim0=X 索引)
        y = gy*res + res/2 - OCC_RANGE   (gy: dim1=Y 索引)
        z = gz*res + res/2 + OCC_Z_BASE   (gz: dim2=Z 索引)
    自定义 31 类体系：0=无标注/忽略(FREE_ID)不生成网格。
    每个非空体素生成一个立方体，按语义类别分组为独立 TriangleMesh
    （每类 paint_uniform_color），便于渲染与 ply 导出。
    """
    if occ is None:
        return []
    if max_z is not None:
        occ = _truncate_occ_z(occ, max_z)
    res = _occ_res(occ)
    s = voxel_size / 2.0

    nz = np.nonzero(~_occ_empty(occ))
    if nz[0].size == 0:
        return []
    gx, gy, gz = nz                                   # dim0=X, dim1=Y, dim2=Z
    if centers_lidar is not None:
        centers = centers_lidar[gx, gy, gz].astype(np.float64)
    else:
        centers = np.stack([
            gx * res + res / 2 - OCC_RANGE,
            gy * res + res / 2 - OCC_RANGE,
            gz * res + res / 2 + OCC_Z_BASE,
        ], axis=1).astype(np.float64)
    sem = occ[nz].astype(np.int32)

    # 单位立方体 8 顶点 / 12 三角面（相对中心 ±s）
    offs = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=np.float64) * s
    tris = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ], dtype=np.int64)

    meshes = []
    n_cls = COLOR_MAP.shape[0]
    for c in range(n_cls):
        sel = np.nonzero(sem == c)[0]
        if sel.size == 0:
            continue
        vc = centers[sel]
        verts = (vc[:, None, :] + offs[None, :, :]).reshape(-1, 3)
        faces = (tris[None, :, :] +
                 8 * np.arange(sel.size)[:, None, None]).reshape(-1, 3)
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts),
            o3d.utility.Vector3iVector(faces))
        mesh.paint_uniform_color(COLOR_MAP[c].astype(np.float64) / 255.0)
        mesh.compute_vertex_normals()
        meshes.append((f'occ_{c}', mesh))
    return meshes


def lidar_to_3d_pcd(lidar_path):
    """读取 lidar 原始点(FLU, x前 y左 z上)为按高度着色的 PointCloud。

    自适应 4/5/6 列（自定义 yk_c1=6 列 / 标准 nuScenes=5 列），取前 3 列。
    """
    pts = _load_lidar_xyz(lidar_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(
        _depth_to_colors(pts[:, 2]).astype(np.float64) / 255.0)
    return pcd


def boxes_to_3d_lineset(corners_lidar, pred_flag):
    """3D 框 8 角点(FLU) -> 12 边 LineSet，pred 青 / GT 黄。

    角点顺序不固定，按两两距离最近的前 12 对自动配对棱边，鲁棒。
    """
    corners = corners_lidar.reshape(-1, 8, 3)
    n = corners.shape[0]
    if n == 0:
        return o3d.geometry.LineSet()
    lines = []
    for k in range(n):
        c = corners[k].astype(np.float64)
        d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
        i, j = np.triu_indices(8, 1)
        pairs = np.stack([i, j], axis=1)
        order = np.argsort(d[i, j])[:12]
        lines.append(pairs[order] + 8 * k)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(
        corners.reshape(-1, 3).astype(np.float64))
    ls.lines = o3d.utility.Vector2iVector(
        np.concatenate(lines, axis=0))
    cols = np.array([color_map[int(flag)] for flag in pred_flag],
                    dtype=np.float64) / 255.0
    ls.colors = o3d.utility.Vector3dVector(np.repeat(cols, 12, axis=0))
    return ls


def show_3d_window(geoms, front, up, lookat, zoom):
    """弹窗交互查看（鼠标旋转/缩放；需 DISPLAY，headless 下提示跳过）。"""
    if not os.environ.get('DISPLAY'):
        print('[3D] 无 DISPLAY，跳过交互弹窗；可用 --occ-3d-export 离屏导出')
        return
    o3d.visualization.draw_geometries(
        [g for _, g in geoms], window_name='FastBEV occ results 3D (custom)',
        width=1600, height=1000,
        front=list(front), lookat=list(lookat), up=list(up), zoom=zoom)


def export_3d_png_ply(geoms, out_dir, prefix, full_img=None, occ=None,
                      occ_gt=None, lidar_path=None, corners_lidar=None,
                      pred_flag=None, scores=None, vis_thred=0.3,
                      draw_lidar=True, canva_size=1000, show_range=50,
                      scale_factor=4, occ_color_mode='top',
                      centers_lidar=None):
    """导出拼接 png（6 相机 + BEV 俯视）+ occ 体素 ply。

    png 复用主流程的完整拼接图 full_img（上 3 相机 + 下 3 相机），
    中间的 BEV 区域用 occ_height_mode='height' + occ_color_mode 重绘后贴回。
    occ_gt 非 None 时，BEV 区域左右分块：左=GT occ + GT 框，右=Pred
    occ + 全部框（与主流程 vis_full 一致）。
    ply 为真 3D occ 体素网格（由 geoms 中 occ_* 网格合并导出）。
    3D 框保留 pred 青 / GT 黄配色。
    """
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f'{prefix}.png')
    try:
        n_inst = 0 if corners_lidar is None else corners_lidar.shape[0] // 8
        if pred_flag is None:
            pred_flag = np.zeros(n_inst, dtype=bool)
        if scores is None:
            scores = np.ones(n_inst, dtype=np.float32)
        if occ_gt is not None:
            # 左右分块：左 GT occ / 右 Pred occ
            flags = pred_flag.reshape(n_inst) if n_inst else np.zeros(0, bool)
            gt_sel = ~flags
            if n_inst:
                corners_gt = corners_lidar.reshape(
                    n_inst, 8, 3)[gt_sel].reshape(-1, 3)
            else:
                corners_gt = np.zeros((0, 3), dtype=np.float32)
            canvas = draw_bev_split_canvas(
                canva_size, show_range, occ_left=occ_gt, occ_right=occ,
                corners_gt=corners_gt, corners_all=corners_lidar,
                pred_flag=flags, scores=scores,
                vis_thred=vis_thred,
                lidar_path=lidar_path, draw_lidar=draw_lidar,
                occ_height_mode='height', occ_color_mode=occ_color_mode,
                centers_lidar=centers_lidar)
        else:
            canvas = draw_bev_canvas(
                canva_size, show_range, occ=occ,
                corners_lidar=corners_lidar,
                pred_flag=pred_flag, scores=scores,
                vis_thred=vis_thred,
                lidar_path=lidar_path, draw_lidar=draw_lidar,
                occ_height_mode='height', occ_color_mode=occ_color_mode,
                centers_lidar=centers_lidar)
        if full_img is not None:
            # BEV 区域在 img 中的位置（与主流程拼接布局一致）
            h, w = full_img.shape[:2]
            w_begin = int((w - canva_size) // 2)
            y_begin = int(900 / scale_factor)
            if (w_begin >= 0 and y_begin >= 0
                    and y_begin + canva_size <= h
                    and w_begin + canva_size <= w):
                out_img = full_img.copy()
                if occ_gt is not None:
                    h_b = canvas.shape[0]
                    y0 = y_begin + (canva_size - h_b) // 2
                    out_img[y0:y0 + h_b, w_begin:w_begin + canva_size, :] = \
                        canvas
                else:
                    out_img[y_begin:y_begin + canva_size,
                            w_begin:w_begin + canva_size, :] = canvas
            else:
                out_img = canvas
        else:
            out_img = canvas
        cv2.imwrite(png_path, out_img)
        print(f'[3D] saved {png_path} (6cam + BEV height+'
              + (', GT/Pred split)' if occ_gt is not None else ')'))
    except Exception as e:
        print(f'[3D] png render failed: {e}')
    meshes = [g for name, g in geoms if str(name).startswith('occ_')]
    if meshes:
        merged = o3d.geometry.TriangleMesh()
        for m in meshes:
            merged += m
        ply_path = os.path.join(out_dir, f'{prefix}.ply')
        o3d.io.write_triangle_mesh(ply_path, merged)
        print(f'[3D] saved {ply_path}')
    else:
        print('[3D] 无 occ 体素，跳过 ply 导出（可用 --occ-3d-lidar '
              '叠加点云）')


# ------------------------- 绘制 -------------------------

def draw_camera_images(infos, corners_lidar, pred_flag, views, scale_factor):
    """绘制 6 相机图（投影 3D 框），返回列表 img（已 resize 到 1600x900）。"""
    imgs = []
    for view in views:
        img = cv2.imread(infos['cams'][view]['data_path'])
        if img is None:
            img = np.zeros((900, 1600, 3), dtype=np.uint8)

        if corners_lidar.shape[0] > 0:
            corners_img, valid = lidar2img(corners_lidar, infos['cams'][view])
            valid = np.logical_and(
                valid,
                check_point_in_img(corners_img, img.shape[0], img.shape[1]))
            valid = valid.reshape(-1, 8)
            corners_img = corners_img.reshape(-1, 8, 2).astype(np.int32)
            for aid in range(valid.shape[0]):
                for index in draw_boxes_indexes_img_view:
                    if valid[aid, index[0]] and valid[aid, index[1]]:
                        cv2.line(
                            img,
                            corners_img[aid, index[0]],
                            corners_img[aid, index[1]],
                            color=color_map[int(pred_flag[aid])],
                            thickness=scale_factor)
        # 自定义数据相机分辨率不统一（如 CAM_FRONT 3840x2160，其余 1920x1536），
        # 画框（按原始尺寸投影）后统一 resize 到 1600x900，保证拼接一致
        if img.shape[:2] != (900, 1600):
            img = cv2.resize(img, (1600, 900))
        imgs.append(img)
    return imgs


def draw_bev_canvas(canva_size, show_range, occ=None, corners_lidar=None,
                    pred_flag=None, scores=None, vis_thred=0.3,
                    lidar_path=None, draw_lidar=True,
                    occ_height_mode='none', occ_z_scale=1.2,
                    occ_color_mode='top', centers_lidar=None,
                    occ_mask=None):
    """绘制 BEV 画布：occ 底图 + lidar 点 + 检测框。

    坐标系与 vis_custom_results.py 完全一致（FLU lidar，地图视角）：
      col = (-y + show_range) / (2*show_range) * canvas   (y左→col左)
      row = (-x + show_range) / (2*show_range) * canvas   (x前→row上)
    即：前向(x+) 朝画布上方，左向(y+) 朝画布左方，FRONT 箭头与坐标系自洽。

    occ 维度排列 dim0=X(前向)、dim1=Y(左向)，且真值/预测均在 ego 系栅格；
    传 centers_lidar（occ_centers_lidar 输出，已做 ego→lidar 逆变换）时
    occ 以激光坐标系与 lidar 点云 / GT 框精确对齐显示。
    """
    canvas = np.zeros((int(canva_size), int(canva_size), 3), dtype=np.uint8)

    # 1) occ 底图
    #    occ 维度 (dim0=X 前向, dim1=Y 左向)，坐标以 lidar 系显示
    #    贴图: occ_rgb[gx, gy] -> occ_bev[row, col]，row=-x、col=-y → 180° 翻转
    proj_fn = None
    if occ is not None:
        # 可选：用观测 mask 裁剪 occ（mask 外置 FREE_ID=0，渲染时自动过滤）。
        # Pred 侧传 GT mask_camera，可避免"无标注空间被乱预测的噪点"淹没有效语义。
        if occ_mask is not None:
            m = occ_mask.astype(bool)
            if m.ndim == 2:
                m = m[..., None]
            occ = np.where(m, occ, 0)
        if occ_height_mode == 'column':
            # 挤压柱伪 3D：每个非空格子按高度竖柱
            canvas = occ_to_extrusion(occ, canva_size, show_range,
                                      centers_lidar=centers_lidar,
                                      z_scale=occ_z_scale)
        elif occ_height_mode == 'isometric':
            # 等距斜视：整幅场景以 30° 菱形投影呈现，含高度柱体
            canvas, proj_fn = occ_to_isometric(occ, canva_size, show_range,
                                               centers_lidar=centers_lidar,
                                               z_scale=occ_z_scale)
        else:
            occ_rgb = occ_to_bev_rgb(occ, height_mode=occ_height_mode,
                                     color_mode=occ_color_mode)
            occ_h = int(canva_size * OCC_RANGE / show_range)
            occ_bev = cv2.resize(
                np.ascontiguousarray(occ_rgb[::-1, ::-1]),
                (occ_h, occ_h),
                interpolation=cv2.INTER_NEAREST)
            margin = int((canva_size - occ_h) / 2)
            # 无标注(0)区域改为透明，只保留有语义的列；
            # mask 基于“该列是否有非 free 体素”（occ != FREE_ID=0），
            # 与 ply 体素列集合一一对应（height 模式调暗后 RGB 阈值不可靠）
            mask = (~_occ_empty(occ)).any(axis=2).astype(np.uint8) * 255
            mask = cv2.resize(mask[::-1, ::-1], (occ_h, occ_h),
                              interpolation=cv2.INTER_NEAREST)
            occ_bev_masked = cv2.bitwise_and(occ_bev, occ_bev, mask=mask)
            # occ 在 ego 系生成，按 centers_lidar 推算整幅贴图相对画布中心的
            # 平移量，使 occ 与 lidar 系点云/框对齐（occ 栅格中心即 (-t_x, -t_y)）
            dx = dy = 0
            if centers_lidar is not None:
                c0 = centers_lidar[centers_lidar.shape[0] // 2,
                                   centers_lidar.shape[1] // 2, 0, :2]
                scale = canva_size / (2 * show_range)
                dy = int(round(-c0[0] * scale))   # x 平移 → row 方向
                dx = int(round(c0[1] * scale))    # y 平移 → col 方向
            x0 = margin + dx
            y0 = margin + dy
            rx0, rx1 = max(x0, 0), min(x0 + occ_h, canva_size)
            ry0, ry1 = max(y0, 0), min(y0 + occ_h, canva_size)
            if ry1 > ry0 and rx1 > rx0:
                roi = canvas[ry0:ry1, rx0:rx1, :]
                roi[:] = cv2.addWeighted(
                    roi, 1.0,
                    occ_bev_masked[ry0 - y0:ry1 - y0, rx0 - x0:rx1 - x0],
                    0.5, 0)

            # GT 有效观测区域背景（occ_mask）：mask=1 且无语义(FREE_ID) 的列
            # 显示半透明深灰底，避免稀疏 GT 左块整片漆黑、丢失观测轮廓
            if occ_mask is not None:
                if occ_mask.ndim == 3:
                    bg_mask = occ_mask.any(axis=2).astype(np.uint8) * 255
                else:
                    bg_mask = (occ_mask > 0).astype(np.uint8) * 255
                bg_mask = cv2.resize(bg_mask[::-1, ::-1], (occ_h, occ_h),
                                     interpolation=cv2.INTER_NEAREST)
                # 只保留“有效但无语义”的列（mask 为 occ 非空列，取反）
                bg_mask = cv2.bitwise_and(bg_mask, cv2.bitwise_not(mask))
                if ry1 > ry0 and rx1 > rx0:
                    gray = np.full_like(
                        occ_bev_masked[ry0 - y0:ry1 - y0, rx0 - x0:rx1 - x0],
                        48)
                    gray = cv2.bitwise_and(
                        gray, gray,
                        mask=bg_mask[ry0 - y0:ry1 - y0, rx0 - x0:rx1 - x0])
                    roi = canvas[ry0:ry1, rx0:rx1, :]
                    roi[:] = cv2.addWeighted(roi, 1.0, gray, 0.5, 0)

        # 方向指示（与 vis_custom_results 一致）：黄点=ego，黄箭头朝上=FRONT
        cx = int(canva_size / 2)
        cy = cx
        if proj_fn is not None:
            fx, fy = proj_fn(float(cx), float(cy))
            fx2, fy2 = proj_fn(float(cx), float(cy - 60))
            dx, dy = fx2 - fx, fy2 - fy
            cv2.arrowedLine(canvas, (int(fx), int(fy)),
                            (int(fx + dx), int(fy + dy)), (0, 255, 255), 2,
                            tipLength=0.3)
            cv2.putText(canvas, 'FRONT', (int(fx + dx + 5), int(fy + dy + 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.circle(canvas, (cx, cy), 5, (0, 255, 255), -1)
            cv2.arrowedLine(canvas, (cx, cy), (cx, cy - 60),
                            (0, 255, 255), 2, tipLength=0.3)
            cv2.putText(canvas, 'FRONT', (cx - 35, cy - 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 2) lidar 点云（自适应 4/5/6 列，取前 3 列；投影同 vis_custom_results）
    if draw_lidar and lidar_path and os.path.exists(lidar_path):
        lidar_points = _load_lidar_xyz(lidar_path)
        xs = lidar_points[:, 0]
        ys = lidar_points[:, 1]
        mask = (np.abs(xs) < show_range) & (np.abs(ys) < show_range)
        xs, ys = xs[mask], ys[mask]
        zs = lidar_points[mask, 2]
        scale = canva_size / (2 * show_range)
        px = ((-ys + show_range) * scale).astype(np.int32)   # col
        py = ((-xs + show_range) * scale).astype(np.int32)   # row
        valid = (px >= 0) & (px < canva_size) & (py >= 0) & (py < canva_size)
        px, py, zs = px[valid], py[valid], zs[valid]
        if proj_fn is not None:
            # 等距投影：逐点映射后画圆
            for i in range(px.shape[0]):
                color = depth2color(zs[i])
                px2, py2 = proj_fn(float(px[i]), float(py[i]))
                cv2.circle(canvas, (int(px2), int(py2)),
                           radius=0, color=color, thickness=1)
        else:
            # 俯视投影：直接索引赋值（向量化，避免数万次 cv2.circle 慢循环）
            colors = _depth_to_colors(zs).astype(np.uint8)
            canvas[py, px] = colors

    # 3) 3D 检测框（新坐标系：col=-y, row=-x，与 vis_custom_results 一致）
    if corners_lidar is not None and corners_lidar.shape[0] > 0:
        corners_lidar = corners_lidar.reshape(-1, 8, 3)
        cx = corners_lidar[:, :, 0]  # forward
        cy = corners_lidar[:, :, 1]  # left
        rot_x = -cy   # col（y左→col左）
        rot_y = -cx   # row（x前→row上）
        bottom_idx = [0, 3, 7, 4]
        bottom_rot_x = rot_x[:, bottom_idx]
        bottom_rot_y = rot_y[:, bottom_idx]
        bottom_corners_bev = np.stack([bottom_rot_x, bottom_rot_y], axis=-1)
        bottom_corners_bev = \
            (bottom_corners_bev + show_range) / (2 * show_range) * canva_size
        center_bev = np.stack([rot_x, rot_y], axis=-1).mean(axis=1)
        head_bev = np.stack([rot_x[:, [4, 7]], rot_y[:, [4, 7]]],
                            axis=-1).mean(axis=1)
        center_canvas = \
            (center_bev + show_range) / (2 * show_range) * canva_size
        head_canvas = (head_bev + show_range) / (2 * show_range) * canva_size
        if proj_fn is not None:
            uu, vv = proj_fn(bottom_corners_bev[..., 0].reshape(-1),
                             bottom_corners_bev[..., 1].reshape(-1))
            bottom_corners_bev = np.stack([uu, vv],
                                          axis=1).reshape(-1, 4, 2)
            uu, vv = proj_fn(center_canvas[:, 0], center_canvas[:, 1])
            center_canvas = np.stack([uu, vv], axis=1)
            uu, vv = proj_fn(head_canvas[:, 0], head_canvas[:, 1])
            head_canvas = np.stack([uu, vv], axis=1)
        bottom_corners_bev = np.round(bottom_corners_bev).astype(np.int32)
        center_canvas = center_canvas.astype(np.int32)
        head_canvas = head_canvas.astype(np.int32)

        for rid in range(corners_lidar.shape[0]):
            score = scores[rid] if scores is not None else 1.0
            if pred_flag[rid] and score < vis_thred:
                continue
            score = min(score * 2.0, 1.0) if pred_flag[rid] else 1.0
            color = color_map[int(pred_flag[rid])]
            c = tuple(int(channel * score) for channel in color)
            for index in draw_boxes_indexes_bev:
                cv2.line(
                    canvas,
                    tuple(bottom_corners_bev[rid, index[0]].tolist()),
                    tuple(bottom_corners_bev[rid, index[1]].tolist()),
                    c, thickness=1)
            cv2.line(
                canvas,
                tuple(center_canvas[rid].tolist()),
                tuple(head_canvas[rid].tolist()),
                c, 1, lineType=8)

    return canvas


def draw_bev_split_canvas(canva_size, show_range, occ_left=None,
                          occ_right=None, corners_gt=None, corners_all=None,
                          pred_flag=None, scores=None, vis_thred=0.3,
                          lidar_path=None, draw_lidar=True,
                          occ_height_mode='none', occ_z_scale=1.2,
                          occ_color_mode=None, centers_lidar=None,
                          occ_left_mask=None, occ_right_mask=None,
                          occ_color_mode_left='bottom',
                          occ_color_mode_right='top',
                          label_left='GT', label_right='Pred'):
    """把 BEV 区域分成左右两块画布：左=occ 真值(GT)，右=occ 预测(Pred)。

    两块均为 half×half 方形画布（half = canva_size//2），并排成
    (half, canva_size) 横条，由调用方垂直居中贴回原 BEV 区域：
      - 左块：GT occ 底图 + GT 框（黄）+ lidar；
      - 右块：Pred occ 底图 + 全部框（pred 青 / GT 黄）+ lidar。
    GT/Pred 可分别指定 color_mode（GT 稀疏建议 bottom，
    Pred 与 ply 对齐建议 top）；occ_left_mask 用于给 GT 有效观测
    区域画半透明背景。
    相机拼接等其余部分完全不变。内部对 corners 一律取副本，
    不影响外部与 3D 导出的原始坐标。
    """
    # 兼容旧调用：显式传 occ_color_mode 时统一覆盖 GT/Pred 取色模式
    if occ_color_mode is not None:
        occ_color_mode_left = occ_color_mode
        occ_color_mode_right = occ_color_mode
    half = int(canva_size // 2)
    # 左块：GT occ + GT 框（黄）
    n_gt = 0 if corners_gt is None else corners_gt.shape[0] // 8
    canvas_l = draw_bev_canvas(
        half, show_range, occ=occ_left,
        corners_lidar=None if corners_gt is None else corners_gt.copy(),
        pred_flag=np.zeros(n_gt, dtype=bool),
        scores=np.ones(n_gt, dtype=np.float32),
        vis_thred=vis_thred,
        lidar_path=lidar_path, draw_lidar=draw_lidar,
        occ_height_mode=occ_height_mode, occ_z_scale=occ_z_scale,
        occ_color_mode=occ_color_mode_left,
        centers_lidar=centers_lidar,
        occ_mask=occ_left_mask)
    # 右块：Pred occ + 全部框（pred 青 / GT 黄）
    n_all = 0 if corners_all is None else corners_all.shape[0] // 8
    if pred_flag is None:
        pred_flag = np.zeros(n_all, dtype=bool)
    if scores is None:
        scores = np.ones(n_all, dtype=np.float32)
    canvas_r = draw_bev_canvas(
        half, show_range, occ=occ_right,
        corners_lidar=None if corners_all is None else corners_all.copy(),
        pred_flag=pred_flag, scores=scores,
        vis_thred=vis_thred,
        lidar_path=lidar_path, draw_lidar=draw_lidar,
        occ_height_mode=occ_height_mode, occ_z_scale=occ_z_scale,
        occ_color_mode=occ_color_mode_right,
        centers_lidar=centers_lidar,
        occ_mask=occ_right_mask)
    # 画布内顶部标签
    for c, lab, color in ((canvas_l, label_left, (255, 255, 255)),
                          (canvas_r, label_right, (255, 255, 255))):
        (tw, th), _ = cv2.getTextSize(
            lab, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(c, lab, (max(6, (half - tw) // 2), th + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    # 拼合成横条（左 GT / 右 Pred），中间白色分隔线
    strip = np.concatenate([canvas_l, canvas_r], axis=1)  # (half, canva_size, 3)
    cv2.line(strip, (half, 0), (half, strip.shape[0] - 1), (255, 255, 255), 1)
    return strip


def _is_ego2global_identity(infos_list):
    """检查所有帧的 ego2global 是否都是恒等变换（自定义数据集标志）。

    自定义数据（yk_c1）ego=global，FastBEV 的 BEV 栅格/检测输出已在 lidar 系
    （实测 boxes_3d.tensor z ≈ GT bottom，与 GT 同系），无需再做 ego→lidar 转换；
    nuScenes 官方数据 Pred 在 ego 系，需 ego→lidar 转换。
    """
    for info in infos_list[:10]:
        rot = Quaternion(info.get('ego2global_rotation', [1, 0, 0, 0]))
        trans = info.get('ego2global_translation', [0, 0, 0])
        if abs(rot.w - 1.0) > 1e-4 or abs(rot.x) > 1e-4 \
                or abs(rot.y) > 1e-4 or abs(rot.z) > 1e-4:
            return False
        if abs(trans[0]) > 1e-4 or abs(trans[1]) > 1e-4 or abs(trans[2]) > 1e-4:
            return False
    return True


def parse_frames(spec, total):
    """解析 --frames：'a-b' 或 'a,b,c'（默认 0..19）。"""
    if spec is None:
        return list(range(min(20, total)))
    if '-' in spec:
        a, b = spec.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(',') if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize custom FastBEV OCC results (yk_c1, zhito 31 '
                    'classes): det boxes + GT boxes + 6 cameras + Pred/GT occ')
    parser.add_argument('res', help='Path to results_occ.pkl (test.py --out)')
    parser.add_argument('--pkl', type=str, required=True,
                        help='Path to infos pkl (yk_c1_infos_train.pkl)')
    parser.add_argument('--save-path', type=str, default=None,
                        help='保存主流程 6 相机+BEV 拼接图的目录；不传则不'
                             '输出 vis_full（内部仍会生成拼接图作为 3D 导出'
                             'png 的底图）')
    parser.add_argument('--frames', type=str, default=None,
                        help="Frame indices to visualize, e.g. '0-20' or "
                             "'0,5,10' (default: first 20)")
    parser.add_argument('--token', type=str, default=None,
                        help='Visualize the frame with this token')
    parser.add_argument('--show-range', type=int, default=40,
                        help='BEV range in meters (default 40，与 yk_c1 自定义'
                             'point_cloud_range [-40,40] 对齐)')
    parser.add_argument('--canva-size', type=int, default=1000)
    parser.add_argument('--scale-factor', type=int, default=4)
    parser.add_argument('--vis-thred', type=float, default=0.3)
    parser.add_argument('--draw-gt', action='store_true',
                        help='Draw GT boxes (yellow)')
    parser.add_argument('--occ-height-mode', type=str, default='none',
                        choices=['none', 'height', 'column', 'isometric'],
                        help="'none'=平面语义 BEV; "
                             "'height'=按最高 occupied 高度调亮度，z 越高越亮; "
                             "'column'=挤压柱伪3D立体; "
                             "'isometric'=等距斜视（30° 菱形投影，整场景立体）")
    parser.add_argument('--occ-z-scale', type=float, default=1.2,
                        help="柱体高度缩放（column/isometric 模式），越大柱子越高")
    parser.add_argument(
        '--occ-bev-color-mode', type=str, default=None,
        choices=['top', 'bottom', 'majority'],
        help="已废弃：请用 --occ-gt-color-mode 和 --occ-pred-color-mode。"
             "若设置，同时覆盖 GT/Pred 的 color_mode。")
    parser.add_argument(
        '--occ-gt-color-mode', type=str, default='bottom',
        choices=['top', 'bottom', 'majority'],
        help="GT occ BEV 每格语义色取哪一层（默认 bottom）："
             "'bottom'=最低非空层，对稀疏 GT 更容易显示地面；"
             "'top'=最高非空层；'majority'=该列最多次非 0 类别")
    parser.add_argument(
        '--occ-pred-color-mode', type=str, default='top',
        choices=['top', 'bottom', 'majority'],
        help="Pred occ BEV 每格语义色取哪一层（默认 top，与 ply 导出对齐）："
             "'top'=最高非空层（默认）；'bottom'=最低非空层；"
             "'majority'=该列最多次类别")
    parser.add_argument('--occ-gt-mask', action='store_true', default=True,
                        help='GT occ BEV 用 mask_lidar 标注有效观测区域，'
                             '无标注的 FREE 体素显示半透明灰色背景（默认开启）')
    parser.add_argument('--no-occ-gt-mask', dest='occ_gt_mask',
                        action='store_false',
                        help='关闭 GT occ 有效区域灰色背景')
    parser.add_argument('--mask-pred-with-gt', action='store_true',
                        default=True,
                        help='Pred occ 用 GT mask_camera 裁剪：只显示 GT 可观测'
                             '区域的预测，避免无标注空间被乱预测的噪点淹没'
                             '（默认开启；--no-mask-pred-with-gt 关闭）')
    parser.add_argument('--no-mask-pred-with-gt', dest='mask_pred_with_gt',
                        action='store_false',
                        help='Pred occ 不裁剪，显示全空间预测（可能含大量无标注'
                             '区域的无效噪声）')
    parser.add_argument('--no-lidar', action='store_true')

    # ---- Open3D 真 3D（参考 vis_fastbev_occ_gt_custom_3d.py）----
    parser.add_argument('--occ-3d-show', action='store_true',
                        help='Open3D 弹窗交互查看真 3D（鼠标旋转/缩放；'
                             '需 DISPLAY，无显示环境自动跳过）')
    parser.add_argument('--occ-3d-export', action='store_true',
                        help='离屏导出：6 相机 + BEV 俯视 png + occ 体素 ply；'
                             'png 的 BEV 固定 occ_height_mode=height + '
                             'occ_color_mode=top（正俯视投影，与 ply 体素'
                             '集合严格对应）')
    parser.add_argument('--occ-3d-save-path', type=str,
                        default='work_dirs/fastbev-occ-r50-cbgs-custom/'
                                'vis_3d_results')
    parser.add_argument('--occ-3d-front', type=str, default='0.5,-0.75,0.4',
                        help='相机视线方向（x,y,z），默认斜 3/4 视角')
    parser.add_argument('--occ-3d-up', type=str, default='0,0,1',
                        help='相机上方向（x,y,z）')
    parser.add_argument('--occ-3d-lookat', type=str, default='0,0,2',
                        help='相机注视点（x,y,z）')
    parser.add_argument('--occ-3d-zoom', type=float, default=4.0,
                        help='相机缩放（越大越近，默认 4.0）')
    parser.add_argument('--occ-3d-voxel-size', type=float, default=0.2,
                        help='体素显示边长（米）；默认 0.2 = 0.2m 栅格满格'
                             '（旧 0.4m 结果请改传 0.4/0.38）')
    parser.add_argument('--occ-3d-max-z', type=float, default=6.4,
                        help='滤除层中心 z > max_z 的体素（米），对 png 与 '
                             'ply 同时生效：预测 occ 高空常被误检为 '
                             'manmade/vegetation 形成“天棚”，在 png 的 '
                             'height 亮度下地面会呈“双层”发亮，设如 3.0 '
                             '即可从 png/ply 中一并去除，保证两者完全对应；'
                             '默认 6.4 全保留')
    parser.add_argument('--occ-3d-lidar', action='store_true',
                        help='在 3D 场景中叠加 lidar 点云（按高度着色）')
    parser.add_argument('--occ-3d-boxes', action='store_true',
                        help='在 3D 场景中叠加 3D 检测框（pred 青 / GT 黄）')
    parser.add_argument('--format', type=str, default='image',
                        choices=['image', 'video'])
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--video-prefix', type=str, default='vis_results')
    return parser.parse_args()


def main():
    args = parse_args()

    results = pickle.load(open(args.res, 'rb'))
    dataset = pickle.load(open(args.pkl, 'rb'))
    # 关键：results 的顺序与 dataset 内部 data_infos 一致，而 data_infos 在
    # mmdet3d/datasets/nuscenes_dataset.py::load_annotations 中按 timestamp 升序排序
    # （data_infos = sorted(data['infos'], key=lambda e: e['timestamp'])），
    # 与 ann_file 里 infos 的原始顺序不同。必须同样排序后 index 才能一一对应，
    # 否则预测框/occ 与 GT/点云/相机图错位（表现为框与真值不对齐）。
    infos_list = sorted(dataset['infos'], key=lambda e: e['timestamp'])
    print(f'loaded {len(results)} results, {len(infos_list)} infos '
          f'(sorted by timestamp)')

    # 自定义数据（ego=global）的 FastBEV 检测输出已在 lidar 系（与 GT 同系），
    # 跳过 ego→lidar 转换；否则会重复施加 lidar2ego_translation（z -1.594m，
    # x -1.535m），导致 Pred 框相对 GT 整体错位（高度明显偏低）。
    use_lidar_frame = _is_ego2global_identity(infos_list)
    print(f'det box frame: {"lidar (custom, skip ego->lidar transform)" if use_lidar_frame else "ego (nuScenes, transform to lidar)"}')

    # 帧选择
    if args.token is not None:
        frames = [i for i, inf in enumerate(infos_list)
                  if inf['token'] == args.token]
        if not frames:
            raise SystemExit(f'token {args.token} not found')
        print(f'token {args.token} -> frame {frames[0]}')
    else:
        frames = parse_frames(args.frames, len(results))
    frames = [f for f in frames if f < len(results)]
    print(f'visualizing {len(frames)} frame(s): {frames}')

    views = [c for c in CAM_ORDER if c in infos_list[0]['cams']]
    scale_factor = args.scale_factor
    canva_size = args.canva_size
    show_range = args.show_range

    # Open3D 真 3D 相机参数
    occ3d_front = _parse_vec3(args.occ_3d_front, (0.5, -0.75, 0.4))
    occ3d_up = _parse_vec3(args.occ_3d_up, (0, 0, 1))
    occ3d_lookat = _parse_vec3(args.occ_3d_lookat, (0, 0, 2))

    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
    vout = None
    if args.format == 'video':
        if not args.save_path:
            raise SystemExit('--format video 需要 --save-path')
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        vout = cv2.VideoWriter(
            os.path.join(args.save_path, args.video_prefix + '.mp4'),
            fourcc, args.fps,
            (int(1600 / scale_factor * 3),
             int(900 / scale_factor * 2 + canva_size)))

    # GT / Pred 独立取色模式（--occ-bev-color-mode 显式指定时统一覆盖）
    occ_color_mode_left = args.occ_gt_color_mode
    occ_color_mode_right = args.occ_pred_color_mode
    if args.occ_bev_color_mode is not None:
        occ_color_mode_left = args.occ_bev_color_mode
        occ_color_mode_right = args.occ_bev_color_mode

    for fi, idx in enumerate(frames):
        infos = infos_list[idx]
        res = results[idx]

        # ---- 预测框 ----
        # FastBEV 的 BEV 特征与检测输出在 ego 系（自车中心），而 GT/点云在 lidar 系。
        # nuscenes_dataset.py::_format_bbox 中也是按 ego->ego2global 转 json。
        # 这里需将预测框 corners 从 ego 系转回 lidar 系：p_lidar = inv(lidar2ego) @ p_ego
        boxes = res['pts_bbox']['boxes_3d']
        scores = np.array(res['pts_bbox']['scores_3d'], dtype=np.float32)
        corners_lidar = np.zeros((0, 3), dtype=np.float32)
        pred_flag = np.zeros(0, dtype=bool)
        if boxes.tensor.shape[0] > 0:
            corners = boxes.corners.numpy().reshape(-1, 3)
            if use_lidar_frame:
                # 自定义数据：Pred 已在 lidar 系，直接用（不再二次转换）
                corners_lidar = corners
            else:
                # nuScenes：FastBEV 检测输出在 ego 系，需转回 lidar 系
                lidar2ego = np.eye(4, dtype=np.float32)
                lidar2ego[:3, :3] = Quaternion(
                    infos['lidar2ego_rotation']).rotation_matrix
                lidar2ego[:3, 3] = infos['lidar2ego_translation']
                corners_lidar = (np.concatenate(
                    [corners,
                     np.ones((corners.shape[0], 1), dtype=np.float32)],
                    axis=1) @ np.linalg.inv(lidar2ego).T)[:, :3]
            pred_flag = np.ones(boxes.tensor.shape[0], dtype=bool)

        # ---- GT 框（可选）----
        # yk_c1 pkl 的 gt_boxes 为 (N,7) ndarray，mmdet3d LiDAR 约定
        # (x,y,z,w,l,h,yaw)，直接构造成 LiDARInstance3DBoxes 即可。
        gt_boxes = infos.get('gt_boxes')
        if args.draw_gt and gt_boxes is not None and len(gt_boxes) > 0:
            gt_boxes = gt_boxes.copy()
            corners_lidar_gt = \
                LB(gt_boxes,
                   origin=(0.5, 0.5, 0.5)).corners.numpy().reshape(-1, 3)
            corners_lidar = np.concatenate([corners_lidar, corners_lidar_gt],
                                           axis=0)
            gt_flag = np.ones((corners_lidar_gt.shape[0] // 8), dtype=bool)
            pred_flag = np.concatenate(
                [pred_flag, np.logical_not(gt_flag)], axis=0)
            scores = np.concatenate([
                scores,
                np.zeros(infos['gt_boxes'].shape[0], dtype=np.float32)
            ])

        # ---- occ（预测，与 GT 同栅格：ego 系 (Dx,Dy,Dz) [X,Y,Z]）----
        occ = res.get('occ')

        # ---- occ（GT 真值，用于中间 BEV 左块对比）----
        occ_gt = None
        occ_gt_mask = None
        occ_pred_mask = None
        occ_path = infos.get('occ_path')
        if occ_path and os.path.exists(os.path.join(occ_path, 'labels.npz')):
            occ_labels = np.load(os.path.join(occ_path, 'labels.npz'))
            occ_gt = occ_labels['semantics']
            if args.occ_gt_mask and 'mask_lidar' in occ_labels.files:
                occ_gt_mask = occ_labels['mask_lidar']
            # Pred 裁剪 mask：与训练/评估一致的相机可见区域（mask_camera）
            if args.mask_pred_with_gt and 'mask_camera' in occ_labels.files:
                occ_pred_mask = occ_labels['mask_camera']

        # occ 体素中心（ego → lidar 逆变换），预测与 GT 同栅格共用；
        # 栅格由实际数组形状决定（当前 0.2m 方案: 400x400x16），
        # 注意：预测与 GT 须为同一栅格，混用会索引错位。
        occ_shape = (occ if occ is not None else occ_gt).shape
        centers_lidar = None
        if occ is not None or occ_gt is not None:
            centers_lidar = occ_centers_lidar(infos, occ_shape)

        # ---- 排序（与 vis.py 一致：低分框先画，高分框后画覆盖）----
        scores = np.array(scores, dtype=np.float32)
        sort_ids = np.argsort(scores)

        # 按 sort_ids 重排 corners / pred_flag / scores
        if corners_lidar.shape[0] > 0:
            n_inst = corners_lidar.shape[0] // 8
            corners_lidar = corners_lidar.reshape(n_inst, 8, 3)
            corners_lidar = corners_lidar[sort_ids].reshape(-1, 3)
            pred_flag = pred_flag[sort_ids]
            scores = scores[sort_ids]

        # ---- image view ----
        imgs = draw_camera_images(infos, corners_lidar, pred_flag,
                                  views, scale_factor)

        # ---- bird-eye-view ----
        # draw_bev_canvas 内部会就地翻转 corners 的 y，提前保存 3D 用原始坐标
        corners_lidar_3d = corners_lidar.copy()
        # 拆分 GT / Pred 框（供左右分块 BEV 使用：左块 GT 框，右块全部框）
        if corners_lidar.shape[0] > 0:
            n_inst = corners_lidar.shape[0] // 8
            flags = pred_flag.reshape(n_inst)
            gt_sel = np.logical_not(flags)
            corners_gt = corners_lidar.reshape(
                n_inst, 8, 3)[gt_sel].reshape(-1, 3)
        else:
            corners_gt = corners_lidar
        # 中间 BEV 区域左右分块：左=GT occ 真值，右=Pred occ 预测
        bev = draw_bev_split_canvas(
            canva_size, show_range, occ_left=occ_gt, occ_right=occ,
            corners_gt=corners_gt, corners_all=corners_lidar,
            pred_flag=pred_flag, scores=scores, vis_thred=args.vis_thred,
            lidar_path=infos.get('lidar_path'),
            draw_lidar=not args.no_lidar,
            occ_height_mode=args.occ_height_mode,
            occ_z_scale=args.occ_z_scale,
            centers_lidar=centers_lidar,
            occ_left_mask=occ_gt_mask,
            occ_right_mask=occ_pred_mask,
            occ_color_mode_left=occ_color_mode_left,
            occ_color_mode_right=occ_color_mode_right)

        # ---- fuse image-view and bev（与 vis.py 布局一致，BEV 左右分块）----
        img = np.zeros((900 * 2 + canva_size * scale_factor, 1600 * 3, 3),
                       dtype=np.uint8)
        img[:900, :, :] = np.concatenate(imgs[:3], axis=1)
        img_back = np.concatenate(
            [imgs[3][:, ::-1, :], imgs[4][:, ::-1, :], imgs[5][:, ::-1, :]],
            axis=1)
        img[900 + canva_size * scale_factor:, :, :] = img_back
        img = cv2.resize(img, (int(1600 / scale_factor * 3),
                               int(900 / scale_factor * 2 + canva_size)))
        w_begin = int((1600 * 3 / scale_factor - canva_size) // 2)
        y_begin = int(900 / scale_factor)
        # 两块 half×half 画布并排成横条，垂直居中贴回原 BEV 区域
        h_b = bev.shape[0]
        img[y_begin + (canva_size - h_b) // 2:
            y_begin + (canva_size - h_b) // 2 + h_b,
            w_begin:w_begin + canva_size, :] = bev

        if args.save_path and args.format == 'image':
            out = os.path.join(args.save_path, f'{idx:06d}.jpg')
            cv2.imwrite(out, img)
            print(f'[{fi + 1}/{len(frames)}] saved {out}')
        elif vout is not None:
            vout.write(img)
            print(f'[{fi + 1}/{len(frames)}] wrote frame {idx}')

        # ---- Open3D 真 3D（参考 vis_fastbev_occ_gt_custom_3d.py）----
        if args.occ_3d_show or args.occ_3d_export:
            # png 与 ply 使用同一截断 occ，保证两者体素集完全一致
            occ_3d = _truncate_occ_z(occ, args.occ_3d_max_z)
            # 3D 同样裁剪到 GT 可观测区域（mask_camera），避免无效噪点体素
            if occ_3d is not None and occ_pred_mask is not None:
                m = occ_pred_mask.astype(bool)
                if m.ndim == 2:
                    m = m[..., None]
                occ_3d = np.where(m, occ_3d, 0)
            geoms = []
            if occ_3d is not None:
                geoms += occ_to_3d_meshes(
                    occ_3d, voxel_size=args.occ_3d_voxel_size,
                    centers_lidar=centers_lidar)
            if args.occ_3d_lidar:
                lp = infos.get('lidar_path')
                if lp and os.path.exists(lp):
                    geoms.append(('lidar', lidar_to_3d_pcd(lp)))
            if args.occ_3d_boxes and corners_lidar_3d.shape[0] > 0:
                geoms.append(('boxes',
                              boxes_to_3d_lineset(corners_lidar_3d,
                                                  pred_flag)))
            if not geoms:
                print('[3D] 无可渲染内容（occ 为空且未开 '
                      '--occ-3d-lidar/--occ-3d-boxes）')
            else:
                if args.occ_3d_show:
                    show_3d_window(geoms, occ3d_front, occ3d_up,
                                   occ3d_lookat, args.occ_3d_zoom)
                if args.occ_3d_export:
                    export_3d_png_ply(
                        geoms, args.occ_3d_save_path, f'{idx:06d}',
                        full_img=img, occ=occ_3d, occ_gt=occ_gt,
                        lidar_path=infos.get('lidar_path'),
                        corners_lidar=corners_lidar_3d.copy(),
                        pred_flag=pred_flag, scores=scores,
                        vis_thred=args.vis_thred,
                        draw_lidar=not args.no_lidar,
                        canva_size=canva_size, show_range=show_range,
                        scale_factor=scale_factor,
                        occ_color_mode=args.occ_bev_color_mode,
                        centers_lidar=centers_lidar)

    if vout is not None:
        vout.release()
    print('Done.')


if __name__ == '__main__':
    main()
