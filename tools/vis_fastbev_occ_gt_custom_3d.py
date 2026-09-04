#!/usr/bin/env python3
"""
可视化自定义数据（yk_c1，zhito 标注规范 v1.2 共 31 类）GT：
激光点云、GT 3D 检测框、6 通道相机图像、GT occ 真值。

本脚本由 tools/vis_fastbev_occ_gt_3d.py 适配而来，针对自定义数据修改了三处：
  1) lidar bin 为 6 列 (x,y,z,intensity,semantic,instance)，读取时取前 3 列；
  2) occ 语义为 zhito 31 类（0=无标注/忽略），颜色表扩展为 32 行，无 free 类；
  3) 无标注(0)体素在 BEV/3D 渲染中按空处理（BEV 过滤、3D 不生成网格）。

布局（与 vis_fastbev_occ_gt_3d.py 一致）：
  - 上方：FRONT_LEFT, FRONT, FRONT_RIGHT 三相机拼接
  - 中间：BEV 俯视图（occ GT + lidar 点云 + GT 3D 框）
  - 下方：BACK_LEFT, BACK, BACK_RIGHT 三相机拼接（水平镜像）

颜色：
  - GT 框 = (255, 255, 0) 黄色
  - occ 语义 = 32 类语义色（索引 0=ignore 无标注，1~31=zhito 31 类）

用法（conda bev 环境，需在项目根目录运行）：
    source /home/server/miniconda3/etc/profile.d/conda.sh && conda activate bev

    # 图片：前 10 帧
    python tools/vis_fastbev_occ_gt_custom_3d.py \
        --pkl yk_c1_infos_train.pkl \
        --frames 0-9 \
        --save-path work_dirs/vis_gt_yk

    # 指定 token 可视化单帧
    python tools/vis_fastbev_occ_gt_custom_3d.py \
        --pkl yk_c1_infos_train.pkl \
        --token <token> \
        --save-path work_dirs/vis_gt_yk

    # 输出视频
    python tools/vis_fastbev_occ_gt_custom_3d.py \
        --pkl yk_c1_infos_train.pkl \
        --frames 0-199 --format video --fps 10 \
        --save-path work_dirs/vis_gt_yk

    # Open3D 真 3D：离屏导出固定视角 png + occ 体素 ply（headless 可用）
    python tools/vis_fastbev_occ_gt_custom_3d.py \
        --pkl yk_c1_infos_train.pkl \
        --frames 0-199 --occ-3d-export --occ-3d-lidar --occ-3d-boxes \
        --occ-3d-save-path work_dirs/vis_gt_3d_1

    # Open3D 真 3D：弹窗交互查看（鼠标旋转/缩放，需 DISPLAY）
    python tools/vis_fastbev_occ_gt_custom_3d.py \
        --pkl yk_c1_infos_train.pkl \
        --token <token> --occ-3d-show
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

# zhito 标注规范 v1.2：0=无标注/ignore，1~12 A 类目标，13~31 B 类语义
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
# 参照 nuScenes/Cityscapes 语义配色惯例
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
OCC_SIZE = 400
OCC_Z = 16
OCC_RANGE = 40.0  # occ 栅格覆盖 [-40, 40]m
# z 起点偏移：与 create_occ_gt_custom.py 的 OCC_Z=[-0.5, 2.7, 0.2] 一致
OCC_Z_OFFSET = -0.5

# 相机显示顺序（与 vis.py 完全一致）
CAM_ORDER = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
]

# 颜色映射：0=GT(黄)  —— 与 vis.py 的 color_map 一致
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


def occ_centers_lidar(infos):
    """occ 体素中心在激光(lidar)系的坐标 (400,400,16,3)。

    create_occ_gt_custom.py 生成 occ 时先把点云从 lidar 系转到 ego 系
    （lidar2ego）再体素化，因此 labels.npz 的体素坐标在 ego 系；
    这里用 lidar2ego 的逆变换转回 lidar 系，与 GT 框 / lidar 点云同系显示。

    体素中心 (gx=X, gy=Y, gz=Z) 的 ego 系坐标（与 OCC_SHAPE=(Dx,Dy,Dz) 对齐）：
        x_ego = (gx + 0.5) * res - OCC_RANGE
        y_ego = (gy + 0.5) * res - OCC_RANGE
        z_ego = (gz + 0.5) * res + OCC_Z_OFFSET
    lidar2ego: p_ego = R @ p_lidar + t  →  p_lidar = R.T @ (p_ego - t) = (p_ego - t) @ R
    """
    res = 2 * OCC_RANGE / OCC_SIZE
    g = (np.arange(OCC_SIZE) + 0.5) * res - OCC_RANGE
    zc = (np.arange(OCC_Z) + 0.5) * res + OCC_Z_OFFSET
    gxx, gyy, gzz = np.meshgrid(g, g, zc, indexing='ij')
    ego = np.stack([gxx, gyy, gzz], axis=-1)           # (400,400,16,3)
    rot = Quaternion(infos['lidar2ego_rotation']).rotation_matrix
    t = np.array(infos['lidar2ego_translation'], dtype=np.float32)
    return (ego - t) @ rot


def occ_to_bev_rgb(occ, height_mode='none'):
    """occ (400,400,16) [X,Y,Z] -> BEV 俯视 RGB 图 (400,400,3)。
    occ 维度排列与 create_occ_gt_custom.py 的 OCC_SHAPE=(Dx,Dy,Dz) 一致：
    dim0=X(前向), dim1=Y(左向), dim2=Z(高度)，即 occ[gx, gy, gz]。

    自定义 31 类体系：0=无标注/忽略（FREE_ID），视为空体素。

    height_mode:
      - 'none':   对每个 (x,y) 取最低非空体素层（贴近地面），无有效体素显示为黑。
                  这样可与 lidar 地面点更好对齐，避免高层 terrain/manmade 遮挡地面结构。
      - 'height': 保持地面语义颜色，但按该列最高 occupied 体素高度调整亮度，
                  z 越高越亮，从而在 BEV 俯视图中体现立体高度信息。
    """
    valid = (~_occ_empty(occ)).astype(np.int32)          # (400,400,16)
    # 最低非空层（用于语义颜色）
    z_bottom = np.argmax(valid, axis=2)                # (400,400)
    # 最高非空层（用于高度编码）
    z_top = (OCC_Z - 1) - np.argmax(valid[..., ::-1], axis=2)
    no_occ = valid.sum(axis=2) == 0
    z_top = np.where(no_occ, 0, z_top)

    sem_bev = occ[np.arange(OCC_SIZE)[:, None], np.arange(OCC_SIZE)[None, :],
                  z_bottom]                            # (400,400)
    rgb = COLOR_MAP[sem_bev].astype(np.float32)        # (400,400,3)

    if height_mode == 'height':
        # z 越高越亮，factor 范围 [0.35, 1.0]
        factor = 0.35 + 0.65 * (z_top.astype(np.float32) / (OCC_Z - 1))
        rgb = rgb * factor[..., None]
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        rgb = rgb.astype(np.uint8)
    return rgb


def occ_to_extrusion(occ, canva_size, show_range, centers_lidar=None,
                     z_scale=1.2, cell_pad=0.9):
    """把 occ (400,400,16) 渲染为 BEV 上的挤压柱画布 (N,N,3)，伪 3D 立体。

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
    cell = occ_h / float(OCC_SIZE)
    margin = int((canva_size - occ_h) / 2)
    half = cell * cell_pad / 2.0

    valid = (~_occ_empty(occ)).astype(np.int32)          # (400,400,16)
    z_bottom = np.argmax(valid, axis=2)                # 最低非空层(语义)
    z_top = (OCC_Z - 1) - np.argmax(valid[..., ::-1], axis=2)
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
        rows = margin + (OCC_SIZE - 1 - ys) * cell + half   # X 大 → row 小(上)
        cols = margin + (OCC_SIZE - 1 - xs) * cell + half   # Y 大 → col 小(左)
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
    """把 occ (400,400,16) 渲染为等距斜视画布 (N,N,3)，伪 3D 立体。

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
    cell = occ_h / float(OCC_SIZE)
    margin = int((canva_size - occ_h) / 2)
    half = cell * cell_pad / 2.0
    pad = 0.02 * N

    valid = (~_occ_empty(occ)).astype(np.int32)          # (400,400,16)
    z_bottom = np.argmax(valid, axis=2)                # 最低非空层(语义)
    z_top = (OCC_Z - 1) - np.argmax(valid[..., ::-1], axis=2)
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
        rows = margin + (OCC_SIZE - 1 - ys) * cell + half   # X 大 → row 小(上)
        cols = margin + (OCC_SIZE - 1 - xs) * cell + half   # Y 大 → col 小(左)
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


# ------------------------- Open3D 真 3D（方案 C） -------------------------

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


def occ_to_3d_meshes(occ, voxel_size=0.38, centers_lidar=None):
    """把 occ (400,400,16) [X,Y,Z] 转换为语义着色的立方体网格列表。

    occ 维度排列 dim0=X(前向)、dim1=Y(左向)、dim2=Z，与 create_occ_gt_custom.py
    的 OCC_SHAPE=(Dx,Dy,Dz) 一致；世界坐标采用 FLU lidar 系（x 前、y 左、z 上），
    与 2D BEV 画布、lidar 点云、GT 框同系。传 centers_lidar 时直接使用其
    体素中心（已做 ego→lidar 逆变换，正确处理 lidar 安装高度等偏移）。

    z 映射按当前 0.2m / z∈[-0.5,2.7]m / Dz=16：层 k 中心 = k*res + res/2 + OCC_Z_OFFSET：
        x = gx*res + res/2 - OCC_RANGE   (gx: dim0=X 索引)
        y = gy*res + res/2 - OCC_RANGE   (gy: dim1=Y 索引)
        z = gz*res + res/2 + OCC_Z_OFFSET (gz: dim2=Z 索引)
    自定义 31 类体系：0=无标注/忽略(FREE_ID)不生成网格。
    每个非空体素生成一个立方体，按语义类别分组为独立 TriangleMesh
    （每类 paint_uniform_color），便于渲染与 ply 导出。
    """
    res = 2 * OCC_RANGE / OCC_SIZE
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
            gz * res + res / 2 + OCC_Z_OFFSET,
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

    自定义数据 bin 为 6 列 (x,y,z,intensity,semantic,instance)，取前 3 列。
    """
    pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 6)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(
        _depth_to_colors(pts[:, 2]).astype(np.float64) / 255.0)
    return pcd


def boxes_to_3d_lineset(corners_lidar):
    """GT 框 8 角点(FLU) -> 12 边 LineSet（黄色）。

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
    ls.points = o3d.utility.Vector3dVector(corners.reshape(-1, 3).astype(np.float64))
    ls.lines = o3d.utility.Vector2iVector(np.concatenate(lines, axis=0))
    ls.paint_uniform_color([1.0, 1.0, 0.0])
    return ls


def show_3d_window(geoms, front, up, lookat, zoom):
    """弹窗交互查看（鼠标旋转/缩放；需 DISPLAY，headless 下提示跳过）。"""
    if not os.environ.get('DISPLAY'):
        print('[3D] 无 DISPLAY，跳过交互弹窗；可用 --occ-3d-export 离屏导出')
        return
    o3d.visualization.draw_geometries(
        [g for _, g in geoms], window_name='FastBEV GT occ 3D (custom)',
        width=1600, height=1000,
        front=list(front), lookat=list(lookat), up=list(up), zoom=zoom)


def export_3d_png_ply(geoms, out_dir, prefix, full_img=None, occ=None,
                      lidar_path=None, corners_lidar=None, draw_lidar=True,
                      canva_size=1000, show_range=50, scale_factor=4,
                      centers_lidar=None):
    """导出拼接 png（6 相机 + BEV 俯视）+ occ 体素 ply。

    png 复用主流程的完整拼接图 full_img（上 3 相机 + 下 3 相机），
    中间的 BEV 区域用 --occ-height-mode height 方式重绘后贴回，
    整幅图与 --occ-height-mode height 布局一致；
    ply 仍为真 3D occ 体素网格（由 geoms 中 occ_* 网格合并导出）。
    """
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f'{prefix}.png')
    try:
        n_inst = 0 if corners_lidar is None else corners_lidar.shape[0] // 8
        canvas = draw_bev_canvas(
            canva_size, show_range, occ=occ,
            corners_lidar=corners_lidar,
            pred_flag=np.zeros(n_inst, dtype=bool),
            scores=np.ones(n_inst, dtype=np.float32),
            vis_thred=0.3,
            lidar_path=lidar_path, draw_lidar=draw_lidar,
            occ_height_mode='height',
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
                out_img[y_begin:y_begin + canva_size,
                        w_begin:w_begin + canva_size, :] = canvas
            else:
                out_img = canvas
        else:
            out_img = canvas
        cv2.imwrite(png_path, out_img)
        print(f'[3D] saved {png_path} (6cam + BEV height)')
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


# ------------------------- 绘制 -------------------------

def draw_camera_images(infos, corners_lidar, pred_flag, views, scale_factor):
    """绘制 6 相机图（投影 3D 框），返回列表 img（已 resize）。"""
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
                    centers_lidar=None):
    """绘制 BEV 画布：occ 底图 + lidar 点 + 检测框。

    坐标系与 vis_custom_results.py 完全一致（FLU lidar，地图视角）：
      col = (-y + show_range) / (2*show_range) * canvas   (y左→col左)
      row = (-x + show_range) / (2*show_range) * canvas   (x前→row上)
    即：前向(x+) 朝画布上方，左向(y+) 朝画布左方，FRONT 箭头与坐标系自洽。

    occ 维度排列 dim0=X(前向)、dim1=Y(左向)，且真值在 ego 系生成；
    传 centers_lidar（occ_centers_lidar 输出，已做 ego→lidar 逆变换）时
    occ 以激光坐标系与 lidar 点云 / GT 框精确对齐显示。
    """
    canvas = np.zeros((int(canva_size), int(canva_size), 3), dtype=np.uint8)

    # 1) occ 底图
    #    occ 维度 (dim0=X 前向, dim1=Y 左向)，坐标以 lidar 系显示
    #    贴图: occ_rgb[gx, gy] -> occ_bev[row, col]，row=-x、col=-y → 180° 翻转
    proj_fn = None
    if occ is not None:
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
            occ_rgb = occ_to_bev_rgb(occ, height_mode=occ_height_mode)
            occ_h = int(canva_size * OCC_RANGE / show_range)
            occ_bev = cv2.resize(
                np.ascontiguousarray(occ_rgb[::-1, ::-1]),
                (occ_h, occ_h),
                interpolation=cv2.INTER_NEAREST)
            margin = int((canva_size - occ_h) / 2)
            # 无标注区域(黑/深色)改为透明，只保留有语义的区域
            mask = (occ_bev.sum(axis=2) > 60 * 3).astype(np.uint8) * 255
            mask = cv2.resize(mask, (occ_h, occ_h), interpolation=cv2.INTER_NEAREST)
            occ_bev_masked = cv2.bitwise_and(occ_bev, occ_bev, mask=mask)
            # occ 在 ego 系生成，按 centers_lidar 推算整幅贴图相对画布中心的
            # 平移量，使 occ 与 lidar 系点云/框对齐（occ 栅格中心即 (-t_x, -t_y)）
            dx = dy = 0
            if centers_lidar is not None:
                c0 = centers_lidar[OCC_SIZE // 2, OCC_SIZE // 2, 0, :2]
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

    # 2) lidar 点云（自定义 bin 为 6 列，取前 3 列；投影同 vis_custom_results）
    if draw_lidar and lidar_path and os.path.exists(lidar_path):
        lidar_points = np.fromfile(lidar_path, dtype=np.float32)
        lidar_points = lidar_points.reshape(-1, 6)[:, :3]
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
        for i in range(px.shape[0]):
            color = depth2color(zs[i])
            px2, py2 = float(px[i]), float(py[i])
            if proj_fn is not None:
                px2, py2 = proj_fn(px2, py2)
            cv2.circle(canvas, (int(px2), int(py2)),
                       radius=0, color=color, thickness=1)

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
        description='Visualize custom FastBEV OCC GT (yk_c1, zhito 31 classes): '
                    'lidar + GT boxes + 6 cameras + GT occ')
    parser.add_argument('--pkl', type=str, required=True,
                        help='Path to infos pkl '
                             '(yk_c1_infos_train.pkl)')
    parser.add_argument('--save-path', type=str,
                        default='work_dirs/vis_gt')
    parser.add_argument('--frames', type=str, default=None,
                        help="Frame indices to visualize, e.g. '0-199' or "
                             "'0,5,10' (default: first 20)")
    parser.add_argument('--token', type=str, default=None,
                        help='Visualize the frame with this token')
    parser.add_argument('--show-range', type=int, default=50,
                        help='BEV range in meters (default 50)')
    parser.add_argument('--canva-size', type=int, default=1000)
    parser.add_argument('--scale-factor', type=int, default=4)
    parser.add_argument('--vis-thred', type=float, default=0.1)
    parser.add_argument('--occ-height-mode', type=str, default='none',
                        choices=['none', 'height', 'column', 'isometric'],
                        help="'none'=平面语义 BEV; "
                             "'height'=按最高 occupied 高度调亮度，z 越高越亮; "
                             "'column'=挤压柱伪3D立体; "
                             "'isometric'=等距斜视（30° 菱形投影，整场景立体）")
    parser.add_argument('--occ-z-scale', type=float, default=1.2,
                        help="柱体高度缩放（column/isometric 模式），越大柱子越高")
    parser.add_argument('--no-lidar', action='store_true')
    parser.add_argument('--no-occ', action='store_true',
                        help='Skip loading/visualizing GT occ')

    # ---- Open3D 真 3D（方案 C）----
    parser.add_argument('--occ-3d-show', action='store_true',
                        help='Open3D 弹窗交互查看真 3D（鼠标旋转/缩放；'
                             '需 DISPLAY，无显示环境自动跳过）')
    parser.add_argument('--occ-3d-export', action='store_true',
                        help='离屏导出：6 相机 + BEV 俯视 png（与 '
                             '--occ-height-mode height 布局一致）+ occ 体素 ply')
    parser.add_argument('--occ-3d-save-path', type=str,
                        default='work_dirs/vis_gt_3d')
    parser.add_argument('--occ-3d-front', type=str, default='0.5,-0.75,0.4',
                        help='相机视线方向（x,y,z），默认斜 3/4 视角')
    parser.add_argument('--occ-3d-up', type=str, default='0,0,1',
                        help='相机上方向（x,y,z）')
    parser.add_argument('--occ-3d-lookat', type=str, default='0,0,2',
                        help='相机注视点（x,y,z）')
    parser.add_argument('--occ-3d-zoom', type=float, default=4.0,
                        help='相机缩放（越大越近，默认 4.0）')
    parser.add_argument('--occ-3d-voxel-size', type=float, default=0.2,
                        help='体素显示边长（米），默认 0.2 满格（与 0.2m occ GT 一致）')
    parser.add_argument('--occ-3d-lidar', action='store_true',
                        help='在 3D 场景中叠加 lidar 点云（按高度着色）')
    parser.add_argument('--occ-3d-boxes', action='store_true',
                        help='在 3D 场景中叠加 GT 3D 检测框（黄色）')
    parser.add_argument('--format', type=str, default='image',
                        choices=['image', 'video'])
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--video-prefix', type=str, default='vis_gt')
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = pickle.load(open(args.pkl, 'rb'))
    # 与 dataset 内部一致：按 timestamp 升序排序
    infos_list = sorted(dataset['infos'], key=lambda e: e['timestamp'])
    print(f'loaded {len(infos_list)} infos (sorted by timestamp)')

    # 帧选择
    if args.token is not None:
        frames = [i for i, inf in enumerate(infos_list)
                  if inf['token'] == args.token]
        if not frames:
            raise SystemExit(f'token {args.token} not found')
        print(f'token {args.token} -> frame {frames[0]}')
    else:
        frames = parse_frames(args.frames, len(infos_list))
    frames = [f for f in frames if f < len(infos_list)]
    print(f'visualizing {len(frames)} frame(s): {frames}')

    views = [c for c in CAM_ORDER if c in infos_list[0]['cams']]
    scale_factor = args.scale_factor
    canva_size = args.canva_size
    show_range = args.show_range

    # Open3D 真 3D 相机参数
    occ3d_front = _parse_vec3(args.occ_3d_front, (0.5, -0.75, 0.4))
    occ3d_up = _parse_vec3(args.occ_3d_up, (0, 0, 1))
    occ3d_lookat = _parse_vec3(args.occ_3d_lookat, (0, 0, 1))

    os.makedirs(args.save_path, exist_ok=True)
    vout = None
    if args.format == 'video':
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        vout = cv2.VideoWriter(
            os.path.join(args.save_path, args.video_prefix + '.mp4'),
            fourcc, args.fps,
            (int(1600 / scale_factor * 3),
             int(900 / scale_factor * 2 + canva_size)))

    for fi, idx in enumerate(frames):
        infos = infos_list[idx]

        # ---- GT 框 ----
        # yk_c1 pkl 的 gt_boxes 为 (N,7) ndarray，mmdet3d LiDAR 约定
        # (x,y,z,w,l,h,yaw)，直接构造成 LiDARInstance3DBoxes 即可。
        gt_boxes = infos.get('gt_boxes')
        corners_lidar = np.zeros((0, 3), dtype=np.float32)
        pred_flag = np.zeros(0, dtype=bool)
        scores = np.zeros(0, dtype=np.float32)

        if gt_boxes is not None and len(gt_boxes) > 0:
            gt_boxes = gt_boxes.copy()
            corners_lidar_gt = \
                LB(gt_boxes,
                   origin=(0.5, 0.5, 0.5)).corners.numpy().reshape(-1, 3)
            corners_lidar = corners_lidar_gt
            # pred_flag=0 表示 GT（黄色）
            pred_flag = np.zeros(gt_boxes.shape[0], dtype=bool)
            scores = np.ones(gt_boxes.shape[0], dtype=np.float32)

        # ---- GT occ ----
        # occ 维度 (dim0=X 前向, dim1=Y 左向, dim2=Z)，与 create_occ_gt_custom.py
        # 的 OCC_SHAPE=(Dx,Dy,Dz) 一致；真值在 ego 系生成，需转回激光系显示
        occ = None
        centers_lidar = None
        if not args.no_occ:
            occ_path = infos.get('occ_path')
            if occ_path and os.path.exists(os.path.join(occ_path, 'labels.npz')):
                occ_gt = np.load(os.path.join(occ_path, 'labels.npz'))
                occ = occ_gt['semantics']  # (400, 400, 16) [X, Y, Z]
                centers_lidar = occ_centers_lidar(infos)

        # ---- 排序（低分框先画，高分框后画覆盖）----
        if corners_lidar.shape[0] > 0:
            sort_ids = np.argsort(scores)
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
        canvas = draw_bev_canvas(
            canva_size, show_range, occ=occ,
            corners_lidar=corners_lidar, pred_flag=pred_flag,
            scores=scores, vis_thred=args.vis_thred,
            lidar_path=infos.get('lidar_path'),
            draw_lidar=not args.no_lidar,
            occ_height_mode=args.occ_height_mode,
            occ_z_scale=args.occ_z_scale,
            centers_lidar=centers_lidar)

        # ---- fuse image-view and bev（与 vis.py 布局完全一致）----
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
        img[int(900 / scale_factor):int(900 / scale_factor) + canva_size,
            w_begin:w_begin + canva_size, :] = canvas

        if args.format == 'image':
            out = os.path.join(args.save_path, f'{idx:06d}.jpg')
            cv2.imwrite(out, img)
            print(f'[{fi + 1}/{len(frames)}] saved {out}')
        else:
            vout.write(img)
            if (fi + 1) % 10 == 0:
                print(f'[{fi + 1}/{len(frames)}] wrote frame {idx}')

        # ---- Open3D 真 3D（方案 C）----
        if args.occ_3d_show or args.occ_3d_export:
            geoms = []
            if occ is not None:
                geoms += occ_to_3d_meshes(
                    occ, voxel_size=args.occ_3d_voxel_size,
                    centers_lidar=centers_lidar)
            if args.occ_3d_lidar:
                lp = infos.get('lidar_path')
                if lp and os.path.exists(lp):
                    geoms.append(('lidar', lidar_to_3d_pcd(lp)))
            if args.occ_3d_boxes and corners_lidar_3d.shape[0] > 0:
                geoms.append(('boxes',
                              boxes_to_3d_lineset(corners_lidar_3d)))
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
                        full_img=img, occ=occ,
                        lidar_path=infos.get('lidar_path'),
                        corners_lidar=corners_lidar_3d.copy(),
                        draw_lidar=not args.no_lidar,
                        canva_size=canva_size, show_range=show_range,
                        scale_factor=scale_factor,
                        centers_lidar=centers_lidar)

    if vout is not None:
        vout.release()
    print('Done.')


if __name__ == '__main__':
    main()
