#!/usr/bin/env python3
"""
可视化 nuScenes 前 N 帧 GT 数据：激光点云、GT 3D 检测框、6 通道相机图像、GT occ 真值。

布局（与 vis_fastbev_occ_results.py 一致）：
  - 上方：FRONT_LEFT, FRONT, FRONT_RIGHT 三相机拼接
  - 中间：BEV 俯视图（occ GT + lidar 点云 + GT 3D 框）
  - 下方：BACK_LEFT, BACK, BACK_RIGHT 三相机拼接（水平镜像）

颜色：
  - GT 框 = (255, 255, 0) 黄色

用法（conda bev 环境）：
    conda activate bev
    python tools/vis_fastbev_occ_gt.py \
        --pkl data/nuscenes/bevdetv3-nuscenes_infos_val.pkl \
        --save-path work_dirs/vis_gt \
        --frames 0-199

    # 指定 token 可视化单帧
    python tools/vis_fastbev_occ_gt.py \
        --pkl data/nuscenes/bevdetv3-nuscenes_infos_val.pkl \
        --token fd8420396768425eabec9bdddf7e64b6 \
        --save-path work_dirs/vis_gt

    # 输出视频
    python tools/vis_fastbev_occ_gt.py \
        --pkl data/nuscenes/bevdetv3-nuscenes_infos_val.pkl \
        --frames 0-199 --format video --fps 10 \
        --save-path work_dirs/vis_gt

    # Open3D 真 3D：离屏导出固定视角 png + occ 体素 ply（headless 可用）
    python tools/vis_fastbev_occ_gt.py \
        --pkl data/nuscenes/bevdetv3-nuscenes_infos_val.pkl \
        --frames 0-199 --occ-3d-export --occ-3d-lidar --occ-3d-boxes \
        --occ-3d-save-path work_dirs/vis_gt_3d

    # Open3D 真 3D：弹窗交互查看（鼠标旋转/缩放，需 DISPLAY）
    python tools/vis_fastbev_occ_gt.py \
        --pkl data/nuscenes/bevdetv3-nuscenes_infos_val.pkl \
        --token fd8420396768425eabec9bdddf7e64b6 --occ-3d-show
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

OCC_CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

# 18 类语义颜色（与 FlashOCC 一致），free 单独处理为深灰
COLOR_MAP_RAW = np.array([
    [0, 0, 0],          # 0  others
    [255, 120, 50],     # 1  barrier
    [255, 192, 203],    # 2  bicycle
    [255, 255, 0],      # 3  bus
    [0, 150, 245],      # 4  car
    [0, 255, 255],      # 5  construction_vehicle
    [200, 180, 0],      # 6  motorcycle
    [255, 0, 0],        # 7  pedestrian
    [255, 240, 150],    # 8  traffic_cone
    [135, 60, 0],       # 9  trailer
    [160, 32, 240],     # 10 truck
    [255, 0, 255],      # 11 driveable_surface
    [175, 0, 75],       # 12 other_flat
    [75, 0, 75],        # 13 sidewalk
    [150, 240, 80],     # 14 terrain
    [230, 230, 250],    # 15 manmade
    [0, 175, 0],        # 16 vegetation
    [60, 60, 60],       # 17 free（深灰底）
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

FREE_ID = 17
OCC_SIZE = 200
OCC_Z = 16
OCC_RANGE = 40.0  # occ 栅格覆盖 [-40, 40]m

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


def occ_to_bev_rgb(occ, height_mode='none'):
    """occ (200,200,16) [y,-x,z] -> BEV 俯视 RGB 图 (200,200,3)。
    occ 标注维度排列为 (Y, -X, Z), 即 dim0=Y(左向), dim1=-X(后向), dim2=Z(高度)。
    dim1 的 X 轴方向与 lidar 相反（正向=后向而非前向）。

    height_mode:
      - 'none':   对每个 (y,x) 取最低非 free 体素层（贴近地面），无有效体素显示为黑。
                  这样可与 lidar 地面点更好对齐，避免高层 terrain/manmade 遮挡地面结构。
      - 'height': 保持地面语义颜色，但按该列最高 occupied 体素高度调整亮度，
                  z 越高越亮，从而在 BEV 俯视图中体现立体高度信息。
    """
    valid = (occ != FREE_ID).astype(np.int32)          # (200,200,16)
    # 最低非 free 层（用于语义颜色）
    z_bottom = np.argmax(valid, axis=2)                # (200,200)
    # 最高非 free 层（用于高度编码）
    z_top = (OCC_Z - 1) - np.argmax(valid[..., ::-1], axis=2)
    no_occ = valid.sum(axis=2) == 0
    z_top = np.where(no_occ, 0, z_top)

    sem_bev = occ[np.arange(OCC_SIZE)[:, None], np.arange(OCC_SIZE)[None, :],
                  z_bottom]                            # (200,200)
    rgb = COLOR_MAP[sem_bev].astype(np.float32)        # (200,200,3)

    if height_mode == 'height':
        # z 越高越亮，factor 范围 [0.35, 1.0]
        factor = 0.35 + 0.65 * (z_top.astype(np.float32) / (OCC_Z - 1))
        rgb = rgb * factor[..., None]
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        rgb = rgb.astype(np.uint8)
    return rgb


def occ_to_extrusion(occ, canva_size, show_range, z_scale=1.2, cell_pad=0.9):
    """把 occ (200,200,16) 渲染为 BEV 上的挤压柱画布 (N,N,3)，伪 3D 立体。

    投影约定：保持 BEV 平面 (x,y) 不变，高度 z 沿画布向上（row 减小）延伸，
    顶面向右上偏移形成菱形面（观察者位于 BEV 下方/近处）。
    每个非空 (y,x) 栅格按最高 occupied 高度竖一根柱：
      顶面 = 语义原色(最亮)，右侧面 = 语义色*0.75，上侧面(后侧) = 语义色*0.55。
    绘制顺序：按柱顶深度分桶（远→近），桶内先上侧面 -> 右侧面 -> 顶面。
    """
    N = int(canva_size)
    canvas = np.zeros((N, N, 3), dtype=np.uint8)
    occ_h = int(canva_size * OCC_RANGE / show_range)
    cell = occ_h / float(OCC_SIZE)
    margin = int((canva_size - occ_h) / 2)
    half = cell * cell_pad / 2.0

    valid = (occ != FREE_ID).astype(np.int32)          # (200,200,16)
    z_bottom = np.argmax(valid, axis=2)                # 最低非 free 层(语义)
    z_top = (OCC_Z - 1) - np.argmax(valid[..., ::-1], axis=2)
    no_occ = valid.sum(axis=2) == 0
    ys, xs = np.nonzero(~no_occ)
    if ys.size == 0:
        return canvas
    sem = occ[ys, xs, z_bottom[ys, xs]]                # (M,) 语义类别

    # 格子中心在画布上的像素坐标（与 occ_rgb[::-1,::-1] 翻转一致）
    rows = margin + (OCC_SIZE - 1 - ys) * cell + half
    cols = margin + (OCC_SIZE - 1 - xs) * cell + half
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


def occ_to_isometric(occ, canva_size, show_range, z_scale=1.2, cell_pad=0.9):
    """把 occ (200,200,16) 渲染为等距斜视画布 (N,N,3)，伪 3D 立体。

    等距投影（2:1 菱形，30° 轴）：
        u = (x - y) * cos30            屏幕水平
        v = (x + y) * sin30 - hpx      屏幕垂直（向下为正，高度 z 使 v 减小）
    观察者位于南侧俯视，x+（前/右下斜）与 y+（左/左下斜）两个斜面可见：
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

    valid = (occ != FREE_ID).astype(np.int32)          # (200,200,16)
    z_bottom = np.argmax(valid, axis=2)                # 最低非 free 层(语义)
    z_top = (OCC_Z - 1) - np.argmax(valid[..., ::-1], axis=2)
    no_occ = valid.sum(axis=2) == 0
    ys, xs = np.nonzero(~no_occ)
    if ys.size == 0:
        return canvas, None
    sem = occ[ys, xs, z_bottom[ys, xs]]                # (M,) 语义类别

    # 格子中心在画布上的像素坐标（与 occ_rgb[::-1,::-1] 翻转一致）
    rows = margin + (OCC_SIZE - 1 - ys) * cell + half
    cols = margin + (OCC_SIZE - 1 - xs) * cell + half
    hpx = (z_top[ys, xs].astype(np.float32) + 1) * cell * z_scale

    cos30 = np.cos(np.pi / 6.0)
    sin30 = 0.5
    x0, x1 = cols - half, cols + half
    y0, y1 = rows - half, rows + half

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


def occ_to_3d_meshes(occ, voxel_size=0.38):
    """把 occ (200,200,16) [Y,-X,Z] 转换为语义着色的立方体网格列表。

    世界坐标采用 FLU lidar 系（x 前、y 左、z 上），与 2D BEV 画布旋转一致。
    z 映射按 occ3d-nuscenes 官方 z∈[-1,5.4]m：层 k 中心 = k*res + res/2 - 1.0
    （GT 层 0-1 几乎全 free=地下，地面类位于层 2-4=-0.2~1.0m，验证一致）：
        x = (OCC_SIZE-1 - gx)*res + res/2 - OCC_RANGE   (gx: dim1=-X 索引)
        y = gy*res + res/2 - OCC_RANGE                  (gy: dim0=Y 索引)
        z = gz*res + res/2 - 1.0                         (gz: dim2=Z 索引)
    每个非 free 体素生成一个立方体，按语义类别分组为独立 TriangleMesh
    （每类 paint_uniform_color），便于渲染与 ply 导出。
    """
    res = 2 * OCC_RANGE / OCC_SIZE
    s = voxel_size / 2.0

    nz = np.nonzero(occ != FREE_ID)
    if nz[0].size == 0:
        return []
    gy, gx, gz = nz
    centers = np.stack([
        (OCC_SIZE - 1 - gx) * res + res / 2 - OCC_RANGE,
        gy * res + res / 2 - OCC_RANGE,
        gz * res + res / 2 - 1.0,
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
    """读取 lidar 原始点(FLU, x前 y左 z上)为按高度着色的 PointCloud。"""
    pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
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
        [g for _, g in geoms], window_name='FastBEV GT occ 3D',
        width=1600, height=1000,
        front=list(front), lookat=list(lookat), up=list(up), zoom=zoom)


def export_3d_png_ply(geoms, out_dir, prefix, full_img=None, occ=None,
                      lidar_path=None, corners_lidar=None, draw_lidar=True,
                      canva_size=1000, show_range=50, scale_factor=4):
    """导出拼接 png（6 相机 + BEV 俯视）+ occ 体素 ply。

    png 复用主流程的完整拼接图 full_img（上 3 相机 + 下 3 相机），
    中间的 BEV 区域用 --occ-height-mode height 方式重绘后贴回，
    整幅图与 vis_fastbev_occ_gt.py --occ-height-mode height 布局一致；
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
            occ_height_mode='height')
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
        imgs.append(img)
    return imgs


def draw_bev_canvas(canva_size, show_range, occ=None, corners_lidar=None,
                    pred_flag=None, scores=None, vis_thred=0.3,
                    lidar_path=None, draw_lidar=True,
                    occ_height_mode='none', occ_z_scale=1.2):
    """绘制 BEV 画布：occ 底图 + lidar 点 + 检测框。
    画布坐标约定 (FLU lidar 坐标系，画布旋转 90° 展示)：
      col = (x + show_range) / (2*show_range) * canvas   (x前→col右)
      row = (-y + show_range) / (2*show_range) * canvas   (y左→row上)
    """
    canvas = np.zeros((int(canva_size), int(canva_size), 3), dtype=np.uint8)

    # 1) occ 底图
    #    occ 标注维度排列为 (Y, -X, Z), 即 dim0=Y(左向), dim1=-X(后向)
    #    画布: col=x(前→右), row=-y(左→上)
    #    dim0=Y: 翻转使 y正(左)→row0(上)
    #    dim1=-X: 翻转使 x正(前,即dim1小值)→col大(右)
    proj_fn = None
    if occ is not None:
        if occ_height_mode == 'column':
            # 挤压柱伪 3D：每个非空格子按高度竖柱
            canvas = occ_to_extrusion(occ, canva_size, show_range,
                                      z_scale=occ_z_scale)
        elif occ_height_mode == 'isometric':
            # 等距斜视：整幅场景以 30° 菱形投影呈现，含高度柱体
            canvas, proj_fn = occ_to_isometric(occ, canva_size, show_range,
                                               z_scale=occ_z_scale)
        else:
            occ_rgb = occ_to_bev_rgb(occ, height_mode=occ_height_mode)
            occ_h = int(canva_size * OCC_RANGE / show_range)
            occ_bev = cv2.resize(
                np.ascontiguousarray(occ_rgb[::-1, ::-1, :]),
                (occ_h, occ_h),
                interpolation=cv2.INTER_NEAREST)
            margin = int((canva_size - occ_h) / 2)
            roi = canvas[margin:margin + occ_h, margin:margin + occ_h, :]
            # free 区域(深灰)改为透明，只保留有语义的区域
            mask = (occ_bev.sum(axis=2) > 60 * 3).astype(np.uint8) * 255
            mask = cv2.resize(mask, (occ_h, occ_h), interpolation=cv2.INTER_NEAREST)
            occ_bev_masked = cv2.bitwise_and(occ_bev, occ_bev, mask=mask)
            roi[:] = cv2.addWeighted(roi, 1.0, occ_bev_masked, 0.5, 0)

        # 绿色 F 箭头：俯视朝上=前向；等距斜视下前向映射到右下方 30° 轴
        cx = int(canva_size / 2)
        cy = cx
        if proj_fn is not None:
            fx, fy = proj_fn(float(cx), float(cy))
            fx2, fy2 = proj_fn(float(cx + 60), float(cy))
            dx, dy = fx2 - fx, fy2 - fy
            cv2.arrowedLine(canvas, (int(fx), int(fy)),
                            (int(fx + dx), int(fy + dy)), (0, 255, 0), 2,
                            tipLength=0.3)
            cv2.putText(canvas, 'F', (int(fx + dx + 5), int(fy + dy + 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.arrowedLine(canvas, (cx, cy), (cx, cy - 60), (0, 255, 0), 2, tipLength=0.3)
            cv2.putText(canvas, 'F', (cx + 5, cy - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 2) lidar 点云
    if draw_lidar and lidar_path and os.path.exists(lidar_path):
        lidar_points = np.fromfile(lidar_path, dtype=np.float32)
        lidar_points = lidar_points.reshape(-1, 5)[:, :3]
        lidar_points[:, 1] = -lidar_points[:, 1]
        lidar_points[:, :2] = \
            (lidar_points[:, :2] + show_range) / (2 * show_range) * canva_size
        for p in lidar_points:
            if check_point_in_img(
                    p.reshape(1, 3), canvas.shape[1], canvas.shape[0])[0]:
                color = depth2color(p[2])
                px, py = float(p[0]), float(p[1])
                if proj_fn is not None:
                    px, py = proj_fn(px, py)
                cv2.circle(
                    canvas, (int(px), int(py)),
                    radius=0,
                    color=color,
                    thickness=1)

    # 3) 3D 检测框
    if corners_lidar is not None and corners_lidar.shape[0] > 0:
        corners_lidar = corners_lidar.reshape(-1, 8, 3)
        corners_lidar[:, :, 1] = -corners_lidar[:, :, 1]
        bottom_corners_bev = corners_lidar[:, [0, 3, 7, 4], :2]
        bottom_corners_bev = \
            (bottom_corners_bev + show_range) / (2 * show_range) * canva_size
        center_bev = corners_lidar[:, [0, 3, 7, 4], :2].mean(axis=1)
        head_bev = corners_lidar[:, [0, 4], :2].mean(axis=1)
        canter_canvas = \
            (center_bev + show_range) / (2 * show_range) * canva_size
        head_canvas = (head_bev + show_range) / (2 * show_range) * canva_size
        if proj_fn is not None:
            uu, vv = proj_fn(bottom_corners_bev[..., 0].reshape(-1),
                             bottom_corners_bev[..., 1].reshape(-1))
            bottom_corners_bev = np.stack([uu, vv],
                                          axis=1).reshape(-1, 4, 2)
            uu, vv = proj_fn(canter_canvas[:, 0], canter_canvas[:, 1])
            canter_canvas = np.stack([uu, vv], axis=1)
            uu, vv = proj_fn(head_canvas[:, 0], head_canvas[:, 1])
            head_canvas = np.stack([uu, vv], axis=1)
        bottom_corners_bev = np.round(bottom_corners_bev).astype(np.int32)
        center_canvas = canter_canvas.astype(np.int32)
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
        description='Visualize FastBEV OCC GT: lidar + GT boxes + 6 cameras + GT occ')
    parser.add_argument('--pkl', type=str, required=True,
                        help='Path to val infos pkl '
                             '(data/nuscenes/bevdetv3-nuscenes_infos_val.pkl)')
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
                             'vis_fastbev_occ_gt.py --occ-height-mode height '
                             '布局一致）+ occ 体素 ply')
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
    parser.add_argument('--occ-3d-voxel-size', type=float, default=0.4,
                        help='体素显示边长（米），默认 0.4 满格')
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
        # bevdetv3-nuscenes_infos_val.pkl 的 gt_boxes 已采用 mmdet3d
        # LiDAR 约定 (x,y,z,w,l,h,yaw)，直接构造成 LiDARInstance3DBoxes 即可。
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
        occ = None
        if not args.no_occ:
            occ_path = infos.get('occ_path')
            if occ_path and os.path.exists(os.path.join(occ_path, 'labels.npz')):
                occ_gt = np.load(os.path.join(occ_path, 'labels.npz'))
                occ = occ_gt['semantics']  # (200, 200, 16) [Y, X, Z]

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
            occ_z_scale=args.occ_z_scale)

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
                    occ, voxel_size=args.occ_3d_voxel_size)
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
                        scale_factor=scale_factor)

    if vout is not None:
        vout.release()
    print('Done.')


if __name__ == '__main__':
    main()
