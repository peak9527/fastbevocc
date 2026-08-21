#!/usr/bin/env python3
"""
可视化 FastBEVOCC 推理结果（results_occ.pkl）——new 版本。
以 vis_fastbev_occ_results_3d.py 为蓝本，核心改进：3D 导出 png 的 BEV
区域采用【正俯视投影】取色（color_mode='top'，即每格 = 该列最高非 free
层的语义类别），与 ply 体素集合的正俯视投影严格一致。

布局（与 vis.py 一致）：
  - 上方：FRONT_LEFT, FRONT, FRONT_RIGHT 三相机拼接
  - 中间：BEV 俯视图（occ + lidar 点云 + 3D 框）
  - 下方：BACK_LEFT, BACK, BACK_RIGHT 三相机拼接（水平镜像，与 vis.py 一致）

颜色：
  - 预测框 = (0, 255, 255) 青色
  - GT 框   = (255, 255, 0) 黄色

为什么默认用 'top' 取色（new 版核心差异）：
  ply 是真 3D occ 体素网格（每个非 free 体素都存在）；png 是 2D BEV
  俯视，每格只能显示一个代表类别。要使 png 的 occ 区域 = ply 体素的
  俯视投影，唯一正确的取色是显示该列【最高非 free 层】（top）：
    - 竖直绿植墙：最高层是 vegetation -> png 显示绿植色，与 ply 中
      该列体素（绿色）完全对应，不再被下层 driveable_surface 截断；
    - GT 数据列内类别单一，top 与 bottom 结果相同，与 gt_3d 观感一致；
    - 高层误检层（“天棚”）由 --occ-3d-max-z 与 ply 一并滤除。
  配合“png 与 ply 使用同一份截断 occ + 语义 mask（该列是否有非 free
  体素）”，png 中某格有颜色 <-> 该列有体素 <-> ply 中该列存在体素，
  三者严格一一对应，绿植等轮廓不再截断。

用法（conda bev 环境）：
    python tools/vis_fastbev_occ_results_3d_new.py \
        work_dirs/fastbev-occ-r50-cbgs/results_occ_train.pkl \
        --pkl data/nuscenes/bevdetv3-nuscenes_infos_val.pkl \
        --frames 0-199 --occ-3d-export \
        --occ-3d-save-path work_dirs/fastbev-occ-r50-cbgs/vis_3d_train \
        --occ-3d-lidar --occ-3d-boxes --draw-gt --occ-3d-max-z 3.0

    # 需要同时保存主流程 vis_full 时，加 --save-path 即可
    python tools/vis_fastbev_occ_results_3d_new.py \
        work_dirs/fastbev-occ-r50-cbgs/results_occ_train.pkl \
        --pkl data/nuscenes/bevdetv3-nuscenes_infos_val.pkl \
        --save-path work_dirs/fastbev-occ-r50-cbgs/vis_full_train \
        --frames 0-199 --occ-3d-export \
        --occ-3d-save-path work_dirs/fastbev-occ-r50-cbgs/vis_3d_train \
        --occ-3d-lidar --occ-3d-boxes --draw-gt
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


# 渲染用的提亮版：各类颜色更亮，适配深色背景（与 vis_fastbev_occ_gt_3d.py 一致）
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


def get_lidar2global(infos):
    lidar2ego = np.eye(4, dtype=np.float32)
    lidar2ego[:3, :3] = Quaternion(infos['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = infos['lidar2ego_translation']
    ego2global = np.eye(4, dtype=np.float32)
    ego2global[:3, :3] = Quaternion(
        infos['ego2global_rotation']).rotation_matrix
    ego2global[:3, 3] = infos['ego2global_translation']
    return ego2global @ lidar2ego


def occ_to_bev_rgb(occ, height_mode='none', color_mode='top'):
    """occ (200,200,16) [y,-x,z] -> BEV 俯视 RGB 图 (200,200,3)。
    occ 标注维度排列为 (Y, -X, Z), 即 dim0=Y(左向), dim1=-X(后向), dim2=Z(高度)。
    dim1 的 X 轴方向与 lidar 相反（正向=后向而非前向）。

    height_mode:
      - 'none':   对每个 (y,x) 取对应层语义色（不调亮度）。
      - 'height': 保持语义颜色，但按该列最高 occupied 体素高度调整亮度，
                  z 越高越亮，从而在 BEV 俯视图中体现立体高度信息。

    color_mode（每列语义颜色取哪一层）：
      - 'top':    最高非 free 层（new 版默认）。这是 ply 体素集合的
                  正俯视投影：高层遮挡低层，每格显示该列最高层类别。
                  竖直绿植墙最高层即 vegetation，png 与 ply 中该列
                  体素完全对应，轮廓不再被下层地面截断。
      - 'bottom': 最低非 free 层（gt_3d 兼容；GT 列内单一与 top 相同，
                  但 pred 列内混合时会被地面层截断）。
      - 'majority': 该列出现次数最多的类别。
    """
    valid = (occ != FREE_ID).astype(np.int32)          # (200,200,16)
    # 最低非 free 层
    z_bottom = np.argmax(valid, axis=2)                # (200,200)
    # 最高非 free 层（用于高度编码与 color_mode='top'）
    z_top = (OCC_Z - 1) - np.argmax(valid[..., ::-1], axis=2)
    no_occ = valid.sum(axis=2) == 0
    z_top = np.where(no_occ, 0, z_top)

    if color_mode == 'top':
        sem_bev = occ[np.arange(OCC_SIZE)[:, None],
                      np.arange(OCC_SIZE)[None, :], z_top]
    elif color_mode == 'bottom':
        sem_bev = occ[np.arange(OCC_SIZE)[:, None],
                      np.arange(OCC_SIZE)[None, :], z_bottom]
    elif color_mode == 'majority':
        n_cls = COLOR_MAP.shape[0]
        counts = np.stack([(occ == c).sum(axis=2)
                           for c in range(n_cls)], axis=2)  # (200,200,n_cls)
        counts[..., FREE_ID] = -1  # 排除 free，使多数类别只在非 free 中选
        sem_bev = counts.argmax(axis=2)
    else:
        sem_bev = occ[np.arange(OCC_SIZE)[:, None],
                      np.arange(OCC_SIZE)[None, :], z_bottom]
    # 无有效体素的列统一显示 free 色（黑，绘制时由语义 mask 置透明）
    sem_bev = np.where(no_occ, FREE_ID, sem_bev)       # (200,200)
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
    sem = occ[ys, xs, z_top[ys, xs]]                   # 顶部层语义（俯视投影）

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
    sem = occ[ys, xs, z_top[ys, xs]]                   # 顶部层语义（俯视投影）

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


# ------------------------- Open3D 真 3D -------------------------

def _parse_vec3(s, default):
    """解析 'x,y,z' 字符串为元组，失败返回默认值。"""
    try:
        parts = [float(x.strip()) for x in s.split(',')]
        return (parts[0], parts[1], parts[2])
    except Exception:
        return default


def _depth_to_colors(depths):
    """向量化 depth2color（与 vis_fastbev_occ_gt_3d.py 一致）：
    depth -> RGB(float32, 0-255)。"""
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


# occ3d-nuscenes 官方 z 范围：[-1.0, 5.4]m（16 层，0.4m/层）。
# 层 k 中心 = OCC_Z_MIN*(-1) + (k+0.5)*res = k*res + res/2 - 1.0。
OCC_Z_MIN = 1.0


def _truncate_occ_z(occ, max_z):
    """把层中心 z > max_z 的层全部置为 FREE_ID，返回副本。

    png 与 ply 共用同一截断 occ（main 中统一 _truncate_occ_z 后传入），
    保证 png 的 BEV 体素集与 ply 严格一致；max_z 为 None 时原样返回。
    """
    if occ is None or max_z is None:
        return occ
    res = 2 * OCC_RANGE / OCC_SIZE
    # 层中心 z = k*res + res/2 - 1.0，只保留中心不超过 max_z 的层
    keep = np.arange(OCC_Z) <= (max_z + OCC_Z_MIN - res / 2) / res
    occ2 = occ.copy()
    occ2[..., ~keep] = FREE_ID
    return occ2


def occ_to_3d_meshes(occ, voxel_size=0.4, max_z=6.4):
    """occ (200,200,16) [Y,-X,Z] -> 语义着色的立方体网格列表
    （网格构造严格对齐 vis_fastbev_occ_gt_3d.py，z 映射修正为
    occ3d-nuscenes 官方 z∈[-1,5.4]：层 k 中心 = k*res + res/2 - 1.0）。

    世界坐标采用 FLU lidar 系（x 前、y 左、z 上），与 2D BEV 画布旋转一致：
        x = (OCC_SIZE-1 - gx)*res + res/2 - OCC_RANGE   (gx: dim1=-X 索引)
        y = gy*res + res/2 - OCC_RANGE                  (gy: dim0=Y 索引)
        z = gz*res + res/2 - OCC_Z_MIN                  (gz: dim2=Z 索引)
    只渲染非 free 体素（occ != FREE_ID，排除 17=free）；max_z 可过滤高空
    体素（默认 6.4 全保留，与 gt_3d 行为一致）。按语义类别分组为独立
    TriangleMesh（每类 paint_uniform_color），便于渲染与 ply 导出。
    """
    if occ is None:
        return []
    # 先按 max_z 截断（与 png BEV 共用的体素集，保证 png/ply 细节一致）
    occ = _truncate_occ_z(occ, max_z)
    res = 2 * OCC_RANGE / OCC_SIZE
    s = voxel_size / 2.0

    nz = np.nonzero(occ != FREE_ID)
    if nz[0].size == 0:
        return []
    gy, gx, gz = nz
    centers = np.stack([
        (OCC_SIZE - 1 - gx) * res + res / 2 - OCC_RANGE,
        gy * res + res / 2 - OCC_RANGE,
        gz * res + res / 2 - OCC_Z_MIN,
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
    """读 npy/bin lidar -> 按高度着色的 PointCloud。"""
    if str(lidar_path).endswith('.npy'):
        pts = np.load(lidar_path)
    else:
        pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
    pts = np.ascontiguousarray(pts, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(
        _depth_to_colors(pts[:, 2]).astype(np.float64) / 255.0)
    return pcd


def boxes_to_3d_lineset(corners_lidar, pred_flag):
    """3D 框 8 角点(FLU) -> 12 边 LineSet，pred 青 / GT 黄。

    与 gt_3d 的黄色框不同：results 场景中 pred(1) 用青色，
    GT(0) 用黄色，与 2D BEV 中 color_map 配色一致。
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


def show_3d_window(geoms, front=(0.5, -0.75, 0.4), up=(0, 0, 1),
                   lookat=(0, 0, 1), zoom=4.0):
    """Open3D 弹窗交互查看（需 DISPLAY，无显示环境自动跳过）。"""
    try:
        if not os.environ.get('DISPLAY'):
            print('[3D] 无 DISPLAY，跳过弹窗（可用 --occ-3d-export 离屏导出）')
            return
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='FastBEV occ results 3D (new)',
                          width=1280, height=900)
        for name, g in geoms:
            vis.add_geometry(g)
        ctr = vis.get_view_control()
        ctr.set_front(front)
        ctr.set_up(up)
        ctr.set_lookat(lookat)
        ctr.set_zoom(zoom)
        opt = vis.get_render_option()
        opt.background_color = np.array([0.05, 0.05, 0.08])
        opt.point_size = 2.0
        opt.line_width = 3.0
        vis.run()
        vis.destroy_window()
    except Exception as e:
        print(f'[3D] 弹窗失败: {e}')


def export_3d_png_ply(geoms, out_dir, prefix, full_img=None, occ=None,
                      occ_gt=None, lidar_path=None, corners_lidar=None,
                      pred_flag=None, scores=None, vis_thred=0.3,
                      draw_lidar=True, canva_size=1000, show_range=50,
                      scale_factor=4, occ_color_mode='top'):
    """导出拼接 png（6 相机 + BEV 俯视）+ occ 体素 ply。

    png 复用主流程的完整拼接图 full_img（上 3 相机 + 下 3 相机），
    中间的 BEV 区域用 occ_height_mode='height' + occ_color_mode='top'
    （正俯视投影）重绘后贴回：
      - height 亮度：z 越高越亮，保留 gt_3d 的立体明暗观感；
      - top 取色：每格 = 该列最高非 free 层，即 ply 体素集合的正俯视
        投影，高层遮挡低层，竖直绿植墙最高层即 vegetation，png 与
        ply 中对应体素严格一致，轮廓不再被下层地面截断；
      - 语义 mask（该列是否有非 free 体素）+ png/ply 共用截断 occ
        （main 中 _truncate_occ_z），保证 png 有颜色的列集合与 ply
        体素列集合一一对应。
    occ_gt 非 None 时，BEV 区域左右分块：左=GT occ + GT 框，右=Pred
    occ + 全部框，其余显示不变（与主流程 vis_full 一致）。
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
                occ_height_mode='height', occ_color_mode=occ_color_mode)
        else:
            canvas = draw_bev_canvas(
                canva_size, show_range, occ=occ,
                corners_lidar=corners_lidar,
                pred_flag=pred_flag, scores=scores,
                vis_thred=vis_thred,
                lidar_path=lidar_path, draw_lidar=draw_lidar,
                occ_height_mode='height', occ_color_mode=occ_color_mode)
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
        print(f'[3D] saved {png_path} (6cam + BEV height+top'
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
                    occ_height_mode='none', occ_z_scale=1.2,
                    occ_color_mode='top'):
    """绘制 BEV 画布：occ 底图 + lidar 点 + 检测框。坐标系与 vis.py 完全一致。"""
    canvas = np.zeros((int(canva_size), int(canva_size), 3), dtype=np.uint8)
    proj_fn = None

    # 1) occ 占用底图（只放在 [-OCC_RANGE, OCC_RANGE] 对应区域，不拉伸到全 canvas）
    #    occ 标注维度排列为 (Y, -X, Z), 即 dim0=Y(左向), dim1=-X(后向)
    #    lidar 画布: col=x(前→右), row=-y(左→上)
    #    dim0=Y: 翻转使 y正(左)→row0(上)
    #    dim1=-X: 翻转使 x正(前,即dim1小值)→col大(右)
    if occ is not None:
        if occ_height_mode == 'column':
            # 挤压柱伪 3D：每个非空格子按高度竖柱
            canvas = occ_to_extrusion(occ, canva_size, show_range,
                                      z_scale=occ_z_scale)
        elif occ_height_mode == 'isometric':
            # 等距斜视伪 3D：返回投影函数供 lidar/框叠加
            canvas, proj_fn = occ_to_isometric(
                occ, canva_size, show_range, z_scale=occ_z_scale)
        else:
            occ_rgb = occ_to_bev_rgb(
                occ, height_mode=occ_height_mode, color_mode=occ_color_mode)
            occ_h = int(canva_size * OCC_RANGE / show_range)
            occ_bev = cv2.resize(
                np.ascontiguousarray(occ_rgb[::-1, ::-1, :]),
                (occ_h, occ_h),
                interpolation=cv2.INTER_NEAREST)
            margin = int((canva_size - occ_h) / 2)
            roi = canvas[margin:margin + occ_h, margin:margin + occ_h, :]
            # free 列改为透明，只保留有体素的列。
            # mask 与 ply 一样基于“该列是否有非 free 体素”（occ != FREE_ID），
            # 而不是 RGB 亮度阈值：height 模式会把低矮绿植等暗色类别大幅调暗
            # （vegetation 提亮后约 (64,195,64)，factor<0.57 时 sum 会低于
            # 180），若沿用 sum>180 的阈值，这些列会被误当 free 滤除，
            # 导致 png 中绿植轮廓“截断”而 ply 中对应体素仍连通。
            mask = (occ != FREE_ID).any(axis=2).astype(np.uint8) * 255
            mask = cv2.resize(mask[::-1, ::-1], (occ_h, occ_h),
                              interpolation=cv2.INTER_NEAREST)
            occ_bev_masked = cv2.bitwise_and(occ_bev, occ_bev, mask=mask)
            roi[:] = cv2.addWeighted(roi, 1.0, occ_bev_masked, 0.5, 0)

        # 绿色 F 箭头：朝上 = 前向
        cx = int(canva_size / 2)
        cy = cx
        if proj_fn is not None:
            fx, fy = proj_fn(np.array([cx]), np.array([cy]))
            cx, cy = int(fx[0]), int(fy[0])
        cv2.arrowedLine(canvas, (cx, cy), (cx, cy - 60), (0, 255, 0), 2, tipLength=0.3)
        cv2.putText(canvas, 'F', (cx + 5, cy - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 2) lidar 点云（与 vis.py 完全一致）
    if draw_lidar and lidar_path and os.path.exists(lidar_path):
        lidar_points = np.fromfile(lidar_path, dtype=np.float32)
        lidar_points = lidar_points.reshape(-1, 5)[:, :3]
        lidar_points[:, 1] = -lidar_points[:, 1]
        lx = (lidar_points[:, 0] + show_range) / (2 * show_range) * canva_size
        ly = (lidar_points[:, 1] + show_range) / (2 * show_range) * canva_size
        if proj_fn is not None:
            px, py = proj_fn(lx, ly)
            lidar_points[:, 0] = px
            lidar_points[:, 1] = py
        else:
            lidar_points[:, 0] = lx
            lidar_points[:, 1] = ly
        for p in lidar_points:
            if check_point_in_img(
                    p.reshape(1, 3), canvas.shape[1], canvas.shape[0])[0]:
                color = depth2color(p[2])
                cv2.circle(
                    canvas, (int(p[0]), int(p[1])),
                    radius=0,
                    color=color,
                    thickness=1)

    # 3) 3D 检测框（与 vis.py 完全一致）
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


def draw_bev_split_canvas(canva_size, show_range, occ_left=None,
                          occ_right=None, corners_gt=None, corners_all=None,
                          pred_flag=None, scores=None, vis_thred=0.3,
                          lidar_path=None, draw_lidar=True,
                          occ_height_mode='none', occ_z_scale=1.2,
                          occ_color_mode='top', label_left='GT',
                          label_right='Pred'):
    """把 BEV 区域分成左右两块画布：左=occ 真值(GT)，右=occ 预测(Pred)。

    两块均为 half×half 方形画布（half = canva_size//2），并排成
    (half, canva_size) 横条，由调用方垂直居中贴回原 BEV 区域：
      - 左块：GT occ 底图 + GT 框（黄）+ lidar；
      - 右块：Pred occ 底图 + 全部框（pred 青 / GT 黄）+ lidar。
    相机拼接等其余部分完全不变。内部对 corners 一律取副本，
    不影响外部与 3D 导出的原始坐标。
    """
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
        occ_color_mode=occ_color_mode)
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
        occ_color_mode=occ_color_mode)
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


def parse_frames(spec, total):
    """解析 --frames：'a-b' 或 'a,b,c'（默认 0..19）。"""
    if spec is None:
        return list(range(min(20, total)))
    if '-' in spec:
        a, b = spec.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(',') if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize FastBEVOCC '
                                     'inference results (det + occ) - new '
                                     'version, png BEV = top-view projection '
                                     'of ply voxels')
    parser.add_argument('res', help='Path to results_occ.pkl (test.py --out)')
    parser.add_argument('--pkl', type=str, required=True,
                        help='Path to val infos pkl '
                             '(data/nuscenes/bevdetv3-nuscenes_infos_val.pkl)')
    parser.add_argument('--save-path', type=str, default=None,
                        help='保存主流程 6 相机+BEV 拼接图的目录；不传则不'
                             '输出 vis_full（内部仍会生成拼接图作为 3D 导出'
                             'png 的底图，与 vis_fastbev_occ_gt_3d.py 一致）')
    parser.add_argument('--frames', type=str, default=None,
                        help="Frame indices to visualize, e.g. '0-20' or "
                             "'0,5,10' (default: first 20)")
    parser.add_argument('--token', type=str, default=None,
                        help='Visualize the frame with this token')
    parser.add_argument('--show-range', type=int, default=50,
                        help='BEV range in meters (default 50, vis.py 一致)')
    parser.add_argument('--canva-size', type=int, default=1000)
    parser.add_argument('--scale-factor', type=int, default=4)
    parser.add_argument('--vis-thred', type=float, default=0.1)
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
        '--occ-bev-color-mode', type=str, default='top',
        choices=['top', 'bottom', 'majority'],
        help="png BEV 每格语义色取哪一层（对 vis_full 与 3D 导出均生效）："
             "'top'=最高非 free 层（默认，正俯视投影，即 ply 体素集合的"
             "俯视投影，png 与 ply 严格对应，竖直绿植不被下层地面截断）；"
             "'bottom'=最低非 free 层（gt_3d 兼容，GT 列内单一与 top 相同，"
             "但 pred 列内混合时绿植会被地面截断）；'majority'=该列体素"
             "最多的类别")
    parser.add_argument('--no-lidar', action='store_true')

    # ---- Open3D 真 3D（参考 vis_fastbev_occ_gt_3d.py）----
    parser.add_argument('--occ-3d-show', action='store_true',
                        help='Open3D 弹窗交互查看真 3D（鼠标旋转/缩放；'
                             '需 DISPLAY，无显示环境自动跳过）')
    parser.add_argument('--occ-3d-export', action='store_true',
                        help='离屏导出：6 相机 + BEV 俯视 png + occ 体素 ply；'
                             'png 的 BEV 固定 occ_height_mode=height + '
                             'occ_color_mode=top（正俯视投影，与 ply 体素'
                             '集合严格对应），绿植等轮廓不再截断')
    parser.add_argument('--occ-3d-save-path', type=str,
                        default='work_dirs/fastbev-occ-r50-cbgs/vis_3d_train_new')
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
    parser.add_argument('--occ-3d-max-z', type=float, default=6.4,
                        help='滤除层中心 z > max_z 的体素（米），对 png 与 '
                             'ply 同时生效：预测 occ 高空常被误检为 '
                             'manmade/vegetation 形成“天棚”，在 png 的 '
                             'height 亮度下地面会呈“双层”发亮，设如 3.0 '
                             '即可从 png/ply 中一并去除，保证两者完全对应；'
                             '默认 6.4 全保留（与 gt_3d 行为一致）')
    parser.add_argument('--occ-3d-lidar', action='store_true',
                        help='在 3D 场景中叠加 lidar 点云（按高度着色）')
    parser.add_argument('--occ-3d-boxes', action='store_true',
                        help='在 3D 场景中叠加 3D 检测框（pred 青 / GT 黄）')
    parser.add_argument('--format', type=str, default='image',
                        choices=['image', 'video'])
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--video-prefix', type=str, default='vis')
    return parser.parse_args()


def main():
    args = parse_args()

    results = pickle.load(open(args.res, 'rb'))
    dataset = pickle.load(open(args.pkl, 'rb'))
    # 关键：results 的顺序与 dataset 内部 data_infos 一致，而 data_infos 在
    # mmdet3d/datasets/nuscenes_dataset.py::load_annotations 中按 timestamp 升序排序
    # （data_infos = sorted(data['infos'], key=lambda e: e['timestamp'])），
    # 与 ann_file 里 infos 的原始顺序不同。必须同样排序后 index 才能一一对应，
    # 否则预测框与 GT/点云/相机图错位（表现为框与真值不对齐）。
    infos_list = sorted(dataset['infos'], key=lambda e: e['timestamp'])
    print(f'loaded {len(results)} results, {len(infos_list)} infos '
          f'(sorted by timestamp)')

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
            corners_ego = boxes.corners.numpy().reshape(-1, 3)
            lidar2ego = np.eye(4, dtype=np.float32)
            lidar2ego[:3, :3] = Quaternion(
                infos['lidar2ego_rotation']).rotation_matrix
            lidar2ego[:3, 3] = infos['lidar2ego_translation']
            corners_lidar = (np.concatenate(
                [corners_ego,
                 np.ones((corners_ego.shape[0], 1), dtype=np.float32)],
                axis=1) @ np.linalg.inv(lidar2ego).T)[:, :3]
            pred_flag = np.ones(boxes.tensor.shape[0], dtype=bool)

        # ---- GT 框（可选）----
        # 注意：bevdetv3-nuscenes_infos_val.pkl 的 gt_boxes 已采用 mmdet3d
        # LiDAR 约定 (x,y,z,w,l,h,yaw)，直接构造成 LiDARInstance3DBoxes 即可。
        # 不需要 vis.py 中的 +pi/2 与 w/l 交换（那是针对官方 nuScenes pkl 的修正）。
        # 经验证：无修正时 GT 框内 lidar 点数约是修正版的 2 倍，方向正确。
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

        # ---- occ（预测）----
        occ = res.get('occ')

        # ---- occ（GT 真值，用于中间 BEV 左块对比）----
        occ_gt = None
        occ_path = infos.get('occ_path')
        if occ_path and os.path.exists(os.path.join(occ_path, 'labels.npz')):
            occ_gt = np.load(os.path.join(occ_path,
                                          'labels.npz'))['semantics']

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
            occ_color_mode=args.occ_bev_color_mode)

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

        # ---- Open3D 真 3D（参考 vis_fastbev_occ_gt_3d.py）----
        if args.occ_3d_show or args.occ_3d_export:
            # png 与 ply 使用同一截断 occ，保证两者体素集完全一致
            occ_3d = _truncate_occ_z(occ, args.occ_3d_max_z)
            geoms = []
            if occ_3d is not None:
                geoms += occ_to_3d_meshes(
                    occ_3d, voxel_size=args.occ_3d_voxel_size)
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
                        occ_color_mode=args.occ_bev_color_mode)

    if vout is not None:
        vout.release()
    print('Done.')


if __name__ == '__main__':
    main()
