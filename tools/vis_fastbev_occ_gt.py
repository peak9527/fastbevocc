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
"""

import argparse
import os
import pickle

import cv2
import numpy as np
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
COLOR_MAP = np.array([
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

FREE_ID = 17
OCC_SIZE = 200
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


def occ_to_bev_rgb(occ):
    """occ (200,200,16) [y,-x,z] -> BEV 俯视 RGB 图 (200,200,3)。
    occ 标注维度排列为 (Y, -X, Z), 即 dim0=Y(左向), dim1=-X(后向), dim2=Z(高度)。
    dim1 的 X 轴方向与 lidar 相反（正向=后向而非前向）。
    对每个 (y,x) 栅格取最低非 free 体素层（贴近地面），无有效体素显示为黑。
    """
    valid = (occ != FREE_ID).astype(np.int32)          # (200,200,16)
    z_idx = np.argmax(valid, axis=2)                   # (200,200)
    sem_bev = occ[np.arange(OCC_SIZE)[:, None], np.arange(OCC_SIZE)[None, :],
                  z_idx]                               # (200,200)
    rgb = COLOR_MAP[sem_bev]                           # (200,200,3)
    return rgb


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
                    lidar_path=None, draw_lidar=True):
    """绘制 BEV 画布：occ 底图 + lidar 点 + 检测框。
    画布坐标约定 (FLU lidar 坐标系，画布旋转 90° 展示)：
      col = (x + show_range) / (2*show_range) * canvas   (x前→col右)
      row = (-y + show_range) / (2*show_range) * canvas   (y左→row上)
    """
    canvas = np.zeros((int(canva_size), int(canva_size), 3), dtype=np.uint8)

    # 1) occ 占用底图
    #    occ 标注维度排列为 (Y, -X, Z), 即 dim0=Y(左向), dim1=-X(后向)
    #    画布: col=x(前→右), row=-y(左→上)
    #    dim0=Y: 翻转使 y正(左)→row0(上)
    #    dim1=-X: 翻转使 x正(前,即dim1小值)→col大(右)
    if occ is not None:
        occ_rgb = occ_to_bev_rgb(occ)
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

        # 绿色 F 箭头：朝上 = 前向
        cx = int(canva_size / 2)
        cy = cx
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
                cv2.circle(
                    canvas, (int(p[0]), int(p[1])),
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
        bottom_corners_bev = np.round(bottom_corners_bev).astype(np.int32)
        center_bev = corners_lidar[:, [0, 3, 7, 4], :2].mean(axis=1)
        head_bev = corners_lidar[:, [0, 4], :2].mean(axis=1)
        canter_canvas = \
            (center_bev + show_range) / (2 * show_range) * canva_size
        center_canvas = canter_canvas.astype(np.int32)
        head_canvas = (head_bev + show_range) / (2 * show_range) * canva_size
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
    parser.add_argument('--no-lidar', action='store_true')
    parser.add_argument('--no-occ', action='store_true',
                        help='Skip loading/visualizing GT occ')
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
        canvas = draw_bev_canvas(
            canva_size, show_range, occ=occ,
            corners_lidar=corners_lidar, pred_flag=pred_flag,
            scores=scores, vis_thred=args.vis_thred,
            lidar_path=infos.get('lidar_path'),
            draw_lidar=not args.no_lidar)

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

    if vout is not None:
        vout.release()
    print('Done.')


if __name__ == '__main__':
    main()
