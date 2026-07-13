#!/usr/bin/env python3
"""
可视化自定义 pkl 中的点云和图像, 用于验证标定是否正确。

用法:
    python tools/vis_custom_pkl.py --pkl custom_infos_val.pkl --show-range 50
"""

import argparse
import os
import pickle

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize custom pkl data')
    parser.add_argument('--pkl', type=str, required=True, help='Path to pkl file')
    parser.add_argument('--show-range', type=float, default=50,
                        help='BEV display range in meters (default: 50)')
    parser.add_argument('--point-size', type=int, default=1,
                        help='Point size in BEV canvas (default: 1)')
    parser.add_argument('--save-dir', type=str, default='./vis_custom_output',
                        help='Directory to save visualization images')
    return parser.parse_args()


def _load_lidar_points(lidar_path):
    """自适应加载点云, 自动检测每点字段数 (3/4/5/6)"""
    raw = np.fromfile(lidar_path, dtype=np.float32)
    for dim in [4, 5, 6, 3]:  # 优先尝试常见格式
        if raw.size % dim == 0:
            return raw.reshape(-1, dim)
    raise ValueError(f"Cannot reshape {raw.size} floats into any of [3,4,5,6] per point")


def draw_bev(lidar_points, show_range=50, resolution=1000, point_size=1):
    """绘制 BEV 俯视图"""
    canvas_size = int(2 * show_range / (2 * show_range / resolution))
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    canvas[:, :] = (40, 40, 40)  # 深灰背景

    # 坐标系: 图像 x=右, y=下; 世界 x=前(上), y=左(右)
    # lidar_points: Nx3, [x(前), y(左), z(上)]
    xs = lidar_points[:, 0]  # 前向
    ys = lidar_points[:, 1]  # 左向

    # 过滤范围外的点
    mask = (np.abs(xs) < show_range) & (np.abs(ys) < show_range)
    xs, ys = xs[mask], ys[mask]

    if len(xs) == 0:
        return canvas

    # 映射到画布坐标: 世界 x→图像上(行减小), 世界 y→图像右(列增大)
    scale = canvas_size / (2 * show_range)
    px = ((-ys + show_range) * scale).astype(np.int32)  # 世界左→图像右
    py = ((-xs + show_range) * scale).astype(np.int32)  # 世界前→图像上

    # 裁剪到画布内
    valid = (px >= 0) & (px < canvas_size) & (py >= 0) & (py < canvas_size)
    px, py = px[valid], py[valid]

    # 按高度着色: 低→蓝, 高→红
    zs = lidar_points[mask][valid, 2] if valid.sum() > 0 else lidar_points[:0, 2]
    if len(zs) > 0:
        z_min, z_max = zs.min(), zs.max()
        if z_max > z_min:
            z_norm = (zs - z_min) / (z_max - z_min)
        else:
            z_norm = np.zeros_like(zs)
        # 蓝色(低) → 绿色 → 红色(高)
        colors = np.zeros((len(zs), 3), dtype=np.uint8)
        colors[:, 2] = (z_norm * 255).astype(np.uint8)            # R
        colors[:, 0] = ((1 - z_norm) * 255).astype(np.uint8)      # B
        colors[:, 1] = (np.abs(z_norm - 0.5) * 255).astype(np.uint8)  # G
    else:
        colors = np.tile([100, 200, 100], (len(px), 1))

    for i in range(len(px)):
        cv2.circle(canvas, (px[i], py[i]), point_size,
                   colors[i].tolist(), -1)

    # 画坐标轴和刻度
    center = canvas_size // 2
    cv2.line(canvas, (0, center), (canvas_size, center), (80, 80, 80), 1)
    cv2.line(canvas, (center, 0), (center, canvas_size), (80, 80, 80), 1)

    # 画原点 (ego 位置)
    cv2.circle(canvas, (center, center), 5, (0, 255, 255), -1)
    # 画前向箭头
    cv2.arrowedLine(canvas, (center, center), (center, center - 30),
                    (0, 255, 255), 2, tipLength=0.3)

    return canvas


def main():
    args = parse_args()

    # 加载 pkl
    dataset = pickle.load(open(args.pkl, 'rb'))
    print(f"Loaded pkl with {len(dataset['infos'])} frames")
    print(f"Cameras: {dataset['metadata'].get('cam_names', 'unknown')}")

    os.makedirs(args.save_dir, exist_ok=True)

    for idx, info in enumerate(dataset['infos']):
        token = info['token']
        print(f"\n[{idx+1}/{len(dataset['infos'])}] Frame: {token}")

        # 1. 读取并显示点云 BEV
        lidar_path = info['lidar_path']
        print(f"  LiDAR: {lidar_path}")
        if os.path.exists(lidar_path):
            lidar_points = _load_lidar_points(lidar_path)
            print(f"    Points: {len(lidar_points)} ({lidar_points.shape[1]} fields/point), "
                  f"range: x=[{lidar_points[:,0].min():.1f},{lidar_points[:,0].max():.1f}], "
                  f"y=[{lidar_points[:,1].min():.1f},{lidar_points[:,1].max():.1f}]")
            bev_img = draw_bev(lidar_points, show_range=args.show_range,
                               point_size=args.point_size)
        else:
            print(f"    WARNING: LiDAR file not found!")
            bev_img = np.zeros((1000, 1000, 3), dtype=np.uint8)

        # 2. 读取并显示每个相机图像
        cam_imgs = []
        for cam_name, cam_data in info['cams'].items():
            img_path = cam_data['data_path']
            print(f"  {cam_name}: {img_path}")
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    h, w = img.shape[:2]
                    # 缩小显示
                    disp_h = 400
                    disp_w = int(w * disp_h / h)
                    img_disp = cv2.resize(img, (disp_w, disp_h))
                    cv2.putText(img_disp, cam_name, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cam_imgs.append(img_disp)
                    print(f"    Shape: {h}x{w}")
                else:
                    print(f"    WARNING: Failed to read image")
            else:
                print(f"    WARNING: Image not found!")

        # 3. 显示 GT 框信息
        if info['gt_boxes'] is not None and len(info['gt_boxes']) > 0:
            print(f"  GT boxes: {len(info['gt_boxes'])}")
            for box, name in zip(info['gt_boxes'], info['gt_names']):
                print(f"    {name}: x={box[0]:.1f}, y={box[1]:.1f}, z={box[2]:.1f}, "
                      f"w={box[3]:.1f}, l={box[4]:.1f}, h={box[5]:.1f}, yaw={box[6]:.2f}")

        # 4. 打印 BEV 说明信息
        print(f"\n  [BEV] 上=车辆前方, 原点(黄点)=ego位置, 箭头=前向")
        print(f"  [BEV] 颜色: 蓝=低处, 红=高处")

        # 5. 拼接相机图像
        cam_panel = None
        if cam_imgs:
            # 统一高度
            max_h = max(img.shape[0] for img in cam_imgs)
            padded_imgs = []
            for img in cam_imgs:
                pad = max_h - img.shape[0]
                if pad > 0:
                    img = cv2.copyMakeBorder(img, 0, pad, 0, 0,
                                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
                padded_imgs.append(img)
            cam_panel = np.hstack(padded_imgs)

        # 6. 保存到文件 (headless 模式, 不使用 GUI 显示)
        save_bev = os.path.join(args.save_dir, f'{token}_bev.png')
        cv2.imwrite(save_bev, bev_img)
        print(f"  Saved: {save_bev}")

        if cam_panel is not None:
            save_cam = os.path.join(args.save_dir, f'{token}_cams.png')
            cv2.imwrite(save_cam, cam_panel)
            print(f"  Saved: {save_cam}")

    print(f"\nDone. Results saved to {args.save_dir}/")


if __name__ == '__main__':
    main()
