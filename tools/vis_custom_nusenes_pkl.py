#!/usr/bin/env python3
"""
可视化自定义 pkl 中的点云和图像, 用于验证标定是否正确。

用法:
    # 遍历全部帧
    python tools/vis_custom_pkl.py --pkl custom_infos_val.pkl --show-range 50
    # 指定帧索引
    python tools/vis_custom_pkl.py --pkl custom_infos_val.pkl --idx 0
    # 指定 token
    python tools/vis_custom_pkl.py --pkl custom_infos_val.pkl --token xxxxx
    # 输入图片路径, 自动时间戳匹配 (推荐)
    python tools/vis_custom_pkl.py --pkl custom_infos_val.pkl --img /path/to/frontmain/xxx.jpg
"""

import argparse
import bisect
import os
import pickle
import re
import sys

import cv2
import numpy as np
from pyquaternion import Quaternion


CAM_NAMES = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
             'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize custom pkl data')
    parser.add_argument('--pkl', type=str, required=True, help='Path to pkl file')
    parser.add_argument('--show-range', type=float, default=50,
                        help='BEV display range in meters (default: 50)')
    parser.add_argument('--point-size', type=int, default=1,
                        help='Point size in BEV canvas (default: 1)')
    parser.add_argument('--save-dir', type=str, default='./vis_custom_output',
                        help='Directory to save visualization images')
    parser.add_argument('--idx', type=int, default=None,
                        help='可视化指定帧索引 (从 0 开始), 不指定则遍历全部')
    parser.add_argument('--token', type=str, default=None,
                        help='可视化指定 token 帧, 与 --idx / --img 互斥')
    parser.add_argument('--img', type=str, default=None,
                        help='输入单张图片路径, 按时间戳自动匹配 pkl 中的帧')
    parser.add_argument('--cam', type=str, default=None,
                        choices=CAM_NAMES,
                        help='相机名称 (默认从图片路径自动推断)')
    parser.add_argument('--max-diff-ms', type=int, default=200,
                        help='lidar 与图片最大允许时差 (ms, 默认: )')
    return parser.parse_args()


def get_ts_from_filename(file_path):
    """从文件名提取时间戳, 返回微秒级整数.

    文件名格式: n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151061512404.jpg
    第一组 "008" 是场景号, 最后一组 "1533151061512404" 才是微秒时间戳。
    """
    basename = os.path.basename(file_path)
    matches = re.findall(r"\d+", basename)
    if not matches:
        raise ValueError(f"无法从文件名提取时间戳: {basename}")
    return int(matches[-1])


def detect_cam_from_path(img_path):
    """从图片路径中自动推断相机名称, 如 .../frontmain/xxx.jpg → CAM_FRONT"""
    norm = img_path.replace('\\', '/')
    # 支持自定义命名 → nuScenes 标准名映射
    name_hints = {
        'frontmain': 'CAM_FRONT',
        'front_main': 'CAM_FRONT',
        'rear': 'CAM_BACK',
        'leftrepeater': 'CAM_FRONT_LEFT',
        'rightrepeater': 'CAM_FRONT_RIGHT',
        'leftpillar': 'CAM_BACK_LEFT',
        'rightpillar': 'CAM_BACK_RIGHT',
    }
    for hint, cam in name_hints.items():
        if hint in norm.lower():
            return cam
    # nuScenes 标准名
    for cam in CAM_NAMES:
        if cam in norm.upper() or cam in norm:
            return cam
    return None


def find_best_frame_from_pkl(infos, cam_name, target_ts, max_diff_ms):
    """按时间戳在 pkl 中二分查找最匹配的帧.

    Args:
        target_ts:   目标时间戳 (微秒)
        max_diff_ms: 最大允许时差 (毫秒)

    返回 (frame_idx, info_dict) 或 (None, None).
    """
    ts_list = []
    for i, info in enumerate(infos):
        if cam_name in info.get('cams', {}):
            data_path = info['cams'][cam_name]['data_path']
            try:
                ts = get_ts_from_filename(data_path)
                ts_list.append((ts, i))
            except ValueError:
                pass

    if not ts_list:
        return None, None

    ts_list.sort(key=lambda x: x[0])
    timestamps = [t[0] for t in ts_list]
    pos = bisect.bisect_left(timestamps, target_ts)

    candidates = []
    if pos < len(timestamps):
        candidates.append((timestamps[pos], ts_list[pos][1]))
    if pos > 0:
        candidates.append((timestamps[pos - 1], ts_list[pos - 1][1]))

    if not candidates:
        return None, None

    best_ts, best_idx = min(candidates, key=lambda x: abs(x[0] - target_ts))
    # 文件名中时间戳是微秒级, 除以 1000 转换为毫秒后与阈值比较
    diff_ms = abs(best_ts - target_ts) / 1000.0

    print(f"\n时间戳匹配:")
    print(f"  目标图片 ts (μs): {target_ts}")
    print(f"  pkl 最近 ts  (μs): {best_ts} (frame[{best_idx}])")
    print(f"  时差:        {diff_ms:.1f} ms (阈值: {max_diff_ms} ms)")

    if diff_ms > max_diff_ms:
        print(f"  时差超过阈值 {max_diff_ms}ms, 无匹配帧!")
        return None, None

    return best_idx, infos[best_idx]


def _load_lidar_points(lidar_path):
    """自适应加载点云, 自动检测每点字段数 (3/4/5/6)"""
    raw = np.fromfile(lidar_path, dtype=np.float32)
    for dim in [4, 5, 6, 3]:  # 优先尝试常见格式
        if raw.size % dim == 0:
            return raw.reshape(-1, dim)
    raise ValueError(f"Cannot reshape {raw.size} floats into any of [3,4,5,6] per point")


def draw_bev(lidar_points, show_range=50, resolution=1000, point_size=1,
             intensity=None):
    """绘制 BEV 俯视图 (FLU 坐标系: x=前, y=左, z=上).

    调用方须确保输入点云已转换到 FLU (ego) 坐标系，
    即已应用 lidar2ego_rotation.

    Args:
        lidar_points: Nx3+ ndarray, [x(前), y(左), z(上), ...]
        show_range:   显示范围 (米)
        resolution:   画布分辨率
        point_size:   点大小
        intensity:    反射强度数组 (N,) 或 None. 提供时按强度着色, 否则按高度着色.
    """
    canvas_size = int(2 * show_range / (2 * show_range / resolution))
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    canvas[:, :] = (40, 40, 40)  # 深灰背景

    xs = lidar_points[:, 0]  # FLU: x = 前
    ys = lidar_points[:, 1]  # FLU: y = 左

    # 过滤范围外的点
    mask = (np.abs(xs) < show_range) & (np.abs(ys) < show_range)
    xs, ys = xs[mask], ys[mask]

    if len(xs) == 0:
        return canvas

    # 映射到画布坐标: 前→画布上, 左→画布左
    scale = canvas_size / (2 * show_range)
    px = ((-ys + show_range) * scale).astype(np.int32)  # 左→画布左
    py = ((-xs + show_range) * scale).astype(np.int32)  # 前→画布上

    # 裁剪到画布内
    valid = (px >= 0) & (px < canvas_size) & (py >= 0) & (py < canvas_size)
    px, py = px[valid], py[valid]

    # 按高度或反射强度着色: 低→蓝, 高→红
    if intensity is not None and len(intensity) > 0:
        color_vals = intensity[mask][valid] if valid.sum() > 0 else intensity[:0]
    else:
        color_vals = lidar_points[mask][valid, 2] if valid.sum() > 0 else lidar_points[:0, 2]

    if len(color_vals) > 0:
        v_min, v_max = color_vals.min(), color_vals.max()
        if v_max > v_min:
            v_norm = (color_vals - v_min) / (v_max - v_min)
        else:
            v_norm = np.zeros_like(color_vals)
        # 蓝色(低) → 绿色 → 红色(高)
        colors = np.zeros((len(color_vals), 3), dtype=np.uint8)
        colors[:, 2] = (v_norm * 255).astype(np.uint8)            # R
        colors[:, 0] = ((1 - v_norm) * 255).astype(np.uint8)      # B
        colors[:, 1] = (np.abs(v_norm - 0.5) * 255).astype(np.uint8)  # G
    else:
        colors = np.tile([100, 200, 100], (len(px), 1))

    for i in range(len(px)):
        cv2.circle(canvas, (px[i], py[i]), point_size,
                   colors[i].tolist(), -1)

    # 画坐标轴
    center = canvas_size // 2
    cv2.line(canvas, (0, center), (canvas_size, center), (80, 80, 80), 1)
    cv2.line(canvas, (center, 0), (center, canvas_size), (80, 80, 80), 1)

    # 原点 (ego 位置), 前向箭头指向上方
    cv2.circle(canvas, (center, center), 5, (0, 255, 255), -1)
    cv2.arrowedLine(canvas, (center, center), (center, center - 30),
                    (0, 255, 255), 2, tipLength=0.3)
    cv2.putText(canvas, 'Fwd', (center - 25, center - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return canvas


def main():
    args = parse_args()

    # 加载 pkl
    dataset = pickle.load(open(args.pkl, 'rb'))
    print(f"Loaded pkl with {len(dataset['infos'])} frames")
    print(f"Cameras: {dataset['metadata'].get('cam_names', 'unknown')}")

    os.makedirs(args.save_dir, exist_ok=True)

    # 确定要处理的帧
    infos = dataset['infos']
    if args.img is not None:
        # ----- --img 模式: 输入图片路径, 自动时间戳匹配 -----
        if not os.path.exists(args.img):
            print(f"ERROR: 图片不存在: {args.img}")
            sys.exit(1)
        target_img_ts = get_ts_from_filename(args.img)
        print(f"输入图片: {args.img}  (ts={target_img_ts} μs)")

        if args.cam is None:
            args.cam = detect_cam_from_path(args.img)
            if args.cam is None:
                print("ERROR: 无法从图片路径自动推断相机名称, 请用 --cam 指定")
                sys.exit(1)
            print(f"自动推断相机: {args.cam}")

        frame_idx, info = find_best_frame_from_pkl(
            infos, args.cam, target_img_ts, args.max_diff_ms)
        if frame_idx is None:
            print("ERROR: 在 pkl 中未找到时间戳匹配的帧")
            sys.exit(1)
        frames = [(frame_idx, info)]
    elif args.token is not None:
        # 按 token 查找
        token_to_info = {i['token']: (idx, i) for idx, i in enumerate(infos)}
        if args.token not in token_to_info:
            print(f"ERROR: token '{args.token}' not found in pkl")
            return
        idx, info = token_to_info[args.token]
        frames = [(idx, info)]
        print(f"Matched: token={args.token}, idx={idx}")
    elif args.idx is not None:
        if args.idx < 0 or args.idx >= len(infos):
            print(f"ERROR: idx={args.idx} out of range [0, {len(infos)-1}]")
            return
        frames = [(args.idx, infos[args.idx])]
    else:
        frames = list(enumerate(infos))

    for idx, info in frames:
        token = info['token']
        print(f"\n[{idx+1}/{len(infos)}] Frame: {token}")

        # 1. 读取点云, 用 lidar2ego_rotation 转换到 FLU 坐标系再绘制 BEV
        lidar_path = info['lidar_path']
        print(f"  LiDAR: {lidar_path}")
        if os.path.exists(lidar_path):
            lidar_points = _load_lidar_points(lidar_path)
            # 打印原始传感器帧的范围 (Velodyne: x=右, y=前, z=上)
            print(f"    Points: {len(lidar_points)} ({lidar_points.shape[1]} fields/point), "
                  f"sensor frame: x=[{lidar_points[:,0].min():.1f},{lidar_points[:,0].max():.1f}], "
                  f"y=[{lidar_points[:,1].min():.1f},{lidar_points[:,1].max():.1f}]")

            # 应用 lidar2ego 旋转, 将点云从 Velodyne 传感器帧 (x=右,y=前) 转到 FLU (x=前,y=左)
            lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
            pts_xyz = lidar_points[:, :3] @ lidar2ego_r.T  # 旋转
            # 可选: 也应用平移 (lidar 传感器在车顶的位置偏移)
            lidar2ego_t = np.array(info['lidar2ego_translation'], dtype=np.float32)
            pts_xyz += lidar2ego_t

            # 只保留 x, y, z (丢弃 intensity, ring 等额外字段)
            lidar_points_flu = pts_xyz  # shape: (N, 3)
            # 提取反射强度用于着色
            pts_intensity = lidar_points[:, 3] if lidar_points.shape[1] >= 4 else None

            print(f"    Ego FLU frame:    x=[{lidar_points_flu[:,0].min():.1f},{lidar_points_flu[:,0].max():.1f}], "
                  f"y=[{lidar_points_flu[:,1].min():.1f},{lidar_points_flu[:,1].max():.1f}]")
            print(f"    lidar2ego_rotation (w,x,y,z): {info['lidar2ego_rotation']}")
            print(f"    lidar2ego_translation:        {lidar2ego_t}")

            bev_img = draw_bev(lidar_points_flu, show_range=args.show_range,
                               point_size=args.point_size,
                               intensity=pts_intensity)
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
        print(f"\n  [BEV] 已通过 lidar2ego 转换到 FLU (ego) 坐标系, 前向朝上, 原点(黄点)=ego位置, 箭头=前向")
        print(f"  [BEV] 颜色: 蓝=低反射强度, 红=高反射强度")

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
