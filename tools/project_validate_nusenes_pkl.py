#!/usr/bin/env python3
"""
从 pkl 读取 sensor2lidar 和相机内参, 验证激光雷达点云到相机图像的投影是否准确。
支持三种指定帧的方式: --img / --idx / --token。

用法:
    # 方式1: 输入图片路径, 自动时间戳匹配
    python tools/project_validate_nusenes_pkl.py --pkl ./xx.pkl --img /path/to/CAM_FRONT/xxx.jpg

    # 方式2: 按帧索引
    python tools/project_validate_nusenes_pkl.py --pkl ./xx.pkl --idx 0 --cam CAM_FRONT

    # 方式3: 按 token
    python tools/project_validate_nusenes_pkl.py --pkl ./xx.pkl --token xxxxx --cam CAM_FRONT

    python tools/project_validate_nusenes_pkl.py --pkl ./xx.pkl --img /path/to/image.jpg --max-diff-ms 200 --undistort
"""

import argparse
import bisect
import os
import pickle
import re
import sys
import yaml

import cv2
import numpy as np
from pyquaternion import Quaternion


# ===================== 全局配置 =====================
SAVE_IMG = "./align_validate_pkl.jpg"
MAX_TIME_DIFF_MS = 100
# ====================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description='验证 lidar→camera 投影对齐, 支持 --img / --idx / --token 三种方式')
    parser.add_argument('--pkl', type=str, required=True,
                        help='pkl 文件路径')

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--img', type=str, default=None,
                      help='相机图片路径, 通过时间戳自动匹配帧')
    mode.add_argument('--idx', type=int, default=None,
                      help='帧索引 (从 0 开始), 需配合 --cam')
    mode.add_argument('--token', type=str, default=None,
                      help='帧 token, 需配合 --cam')

    parser.add_argument('--cam', type=str, default=None,
                        choices=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
                        help='相机名称 (--img 模式下可从路径自动推断, --idx/--token 模式下必须指定)')
    parser.add_argument('--max-diff-ms', type=int, default=MAX_TIME_DIFF_MS,
                        help=f'--img 模式下 lidar 与图片最大允许时差 (ms, 默认: {MAX_TIME_DIFF_MS})')
    parser.add_argument('--save', type=str, default=SAVE_IMG,
                        help=f'输出图片路径 (默认: {SAVE_IMG})')
    parser.add_argument('--point-size', type=int, default=2,
                        help='投影像素大小 (默认: 2)')
    parser.add_argument('--min-depth', type=float, default=0.2,
                        help='最小深度过滤 (默认: 0.2m)')
    parser.add_argument('--undistort', action='store_true',
                        help='是否做图像去畸变 (需同时提供 --calib-root)')
    parser.add_argument('--calib-root', type=str, default='./data/calibration',
                        help='标定文件根目录 (用于读取畸变系数 D, 默认: ./data/calibration)')
    return parser.parse_args()


CAM_NAMES = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
             'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def get_ts_from_filename(file_path):
    """从文件名提取时间戳, 返回微秒级整数.

    文件名格式: n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151061512404.jpg
    第一组 "008" 是场景号, 最后一组 "1533151061512404" 才是微秒时间戳。
    """
    basename = os.path.basename(file_path)
    matches = re.findall(r"\d+", basename)
    if not matches:
        raise ValueError(f"无法从文件名提取时间戳: {basename}")
    # 取最后一组 (最长的一组数字 = 微秒时间戳)
    return int(matches[-1])


def detect_cam_from_path(img_path):
    """从图片路径中自动推断相机名称, 如 .../CAM_BACK/xxx.jpg → CAM_BACK"""
    norm = img_path.replace('\\', '/')
    for cam in CAM_NAMES:
        if cam in norm:
            return cam
    return None


def load_lidar_bin(path):
    """加载 lidar bin 文件, 自适应检测每点字段数"""
    raw = np.fromfile(path, dtype=np.float32)
    for dim in [4, 5, 6, 3]:
        if raw.size % dim == 0:
            return raw.reshape(-1, dim)
    raise ValueError(f"Cannot reshape {raw.size} floats into any of [3,4,5,6] per point")


def build_lidar2camera(sensor2lidar_R, sensor2lidar_t):
    """从 pkl 的 sensor2lidar 构造 lidar→camera 4x4 变换矩阵.

    nuScenes pkl 中 sensor2lidar 的语义为:
        p_lidar = p_sensor @ sensor2lidar_R + sensor2lidar_t
    因此通过 4x4 矩阵求逆得到 lidar→camera, 与 vis.py 保持一致,
    避免行/列向量约定混淆。
    """
    camera2lidar = np.eye(4, dtype=np.float32)
    camera2lidar[:3, :3] = sensor2lidar_R
    camera2lidar[:3, 3] = sensor2lidar_t
    lidar2camera = np.linalg.inv(camera2lidar)
    return lidar2camera


def read_opencv_yaml(file_path):
    """读取 OpenCV 格式标定 yaml (含 !!opencv-matrix)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        raise FileNotFoundError(f"标定文件不存在 {file_path}: {str(e)}")

    if content.startswith('%YAML'):
        idx = content.find('---')
        if idx != -1:
            content = content[idx+3:]
        else:
            lines = content.split('\n')[1:]
            content = '\n'.join(lines)

    mat_pattern = r'!!opencv-matrix\s*\n\s*rows:\s*(\d+)\s*\n\s*cols:\s*(\d+)\s*\n\s*dt:\s*(\w+)\s*\n\s*data:\s*\[(.*?)\]'
    mat_dict = {}
    matches = list(re.finditer(mat_pattern, content, re.DOTALL))

    for idx, match in enumerate(matches):
        rows = int(match.group(1))
        cols = int(match.group(2))
        data_str = match.group(4)
        clean_str = data_str.replace("\n", "").replace(" ", "")
        str_list = clean_str.split(",")
        data = []
        for s in str_list:
            s_strip = s.strip()
            if s_strip == "" or s_strip == ".":
                continue
            data.append(float(s_strip))
        mat = np.array(data, dtype=np.float32).reshape(rows, cols)
        mat_dict[idx] = mat
        content = content.replace(match.group(0), f'__MAT_PLACE_{idx}__', 1)

    raw = yaml.safe_load(content)

    def fill_mat(obj):
        if isinstance(obj, dict):
            return {k: fill_mat(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [fill_mat(item) for item in obj]
        elif isinstance(obj, str) and '__MAT_PLACE_' in obj:
            m_idx = int(re.search(r'__MAT_PLACE_(\d+)__', obj).group(1))
            return mat_dict.get(m_idx, obj)
        return obj

    return fill_mat(raw)


def get_distortion_from_calib(calib_root, cam_name):
    """从标定文件读取畸变系数 D"""
    cam_name_map = {
        'CAM_FRONT': 'frontmain.yaml',
        'CAM_BACK': 'rear.yaml',
        'CAM_FRONT_LEFT': 'leftrepeater.yaml',
        'CAM_FRONT_RIGHT': 'rightrepeater.yaml',
        'CAM_BACK_LEFT': 'leftpillar.yaml',
        'CAM_BACK_RIGHT': 'rightpillar.yaml',
    }
    calib_fname = cam_name_map.get(cam_name)
    if not calib_fname:
        raise ValueError(f'Unknown camera: {cam_name}')
    calib_path = os.path.join(calib_root, 'camera', calib_fname)
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f'Camera calibration not found: {calib_path}')
    calib = read_opencv_yaml(calib_path)
    return calib['D'].astype(np.float32)


def find_best_frame_from_pkl(infos, cam_name, target_img_ts, max_diff_ms):
    """
    在 pkl 的 infos 列表中, 找到 cam_name 相机的时间戳与 target_img_ts 最接近的帧.

    Args:
        target_img_ts: 目标时间戳 (微秒)
        max_diff_ms:   最大允许时差 (毫秒)

    返回 (frame_idx, info_dict) 或 (None, None).
    """
    # 建立每个帧中 cam_name 相机的时间戳列表
    ts_list = []       # [(timestamp, frame_idx), ...]
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

    # 二分查找最接近的时间戳
    timestamps = [t[0] for t in ts_list]
    pos = bisect.bisect_left(timestamps, target_img_ts)

    candidates = []
    if pos < len(timestamps):
        candidates.append((timestamps[pos], ts_list[pos][1]))
    if pos > 0:
        candidates.append((timestamps[pos - 1], ts_list[pos - 1][1]))

    if not candidates:
        return None, None

    best_ts, best_idx = min(candidates, key=lambda x: abs(x[0] - target_img_ts))
    # 文件名中时间戳是微秒级, 除以 1000 转换为毫秒后与阈值比较
    diff_ms = abs(best_ts - target_img_ts) / 1000.0

    print(f"\n时间戳匹配:")
    print(f"  目标图片 ts (μs): {target_img_ts}")
    print(f"  pkl 最近 ts  (μs): {best_ts} (frame[{best_idx}])")
    print(f"  时差:        {diff_ms:.1f} ms (阈值: {max_diff_ms} ms)")

    if diff_ms > max_diff_ms:
        print(f"  时差超过阈值 {max_diff_ms}ms, 无匹配帧!")
        return None, None

    return best_idx, infos[best_idx]


def main():
    args = parse_args()

    # ==================== 1. 加载 pkl ====================
    print(f"\n加载 pkl: {args.pkl}")
    dataset = pickle.load(open(args.pkl, 'rb'))
    infos = dataset['infos']
    print(f"  总帧数: {len(infos)}")
    print(f"  相机列表: {dataset['metadata'].get('cam_names', 'unknown')}")

    # ==================== 2. 确定目标帧 (三种方式) ====================
    if args.img is not None:
        # ----- 方式1: --img 时间戳匹配 -----
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

        frame_idx, info = find_best_frame_from_pkl(infos, args.cam, target_img_ts, args.max_diff_ms)
        if frame_idx is None:
            print("ERROR: 在 pkl 中未找到时间戳匹配的帧")
            sys.exit(1)

        img_path = args.img  # 使用用户传入的图片

    elif args.idx is not None:
        # ----- 方式2: --idx 索引 -----
        if args.idx < 0 or args.idx >= len(infos):
            print(f"ERROR: idx={args.idx} 超出范围 [0, {len(infos)-1}]")
            sys.exit(1)
        if args.cam is None:
            print("ERROR: --idx 模式下必须用 --cam 指定相机")
            sys.exit(1)

        frame_idx = args.idx
        info = infos[frame_idx]
        if args.cam not in info.get('cams', {}):
            print(f"ERROR: 相机 {args.cam} 不存在于帧 [{frame_idx}] 中, "
                  f"可用: {list(info.get('cams', {}).keys())}")
            sys.exit(1)

        img_path = info['cams'][args.cam]['data_path']
        if not os.path.exists(img_path):
            print(f"ERROR: pkl 中记录的图片不存在: {img_path}")
            sys.exit(1)
        print(f"输入索引: [{frame_idx}] → 图片: {img_path}")

    elif args.token is not None:
        # ----- 方式3: --token 匹配 -----
        token_to_idx = {i['token']: idx for idx, i in enumerate(infos)}
        if args.token not in token_to_idx:
            print(f"ERROR: token '{args.token}' 不在 pkl 中")
            sys.exit(1)
        if args.cam is None:
            print("ERROR: --token 模式下必须用 --cam 指定相机")
            sys.exit(1)

        frame_idx = token_to_idx[args.token]
        info = infos[frame_idx]
        if args.cam not in info.get('cams', {}):
            print(f"ERROR: 相机 {args.cam} 不存在于 token={args.token} 中, "
                  f"可用: {list(info.get('cams', {}).keys())}")
            sys.exit(1)

        img_path = info['cams'][args.cam]['data_path']
        if not os.path.exists(img_path):
            print(f"ERROR: pkl 中记录的图片不存在: {img_path}")
            sys.exit(1)
        print(f"输入 token: {args.token}  → 索引 [{frame_idx}] → 图片: {img_path}")
    else:
        print("ERROR: 必须指定 --img / --idx / --token 之一")
        sys.exit(1)

    token = info['token']
    print(f"\n=== 匹配帧 [{frame_idx}] token={token} ===")

    # ==================== 4. 加载 lidar 点云, 转换到 FLU (ego) 坐标系 ====================
    lidar_path = info['lidar_path']
    print(f"LiDAR path: {lidar_path}")
    if not os.path.exists(lidar_path):
        print(f"ERROR: LiDAR 文件不存在: {lidar_path}")
        sys.exit(1)
    pts_lidar_raw = load_lidar_bin(lidar_path)
    pts_lidar_sensor = pts_lidar_raw[:, :3]  # Velodyne 传感器帧: x=右, y=前, z=上
    # 提取反射强度 (第 4 列), 用于投影着色
    pts_intensity = pts_lidar_raw[:, 3] if pts_lidar_raw.shape[1] >= 4 else np.ones(len(pts_lidar_raw))
    print(f"  点数: {len(pts_lidar_sensor)}")
    print(f"  Velodyne 传感器帧 (x=右, y=前, z=上): "
          f"x=[{pts_lidar_sensor[:,0].min():.1f},{pts_lidar_sensor[:,0].max():.1f}], "
          f"y=[{pts_lidar_sensor[:,1].min():.1f},{pts_lidar_sensor[:,1].max():.1f}], "
          f"z=[{pts_lidar_sensor[:,2].min():.1f},{pts_lidar_sensor[:,2].max():.1f}]")

    # 用 lidar2ego 转换到 FLU (ego) 坐标系: x=前, y=左, z=上
    lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    lidar2ego_t = np.array(info['lidar2ego_translation'], dtype=np.float32)
    pts_lidar = pts_lidar_sensor @ lidar2ego_r.T + lidar2ego_t  # 旋转 + 平移
    print(f"  Ego FLU 帧       (x=前, y=左, z=上): "
          f"x=[{pts_lidar[:,0].min():.1f},{pts_lidar[:,0].max():.1f}], "
          f"y=[{pts_lidar[:,1].min():.1f},{pts_lidar[:,1].max():.1f}], "
          f"z=[{pts_lidar[:,2].min():.1f},{pts_lidar[:,2].max():.1f}]")

    # ==================== 5. 获取相机内参/外参 ====================
    cam_data = info['cams'][args.cam]
    img_path_in_pkl = cam_data['data_path']
    K = np.array(cam_data['cam_intrinsic'], dtype=np.float32)
    sensor2lidar_R = np.array(cam_data['sensor2lidar_rotation'], dtype=np.float32)
    sensor2lidar_t = np.array(cam_data['sensor2lidar_translation'], dtype=np.float32)

    print(f"\n相机: {args.cam}")
    print(f"  pkl 中图像路径: {img_path_in_pkl}")
    print(f"  K (3x3):\n{K}")
    print(f"  sensor2lidar_R:\n{sensor2lidar_R}")
    print(f"  sensor2lidar_t: {sensor2lidar_t}")

    # ==================== 6. 读取图像 (--img 用传入的, --idx/--token 用 pkl 中记录的) ====================
    img = cv2.imread(img_path)
    if img is None:
        print(f"ERROR: 读取图片失败: {args.img}")
        sys.exit(1)
    h, w = img.shape[:2]
    print(f"\n图像尺寸: {w}x{h}")

    # ==================== 7. 图像去畸变 (可选) ====================
    if args.undistort:
        D = get_distortion_from_calib(args.calib_root, args.cam)
        print(f"畸变系数 D: {D}")
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=1)
        img = cv2.undistort(img, K, D, None, new_K)
        K_proj = new_K
        print("  已应用去畸变")
    else:
        K_proj = K

    # ==================== 8. lidar → camera 投影 ====================
    # 仍使用 pkl 中预计算好的 sensor2lidar (Velodyne 传感器帧 → camera 传感器帧)。
    # 该矩阵在 pkl 生成阶段已统一好坐标约定，与 tools/analysis_tools/vis.py 保持一致。
    lidar2camera = build_lidar2camera(sensor2lidar_R, sensor2lidar_t)

    # 变换 lidar 点到相机坐标系 (齐次坐标 + 行向量约定)
    # 注意：sensor2lidar 的输入必须是 Velodyne 传感器帧原始点云，不是 FLU 帧
    pts_lidar_homo = np.concatenate(
        [pts_lidar_sensor, np.ones((pts_lidar_sensor.shape[0], 1), dtype=np.float32)],
        axis=1)
    pts_cam = (pts_lidar_homo @ lidar2camera.T)[:, :3]
    mask_front = pts_cam[:, 2] > args.min_depth
    pts_valid = pts_cam[mask_front]
    print(f"\n投影 (Velodyne → camera sensor):")
    print(f"  相机前方点数 (深度 > {args.min_depth}m): {len(pts_valid)}")
    if len(pts_valid) > 0:
        print(f"  相机坐标范围: "
              f"x=[{pts_valid[:,0].min():.2f},{pts_valid[:,0].max():.2f}], "
              f"y=[{pts_valid[:,1].min():.2f},{pts_valid[:,1].max():.2f}], "
              f"z=[{pts_valid[:,2].min():.2f},{pts_valid[:,2].max():.2f}]")

    # 透视投影
    pts_homo = pts_valid / pts_valid[:, 2:3]
    pixels = (K_proj @ pts_homo.T).T
    u = pixels[:, 0]
    v = pixels[:, 1]

    # ==================== 9. 按反射强度着色绘制投影点 ====================
    # 过滤出前方点对应的强度值
    valid_intensity = pts_intensity[mask_front]

    # 归一化强度到 [0, 1]
    i_min, i_max = valid_intensity.min(), valid_intensity.max()
    if i_max > i_min:
        i_norm = (valid_intensity - i_min) / (i_max - i_min)
    else:
        i_norm = np.zeros_like(valid_intensity)
    print(f"  反射强度范围: [{i_min:.1f}, {i_max:.1f}]")

    n_drawn = 0
    for ui, vi, iv in zip(u, v, i_norm):
        x = int(round(ui))
        y = int(round(vi))
        if 0 <= x < w and 0 <= y < h:
            # 低强度→蓝色, 高强度→红色
            r = int(iv * 255)
            b = int((1 - iv) * 255)
            g = int(abs(iv - 0.5) * 255)
            cv2.circle(img, (x, y), radius=args.point_size,
                       color=(b, g, r), thickness=-1)
            n_drawn += 1

    print(f"  绘制的投影点数: {n_drawn} / {len(pts_valid)}")

    # ==================== 10. 输出调试信息 ====================
    print(f"\n=== 坐标变换信息 (帧 [{frame_idx}]) ===")
    print(f"  1) Velodyne 传感器帧 → FLU (ego) 帧 (lidar2ego), 仅用于 BEV/调试:")
    print(f"     lidar2ego_rotation (w,x,y,z): {info['lidar2ego_rotation']}")
    print(f"     lidar2ego_translation:        {np.array(info['lidar2ego_translation'])}")
    print(f"  2) Velodyne 传感器帧 → {args.cam} 传感器帧 (sensor2lidar), 用于投影:")
    print(f"     sensor2lidar_rotation:\n{cam_data['sensor2lidar_rotation']}")
    print(f"     sensor2lidar_translation:     {np.array(cam_data['sensor2lidar_translation'])}")

    # ==================== 11. 保存 ====================
    cv2.imwrite(args.save, img)
    print(f"\n投影验证图已保存: {args.save}")
    print(f"  颜色: 低反射强度→蓝色, 高反射强度→红色, lidar 投影点应覆盖在图像对应物体上")


if __name__ == "__main__":
    main()
