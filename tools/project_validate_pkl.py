#!/usr/bin/env python3
"""
从 pkl 读取 sensor2lidar 和相机内参, 验证激光雷达点云到相机图像的投影是否准确。
通过输入一张图片路径, 自动在 pkl 中找到时间戳最接近的帧和对应的 lidar bin。

用法:
    python tools/project_validate_pkl.py --pkl ./custom_infos_val.pkl --img /path/to/image.jpg
    python tools/project_validate_pkl.py --pkl ./custom_infos_val.pkl --img /path/to/image.jpg --cam CAM_BACK
    python tools/project_validate_pkl.py --pkl ./custom_infos_val.pkl --img /path/to/image.jpg --max-diff-ms 200 --undistort
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


# ===================== 全局配置 =====================
SAVE_IMG = "./align_validate_pkl.jpg"
MAX_TIME_DIFF_MS = 100
# ====================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description='通过图片路径从 pkl 匹配 lidar, 验证 lidar→camera 投影对齐')
    parser.add_argument('--pkl', type=str, required=True,
                        help='pkl 文件路径')
    parser.add_argument('--img', type=str, required=True,
                        help='要验证的相机图片路径 (通过时间戳匹配)')
    parser.add_argument('--cam', type=str, default=None,
                        choices=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
                        help='相机名称 (默认从图片路径自动推断, 如 /path/CAM_BACK/xx.jpg → CAM_BACK)')
    parser.add_argument('--max-diff-ms', type=int, default=MAX_TIME_DIFF_MS,
                        help=f'lidar 与图片最大允许时差 (ms, 默认: {MAX_TIME_DIFF_MS})')
    parser.add_argument('--save', type=str, default=SAVE_IMG,
                        help=f'输出图片路径 (默认: {SAVE_IMG})')
    parser.add_argument('--point-size', type=int, default=1,
                        help='投影像素大小 (默认: 1)')
    parser.add_argument('--min-depth', type=float, default=0.2,
                        help='最小深度过滤 (默认: 0.2m)')
    parser.add_argument('--undistort', action='store_true',
                        help='是否做图像去畸变 (需同时提供 --calib-root)')
    parser.add_argument('--calib-root', type=str, default='./data/calibration',
                        help='标定文件根目录 (用于读取畸变系数 D, 默认: ./data/calibration)')
    parser.add_argument('--color', type=str, default='green',
                        choices=['green', 'red', 'blue', 'yellow', 'cyan', 'white'],
                        help='投影点颜色 (默认: green)')
    return parser.parse_args()


CAM_NAMES = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
             'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def get_ts_from_filename(file_path):
    """从文件名提取时间戳 (取名字中第一组连续数字)"""
    basename = os.path.basename(file_path)
    match = re.search(r"\d+", basename)
    if not match:
        raise ValueError(f"无法从文件名提取时间戳: {basename}")
    return int(match.group())


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


def invert_extrinsic(R, t):
    """外参求逆: lidar→cam = inv(cam→lidar)"""
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv


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
    优先精确路径匹配, 否则按时间戳二分查找.
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
    diff_ms = abs(best_ts - target_img_ts)

    print(f"\n时间戳匹配:")
    print(f"  目标图片 ts: {target_img_ts}")
    print(f"  pkl 最近 ts: {best_ts} (frame[{best_idx}])")
    print(f"  时差:        {diff_ms} ms (阈值: {max_diff_ms} ms)")

    if diff_ms > max_diff_ms:
        print(f"  时差超过阈值 {max_diff_ms}ms, 无匹配帧!")
        return None, None

    return best_idx, infos[best_idx]


COLOR_MAP = {
    'green': (0, 255, 0),
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'cyan': (255, 255, 0),
    'white': (255, 255, 255),
}


def main():
    args = parse_args()

    # ==================== 1. 检查输入图片 ====================
    if not os.path.exists(args.img):
        print(f"ERROR: 图片不存在: {args.img}")
        sys.exit(1)
    target_img_ts = get_ts_from_filename(args.img)
    print(f"输入图片: {args.img}  (ts={target_img_ts})")

    # 自动推断相机名称
    if args.cam is None:
        args.cam = detect_cam_from_path(args.img)
        if args.cam is None:
            print("ERROR: 无法从图片路径自动推断相机名称, 请用 --cam 指定")
            sys.exit(1)
        print(f"自动推断相机: {args.cam}")

    # ==================== 2. 加载 pkl ====================
    print(f"\n加载 pkl: {args.pkl}")
    dataset = pickle.load(open(args.pkl, 'rb'))
    infos = dataset['infos']
    print(f"  总帧数: {len(infos)}")
    print(f"  相机列表: {dataset['metadata'].get('cam_names', 'unknown')}")

    # ==================== 3. 按时间戳在 pkl 中匹配帧 ====================
    frame_idx, info = find_best_frame_from_pkl(infos, args.cam, target_img_ts, args.max_diff_ms)
    if frame_idx is None:
        print("ERROR: 在 pkl 中未找到时间戳匹配的帧")
        sys.exit(1)

    token = info['token']
    print(f"\n=== 匹配帧 [{frame_idx}] token={token} ===")

    # ==================== 4. 加载 lidar 点云 ====================
    lidar_path = info['lidar_path']
    print(f"LiDAR path: {lidar_path}")
    if not os.path.exists(lidar_path):
        print(f"ERROR: LiDAR 文件不存在: {lidar_path}")
        sys.exit(1)
    pts_lidar_raw = load_lidar_bin(lidar_path)
    pts_lidar = pts_lidar_raw[:, :3]
    print(f"  点数: {len(pts_lidar)}, 范围: "
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

    # ==================== 6. 读取用户传入的图像 ====================
    img = cv2.imread(args.img)
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
    # sensor2lidar = cam→lidar, 求逆得 lidar→cam
    l2s_R, l2s_t = invert_extrinsic(sensor2lidar_R, sensor2lidar_t)

    # 变换 lidar 点到相机坐标系
    pts_cam = (l2s_R @ pts_lidar.T).T + l2s_t
    mask_front = pts_cam[:, 2] > args.min_depth
    pts_valid = pts_cam[mask_front]
    print(f"\n投影:")
    print(f"  相机前方点数 (深度 > {args.min_depth}m): {len(pts_valid)}")
    if len(pts_valid) > 0:
        print(f"  相机坐标范围: "
              f"x=[{pts_valid[:,0].min():.2f},{pts_valid[:,0].max():.2f}], "
              f"y=[{pts_valid[:,1].min():.2f},{pts_valid[:,1].max():.2f}], "
              f"z=[{pts_valid[:,2].min():.2f},{pts_valid[:,2].max():.2f}]")

    # 透视投影
    pts_homo = pts_valid.T
    pixels = (K_proj @ pts_homo).T
    u = pixels[:, 0] / pixels[:, 2]
    v = pixels[:, 1] / pixels[:, 2]

    # ==================== 9. 绘制投影点 ====================
    color = COLOR_MAP[args.color]
    n_drawn = 0
    for ui, vi in zip(u, v):
        x = int(round(ui))
        y = int(round(vi))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), radius=args.point_size,
                       color=color, thickness=-1)
            n_drawn += 1

    print(f"  绘制的投影点数: {n_drawn} / {len(pts_valid)}")

    # ==================== 10. 输出调试信息 ====================
    print(f"\n=== pkl 标定信息 (帧 [{frame_idx}]) ===")
    lidar2ego_t = np.array(info['lidar2ego_translation'])
    lidar2ego_r = np.array(info['lidar2ego_rotation'])
    print(f"  lidar2ego_translation: {lidar2ego_t}")
    print(f"  lidar2ego_rotation (w,x,y,z): {lidar2ego_r}")

    s2e_t = np.array(cam_data['sensor2ego_translation'])
    s2e_r = np.array(cam_data['sensor2ego_rotation'])
    print(f"  sensor2ego_translation: {s2e_t}")
    print(f"  sensor2ego_rotation (w,x,y,z): {s2e_r}")

    # ==================== 11. 保存 ====================
    cv2.imwrite(args.save, img)
    print(f"\n投影验证图已保存: {args.save}")
    print(f"  {args.color} 点 = lidar 点云投影, 应正确覆盖在图像对应物体上")


if __name__ == "__main__":
    main()
