#!/usr/bin/env python3
"""
可视化激光雷达点云 BEV 俯视图, 直接读取标定文件和 bin 文件。

用法:
    python tools/vis_custom_pkl.py \
        --bin data/custom_data/sequence2/LIDAR_TOP/1781488453100.bin \
        --calib data/calibration/lidar/helios32_top_vehicle_extrinsics.yaml \
        --show-range 50
"""

import argparse
import os

import cv2
import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize lidar point cloud BEV')
    parser.add_argument('--bin', type=str, required=True,
                        help='Path to a single lidar .bin file')
    parser.add_argument('--calib', type=str,
                        default='data/calibration/lidar/helios32_top_vehicle_extrinsics.yaml',
                        help='Path to lidar extrinsics calibration yaml')
    parser.add_argument('--show-range', type=float, default=50,
                        help='BEV display range in meters (default: 50)')
    parser.add_argument('--point-size', type=int, default=1,
                        help='Point size in BEV canvas (default: 1)')
    parser.add_argument('--save-dir', type=str, default='./vis_custom_output',
                        help='Directory to save visualization images')
    parser.add_argument('--frame', type=str, default='vehicle',
                        choices=['lidar', 'vehicle'],
                        help='Coordinate frame to visualize: lidar (raw) or vehicle (after transform)')
    return parser.parse_args()


def _load_lidar_points(lidar_path):
    """自适应加载点云, 自动检测每点字段数 (3/4/5/6)"""
    raw = np.fromfile(lidar_path, dtype=np.float32)
    for dim in [4, 5, 6, 3]:  # 优先尝试常见格式
        if raw.size % dim == 0:
            return raw.reshape(-1, dim)
    raise ValueError(f"Cannot reshape {raw.size} floats into any of [3,4,5,6] per point")


def quaternion_to_matrix(qx, qy, qz, qw):
    """四元数转旋转矩阵 (scalar-last 格式: x,y,z,w), 纯 numpy 实现"""
    R = np.zeros((3, 3))
    R[0, 0] = 1 - 2 * (qy**2 + qz**2)
    R[0, 1] = 2 * (qx * qy - qz * qw)
    R[0, 2] = 2 * (qx * qz + qy * qw)
    R[1, 0] = 2 * (qx * qy + qz * qw)
    R[1, 1] = 1 - 2 * (qx**2 + qz**2)
    R[1, 2] = 2 * (qy * qz - qx * qw)
    R[2, 0] = 2 * (qx * qz - qy * qw)
    R[2, 1] = 2 * (qy * qz + qx * qw)
    R[2, 2] = 1 - 2 * (qx**2 + qy**2)
    return R


def read_calib_yaml(calib_path):
    """读取激光雷达外参标定 yaml"""
    with open(calib_path, 'r') as f:
        calib = yaml.safe_load(f)

    translation = calib['transform']['translation']
    rotation = calib['transform']['rotation']

    t = np.array([translation['x'], translation['y'], translation['z']])
    R = quaternion_to_matrix(rotation['x'], rotation['y'], rotation['z'], rotation['w'])

    # frame_id -> child_frame_id 的外参, T_parent_child = T_vehicle_lidar
    frame_id = calib['header']['frame_id']
    child_frame_id = calib['child_frame_id']
    coord = calib.get('coordination', 'unknown')

    print(f"[Calib] {calib_path}")
    print(f"  parent frame:  {frame_id}")
    print(f"  child frame:   {child_frame_id}")
    print(f"  coordination:  {coord}")
    print(f"  translation:   [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
    print(f"  rotation (quat): [{rotation['x']:.6f}, {rotation['y']:.6f}, "
          f"{rotation['z']:.6f}, {rotation['w']:.6f}]")
    print(f"  T_parent_child (4x4):\n{R}\n{t}")

    # 构建 T_vehicle_lidar (parent -> child)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T, frame_id, child_frame_id


def draw_bev(lidar_points, show_range=50, resolution=1000, point_size=1,
             frame_label='lidar', coord='UNKNOWN'):
    """绘制 BEV 俯视图

    lidar_points: Nx3, 坐标系取决于 coord
      - coord=RFU: x=右, y=前, z=上
      - 其他: 按常规 x=前, y=左, z=上 处理

    BEV 画布: 原点=中心, 上=前方, 右=右侧
    """
    canvas_size = int(resolution)
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    canvas[:, :] = (40, 40, 40)  # 深灰背景

    if coord.upper() == 'RFU':
        # RFU: x=右, y=前, z=上
        vert_axis = lidar_points[:, 1]   # y=前 → 图像上方
        horz_axis = lidar_points[:, 0]   # x=右 → 图像右侧
        up_axis = lidar_points[:, 2]     # z=上
    else:
        # 默认: x=前, y=左, z=上
        vert_axis = lidar_points[:, 0]   # x=前 → 图像上方
        horz_axis = lidar_points[:, 1]   # y=左 → 需要反转到右侧
        up_axis = lidar_points[:, 2]

    # 过滤范围外的点
    mask = (np.abs(vert_axis) < show_range) & (np.abs(horz_axis) < show_range)
    vert = vert_axis[mask]
    horz = horz_axis[mask]

    if len(vert) == 0:
        return canvas

    # 映射到画布坐标 (原点在中心, 前=上, 右=右)
    scale = canvas_size / (2 * show_range)
    px = ((horz + show_range) * scale).astype(np.int32)   # 水平→图像x(右正)
    py = ((-vert + show_range) * scale).astype(np.int32)  # 前方→图像y(上=值小)

    # 裁剪到画布内
    valid = (px >= 0) & (px < canvas_size) & (py >= 0) & (py < canvas_size)
    px, py = px[valid], py[valid]

    # 按高度着色: 低→蓝, 高→红
    zs = up_axis[mask][valid]
    if len(zs) > 0:
        z_min, z_max = np.percentile(zs, [2, 98])  # 用百分位去噪
        if z_max > z_min:
            z_norm = np.clip((zs - z_min) / (z_max - z_min), 0, 1)
        else:
            z_norm = np.zeros_like(zs)
        colors = np.zeros((len(zs), 3), dtype=np.uint8)
        colors[:, 2] = (z_norm * 255).astype(np.uint8)              # R
        colors[:, 0] = ((1 - z_norm) * 255).astype(np.uint8)        # B
        colors[:, 1] = (np.abs(z_norm - 0.5) * 255).astype(np.uint8)  # G
    else:
        colors = np.tile([100, 200, 100], (len(px), 1))

    for i in range(len(px)):
        cv2.circle(canvas, (px[i], py[i]), point_size,
                   colors[i].tolist(), -1)

    # 画坐标系
    center = canvas_size // 2
    cv2.line(canvas, (0, center), (canvas_size, center), (80, 80, 80), 1)
    cv2.line(canvas, (center, 0), (center, canvas_size), (80, 80, 80), 1)

    # 画原点 (ego 位置)
    cv2.circle(canvas, (center, center), 5, (0, 255, 255), -1)
    # 画前向箭头
    cv2.arrowedLine(canvas, (center, center), (center, center - 30),
                    (0, 255, 255), 2, tipLength=0.3)

    # 标注坐标系信息
    cv2.putText(canvas, f'Frame: {frame_label} | {coord}', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # 画距离刻度环
    for r in [10, 20, 30, 40]:
        radius = int(r * scale)
        cv2.circle(canvas, (center, center), radius, (60, 60, 60), 1)
        cv2.putText(canvas, f'{r}m', (center + 3, center - radius + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

    return canvas


def transform_points(T, points_xyz):
    """将 Nx3 点云通过 4x4 变换矩阵转换"""
    N = points_xyz.shape[0]
    pts_h = np.hstack([points_xyz, np.ones((N, 1))])
    pts_t = (T @ pts_h.T).T
    return pts_t[:, :3]


def main():
    args = parse_args()

    if not os.path.exists(args.bin):
        raise FileNotFoundError(f"Bin file not found: {args.bin}")
    if not os.path.exists(args.calib):
        raise FileNotFoundError(f"Calib file not found: {args.calib}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- 1. 读取标定 ----
    T, parent_frame, child_frame = read_calib_yaml(args.calib)

    # ---- 2. 加载点云 ----
    lidar_points = _load_lidar_points(args.bin)
    xyz = lidar_points[:, :3]  # 只用 xyz
    print(f"\n[Points] {len(xyz)} points")
    print(f"  Raw:  x=[{xyz[:,0].min():.2f}, {xyz[:,0].max():.2f}] "
          f"y=[{xyz[:,1].min():.2f}, {xyz[:,1].max():.2f}] "
          f"z=[{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}]")

    # ---- 3. 转换到 vehicle 坐标系 ----
    xyz_vehicle = transform_points(T, xyz)
    print(f"  Vehicle: x=[{xyz_vehicle[:,0].min():.2f}, {xyz_vehicle[:,0].max():.2f}] "
          f"y=[{xyz_vehicle[:,1].min():.2f}, {xyz_vehicle[:,1].max():.2f}] "
          f"z=[{xyz_vehicle[:,2].min():.2f}, {xyz_vehicle[:,2].max():.2f}]")

    # ---- 4. 可视化 ----
    # 获取标定中声明的坐标系 (RFU 等)
    with open(args.calib, 'r') as f:
        calib_raw = yaml.safe_load(f)
    coord = calib_raw.get('coordination', 'UNKNOWN')

    bin_name = os.path.splitext(os.path.basename(args.bin))[0]

    if args.frame == 'lidar':
        bev = draw_bev(xyz, show_range=args.show_range,
                       point_size=args.point_size,
                       frame_label=child_frame, coord=coord)
        save_path = os.path.join(args.save_dir, f'{bin_name}_{child_frame}_bev.png')
    else:
        bev = draw_bev(xyz_vehicle, show_range=args.show_range,
                       point_size=args.point_size,
                       frame_label=parent_frame, coord=coord)
        save_path = os.path.join(args.save_dir, f'{bin_name}_{parent_frame}_bev.png')

    cv2.imwrite(save_path, bev)
    print(f"\nSaved: {save_path}")

    # ---- 5. 同时在 vehicle 坐标系下也画一张对比 ----
    if args.frame == 'lidar':
        bev_v = draw_bev(xyz_vehicle, show_range=args.show_range,
                         point_size=args.point_size,
                         frame_label=parent_frame, coord=coord)
        save_path_v = os.path.join(args.save_dir, f'{bin_name}_{parent_frame}_bev.png')
        cv2.imwrite(save_path_v, bev_v)
        print(f"Saved: {save_path_v}")

    print(f"\nDone. Results saved to {args.save_dir}/")


if __name__ == '__main__':
    main()
