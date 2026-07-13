#!/usr/bin/env python3
"""
诊断自定义 pkl 的标定是否正确。
检查项:
  1. 每个相机在 lidar 坐标系下的前向方向
  2. lidar2ego 旋转是否为单位阵/合理
  3. 将前向 5m/10m 处的虚拟点投影到各相机，验证图像位置

用法:
    python tools/check_custom_calibration.py \
        --pkl custom_infos_val.pkl \
        --frame 0
"""

import argparse
import os
import pickle

import numpy as np
from pyquaternion.quaternion import Quaternion


def parse_args():
    parser = argparse.ArgumentParser(description='Check custom pkl calibration')
    parser.add_argument('--pkl', type=str, required=True, help='Path to pkl file')
    parser.add_argument('--frame', type=int, default=0, help='Frame index to check')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset = pickle.load(open(args.pkl, 'rb'))
    infos = dataset['infos'][args.frame]

    cam_names = sorted(infos['cams'].keys())
    print(f'Frame token: {infos["token"]}')
    print(f'Cameras: {cam_names}\n')

    # 1. 检查 lidar2ego
    lidar2ego_t = np.array(infos['lidar2ego_translation'], dtype=np.float32)
    lidar2ego_r = Quaternion(infos['lidar2ego_rotation']).rotation_matrix
    ego2global_r = Quaternion(infos['ego2global_rotation']).rotation_matrix
    print('LiDAR -> ego:')
    print(f'  translation: {lidar2ego_t}')
    print(f'  rotation matrix:\n{lidar2ego_r}')
    print(f'  ego -> global rotation:\n{ego2global_r}')
    print()

    # 2. 检查每个相机前向在 lidar 下的方向
    print('Camera forward direction in LiDAR coordinates (should be ~):')
    print('  CAM_FRONT:       [ 1.00,  0.00,  0.00]')
    print('  CAM_BACK:        [-1.00,  0.00,  0.00]')
    print('  CAM_FRONT_LEFT:  [ 0.70,  0.70,  0.00]')
    print('  CAM_BACK_LEFT:   [-0.70,  0.70,  0.00]')
    print('  CAM_FRONT_RIGHT: [ 0.70, -0.70,  0.00]')
    print('  CAM_BACK_RIGHT:  [-0.70, -0.70,  0.00]')
    print()
    for cam_name in cam_names:
        cam = infos['cams'][cam_name]
        R = np.array(cam['sensor2lidar_rotation'], dtype=np.float32)
        forward_cam = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        forward_lidar = forward_cam @ R.T
        forward_lidar /= np.linalg.norm(forward_lidar)
        print(f'  {cam_name}: [{forward_lidar[0]:+.2f}, '
              f'{forward_lidar[1]:+.2f}, {forward_lidar[2]:+.2f}]')
    print()

    # 3. 将虚拟点云投影到每个相机 (5m/10m 正前方、左右)
    print('Projecting reference 3D points (in LiDAR coords) to camera views:')
    points = np.array([
        [5, 0, 0],   # 正前 5m
        [10, 0, 0],  # 正前 10m
        [5, 2, 0],   # 左前 5m
        [5, -2, 0],  # 右前 5m
    ], dtype=np.float32)
    print(f'  reference points (x=fwd, y=left, z=up):\n{points}')

    for cam_name in cam_names:
        cam = infos['cams'][cam_name]
        K = np.array(cam['cam_intrinsic'], dtype=np.float32)
        R = np.array(cam['sensor2lidar_rotation'], dtype=np.float32)
        t = np.array(cam['sensor2lidar_translation'], dtype=np.float32)

        # lidar -> camera
        pts_cam = points @ R + t
        pts_img = pts_cam[:, :2] / pts_cam[:, 2:3]
        pts_img = pts_img @ K[:2, :2].T + K[:2, 2]
        valid = pts_cam[:, 2] > 0.5
        print(f'  {cam_name}:')
        for i, (pt, is_valid) in enumerate(zip(pts_img, valid)):
            status = 'valid' if is_valid else 'behind camera'
            print(f'    pt{i} {points[i]} -> img ({pt[0]:.1f}, {pt[1]:.1f}) [{status}]')

    print('\nCheck:')
    print('  - CAM_FRONT should project the forward points near image center')
    print('  - Left/right cameras should project to their respective sides')
    print('  - If directions are 90° off, the sensor2lidar_rotation is wrong')


if __name__ == '__main__':
    main()
