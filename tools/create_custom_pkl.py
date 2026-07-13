#!/usr/bin/env python3
"""
修复版本：直接从标定文件读取外参
"""

import argparse
import os
import pickle
import sys
import yaml

import numpy as np
from pyquaternion import Quaternion


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate FastBEV-compatible pkl from custom dataset')
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='Root directory of the custom dataset')
    parser.add_argument(
        '--output',
        type=str,
        default='./custom_infos_val.pkl',
        help='Output pkl path')
    parser.add_argument(
        '--calib-root',
        type=str,
        default='./data/calibration',
        help='Root directory of calibration files')
    parser.add_argument(
        '--cam-names',
        type=str,
        nargs='+',
        default=['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        help='Camera names (nuScenes convention, 6 cameras)')
    return parser.parse_args()


def opencv_matrix_constructor(loader, node):
    """自定义构造器：解析 OpenCV 的 !!opencv-matrix 标签"""
    mapping = loader.construct_mapping(node)
    rows = mapping['rows']
    cols = mapping['cols']
    data = mapping.get('data', [])
    if not data:
        return np.zeros((rows, cols), dtype=np.float32)
    return np.array(data, dtype=np.float32).reshape(rows, cols)


def read_calibration_yaml(file_path):
    """读取标定YAML文件，处理 OpenCV 格式的 YAML 文件"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 移除 %YAML:1.0 声明，因为 PyYAML 不支持这种格式
    if content.startswith('%YAML'):
        idx = content.find('---')
        if idx != -1:
            content = content[idx+3:]
        else:
            lines = content.split('\n')[1:]
            content = '\n'.join(lines)
    
    # 使用正则表达式手动提取opencv-matrix数据
    import re
    
    # 先提取所有opencv-matrix块
    matrix_pattern = r'!!opencv-matrix\s*\n\s*rows:\s*(\d+)\s*\n\s*cols:\s*(\d+)\s*\n\s*dt:\s*(\w+)\s*\n\s*data:\s*\[(.*?)\]'
    
    def replace_matrix(match):
        rows = int(match.group(1))
        cols = int(match.group(2))
        data_str = match.group(4)
        # 解析数据
        data = [float(x.strip()) for x in data_str.split(',')]
        matrix = np.array(data, dtype=np.float32).reshape(rows, cols)
        # 转换为YAML列表格式
        lines = ['\n  - ' + ', '.join([str(v) for v in row]) for row in matrix]
        return '!!python/object/apply:numpy.core.multiarray._reconstruct\n  args: [numpy.ndarray]\n  state: [null, !binary "' + matrix.tobytes().hex() + '", null, null]'
    
    # 简单方法：直接用正则提取并替换为Python对象
    matrices = {}
    for i, match in enumerate(re.finditer(matrix_pattern, content, re.DOTALL)):
        rows = int(match.group(1))
        cols = int(match.group(2))
        data_str = match.group(4)
        data = [float(x.strip()) for x in data_str.split(',')]
        matrix = np.array(data, dtype=np.float32).reshape(rows, cols)
        matrices[i] = matrix
        # 替换为占位符
        content = content.replace(match.group(0), f'__MATRIX_PLACEHOLDER_{i}__', 1)
    
    # 解析修改后的YAML
    data = yaml.safe_load(content)
    
    # 递归替换占位符
    def replace_placeholders(obj):
        if isinstance(obj, dict):
            return {k: replace_placeholders(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_placeholders(v) for v in obj]
        elif isinstance(obj, str) and '__MATRIX_PLACEHOLDER_' in obj:
            # 提取索引数字
            import re
            match = re.search(r'__MATRIX_PLACEHOLDER_(\d+)__', obj)
            if match:
                idx = int(match.group(1))
                return matrices.get(idx, obj)
        return obj
    
    return replace_placeholders(data)


def rotmat_to_quat(R):
    """旋转矩阵转四元数 (w,x,y,z)"""
    trace = np.trace(R)
    if trace > 0:
        s = 2 * np.sqrt(trace + 1)
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([qw, qx, qy, qz])


def rfu_to_flu(R_rfu, t_rfu):
    """
    RFU (右前上) -> FLU (前左上) 坐标系转换（仅对 ego 端做坐标旋转，传感器自身坐标系不变）
    RFU: x=右, y=前, z=上
    FLU: x=前, y=左, z=上
    
    转换矩阵 T:
    [0,  1, 0]  # x_flu = y_rfu
    [-1, 0, 0]  # y_flu = -x_rfu
    [0,  0, 1]  # z_flu = z_rfu
    
    缩放: R_flu @ p_sensor_rfu = T @ (R_rfu @ p_sensor_rfu) = (T @ R_rfu) @ p_sensor_rfu
    所以: R_flu = T @ R_rfu, t_flu = T @ t_rfu (仅左乘，因为传感器输入端仍为 RFU)
    """
    T = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)
    R_flu = T @ R_rfu           # 传感器_RFU → ego_FLU
    t_flu = T @ t_rfu           # 传感器原点在 ego_FLU 中的位置
    return R_flu, t_flu


def build_single_frame(data_root, frame_id, cam_names, calib_root,
                       lidar_frame_id=None, cam_frame_map=None, 
                       lidar_fname=None, lidar_subdir='lidar'):
    """
    构建单帧的 info dict。从标定文件读取准确外参。
    """
    # ==================== 从标定文件读取 LiDAR 外参 ====================
    # 使用 yaml.safe_load 直接读取 (lidar 标定文件是标准 YAML，非 OpenCV 格式)
    lidar_calib_path = os.path.join(calib_root, 'lidar', 'helios32_top_vehicle_extrinsics.yaml')
    if not os.path.exists(lidar_calib_path):
        raise FileNotFoundError(f'LiDAR calibration not found: {lidar_calib_path}')
    
    with open(lidar_calib_path, 'r') as f:
        lidar_calib = yaml.safe_load(f)
    
    trans = lidar_calib['transform']['translation']
    rot = lidar_calib['transform']['rotation']
    
    # LiDAR 外参: 四元数 → 旋转矩阵 (scalar-last: x,y,z,w)
    lidar2ego_t_rfu = np.array([trans['x'], trans['y'], trans['z']], dtype=np.float32)
    lidar2ego_r_quat_rfu = np.array([rot['w'], rot['x'], rot['y'], rot['z']], dtype=np.float32)
    lidar2ego_r_mat_rfu = Quaternion(lidar2ego_r_quat_rfu).rotation_matrix
    
    # RFU → FLU: 对 ego 端左乘 T (传感器自身坐标系不变，同 project_validate_direct_calib_bak.py 方式)
    lidar2ego_r_mat, lidar2ego_t = rfu_to_flu(lidar2ego_r_mat_rfu, lidar2ego_t_rfu)
    lidar2ego_r = rotmat_to_quat(lidar2ego_r_mat)
    
    ego2global_t = [0.0, 0.0, 0.0]
    ego2global_r = [1.0, 0.0, 0.0, 0.0]

    # ==================== 从标定文件读取相机外参 ====================
    cam_configs = {}
    # 相机名称与实际安装位置的映射（修正版）
    # 根据实际安装位置调整：
    # - pillar相机实际安装在车辆后方
    # - repeater相机实际安装在车辆前方
    cam_name_map = {
        'CAM_FRONT': 'frontmain.yaml',
        'CAM_BACK': 'rear.yaml',
        'CAM_FRONT_LEFT': 'leftrepeater.yaml',   # 实际在左前，视角看左前
        'CAM_FRONT_RIGHT': 'rightrepeater.yaml', # 实际在右前，视角看右前
        'CAM_BACK_LEFT': 'leftpillar.yaml',      # 实际在左后，视角看左后
        'CAM_BACK_RIGHT': 'rightpillar.yaml',    # 实际在右后，视角看右后
    }
    
    for cam_name in cam_names:
        calib_fname = cam_name_map.get(cam_name)
        if not calib_fname:
            raise ValueError(f'Unknown camera: {cam_name}')
        
        calib_path = os.path.join(calib_root, 'camera', calib_fname)
        if not os.path.exists(calib_path):
            raise FileNotFoundError(f'Camera calibration not found: {calib_path}')
        
        calib = read_calibration_yaml(calib_path)
        
        # 内参
        M = calib['M'].astype(np.float32)
        
        # 外参矩阵 (camera -> vehicle, RFU)
        Tr = calib['Tr_cam_to_vehicle'].astype(np.float32)
        R_rfu = Tr[:3, :3]
        t_rfu = Tr[:3, 3]
        
        # 转换到 FLU 坐标系
        R_flu, t_flu = rfu_to_flu(R_rfu, t_rfu)
        quat_flu = rotmat_to_quat(R_flu)
        
        cam_configs[cam_name] = {
            'cam_intrinsic': M,
            'sensor2ego_translation': t_flu.tolist(),
            'sensor2ego_rotation': quat_flu.tolist(),
            'ego2global_translation': [0.0, 0.0, 0.0],
            'ego2global_rotation': [1.0, 0.0, 0.0, 0.0],
            'width': calib['width'],
            'height': calib['height'],
        }

    # ==================== 填写路径 ====================
    if lidar_fname is not None:
        lidar_path = f'{lidar_subdir}/{lidar_fname}'
    else:
        if lidar_frame_id is None:
            lidar_frame_id = frame_id
        lidar_path = f'{lidar_subdir}/{lidar_frame_id}.bin'
    
    if cam_frame_map is not None:
        img_paths = {cam: f'{cam}/{fname}' for cam, fname in cam_frame_map.items()}
    else:
        img_paths = {cam: f'{cam}/{frame_id}.jpg' for cam in cam_names}

    # ==================== GT 框 (可选) ====================
    gt_boxes = np.array([
        # [x, y, z, w, l, h, yaw] in FLU coordinates
        [10.0, 2.0, 0.0, 1.8, 4.5, 1.5, 0.0],
        [15.0, -1.0, 0.0, 0.8, 0.8, 1.7, 0.0],
    ], dtype=np.float32)
    
    class_to_idx = {
        'car': 0, 'truck': 1, 'construction_vehicle': 2, 'bus': 3, 'trailer': 4,
        'barrier': 5, 'motorcycle': 6, 'bicycle': 7, 'pedestrian': 8, 'traffic_cone': 9,
    }
    gt_names_str = ['car', 'pedestrian']
    gt_names = np.array([class_to_idx[n] for n in gt_names_str], dtype=np.int64)
    gt_velocity = np.zeros((len(gt_boxes), 2), dtype=np.float32)
    valid_flag = np.ones(len(gt_boxes), dtype=bool)

    # ==================== 自动计算 sensor2lidar ====================
    # 同 project_validate_direct_calib_bak.py 验证通过的坐标系变换链路:
    # sensor2lidar = ego→lidar @ sensor→ego (FLU 下, 传感器输入端仍为 RFU)
    #   R_sl = R_le^T @ R_se = (T @ R_rfu)^T @ (T @ R_cam_rfu)
    #        = R_rfu^T @ T^T @ T @ R_cam_rfu = R_rfu^T @ R_cam_rfu
    #   t_sl = R_le^T @ (t_se - t_le) = R_rfu^T @ (t_cam_rfu - t_rfu)
    l2e_r_mat = Quaternion(lidar2ego_r).rotation_matrix
    e2g_r_mat = Quaternion(ego2global_r).rotation_matrix

    cams = {}
    for cam_name in cam_names:
        cfg = cam_configs[cam_name]
        s2e_r_mat = Quaternion(cfg['sensor2ego_rotation']).rotation_matrix
        s2e_t = np.array(cfg['sensor2ego_translation'], dtype=np.float32)
        e2g_r_s_mat = Quaternion(cfg['ego2global_rotation']).rotation_matrix
        e2g_t_s = np.array(cfg['ego2global_translation'], dtype=np.float32)

        # Compute sensor->lidar transform:
        # p_ego = R_se @ p_sensor + t_se   (sensor→ego, FLU)
        # p_lidar = R_le^T @ (p_ego - t_le) = R_le^T @ R_se @ p_sensor + R_le^T @ (t_se - t_le)
        R_le = l2e_r_mat   # lidar→ego_FLU
        R_se = s2e_r_mat   # camera→ego_FLU
        t_le = np.array(lidar2ego_t, dtype=np.float32)
        t_se = s2e_t

        sensor2lidar_rotation = R_le.T @ R_se
        sensor2lidar_translation = R_le.T @ (t_se - t_le)

        data_path = os.path.join(data_root, img_paths[cam_name])
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f'Image file not found: {data_path}')
        
        cams[cam_name] = {
            'data_path': data_path,
            'sensor2ego_translation': cfg['sensor2ego_translation'],
            'sensor2ego_rotation': cfg['sensor2ego_rotation'],
            'ego2global_translation': cfg['ego2global_translation'],
            'ego2global_rotation': cfg['ego2global_rotation'],
            'cam_intrinsic': cfg['cam_intrinsic'],
            'sensor2lidar_rotation': sensor2lidar_rotation,
            'sensor2lidar_translation': sensor2lidar_translation,
        }

    # 构造 ann_infos
    if len(gt_boxes) > 0:
        gt_boxes_9d = np.concatenate([gt_boxes, gt_velocity], axis=1)
    else:
        gt_boxes_9d = np.zeros((0, 9), dtype=np.float32)
    ann_infos = (gt_boxes_9d, gt_names)

    lidar_full_path = os.path.join(data_root, lidar_path)
    if not os.path.isfile(lidar_full_path):
        raise FileNotFoundError(f'LiDAR file not found: {lidar_full_path}')

    info = {
        'token': frame_id,
        'lidar_path': lidar_full_path,
        'timestamp': 0,
        'lidar2ego_translation': lidar2ego_t.tolist(),
        'lidar2ego_rotation': lidar2ego_r.tolist(),
        'ego2global_translation': ego2global_t,
        'ego2global_rotation': ego2global_r,
        'cams': cams,
        'sweeps': [],
        'ann_infos': ann_infos,
        'gt_boxes': gt_boxes,
        'gt_names': gt_names_str,
        'gt_velocity': gt_velocity,
        'valid_flag': valid_flag,
        'scene_token': 'custom_scene',
    }

    return info


def _find_nearest_cam_frame(lidar_ts: int, cam_ts_map: dict, max_diff_ms: int = 100) -> str:
    best_ts = None
    best_diff = max_diff_ms + 1
    for cam_name, ts_list in cam_ts_map.items():
        for ts in ts_list:
            diff = abs(ts - lidar_ts)
            if diff < best_diff:
                best_diff = diff
                best_ts = ts
    if best_ts is not None and best_diff <= max_diff_ms:
        return str(best_ts)
    return ''


def main():
    args = parse_args()
    data_root = args.data_root
    calib_root = args.calib_root
    cam_names = args.cam_names

    # 探测 lidar 目录
    lidar_abs_dir = None
    lidar_rel_dir = None
    lidar_candidates = ['lidar', 'LIDAR_TOP']
    for cand in lidar_candidates:
        cand_path = os.path.join(data_root, cand)
        if os.path.isdir(cand_path):
            lidar_abs_dir = cand_path
            lidar_rel_dir = cand
            break
    
    if lidar_abs_dir is None:
        print(f'ERROR: no lidar directory found in {data_root}')
        sys.exit(1)

    lidar_fnames = {}
    for f in os.listdir(lidar_abs_dir):
        stem, ext = os.path.splitext(f)
        if ext.lower() in ('.bin', '.pcd', '.ply'):
            lidar_fnames[stem] = f
    
    frame_ids = sorted(lidar_fnames.keys())
    if not frame_ids:
        print(f'ERROR: no LiDAR files found in {lidar_abs_dir}')
        sys.exit(1)
    
    print(f'Found {len(frame_ids)} LiDAR frames')

    # 扫描相机目录
    cam_ts_map = {}
    cam_fname_map = {}
    for cam in cam_names:
        cam_dir = os.path.join(data_root, cam)
        if not os.path.isdir(cam_dir):
            print(f'WARNING: camera dir not found: {cam_dir}')
            continue
        ts_list = []
        fname_map = {}
        for f in sorted(os.listdir(cam_dir)):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                try:
                    ts = int(f.rsplit('.', 1)[0])
                    ts_list.append(ts)
                    fname_map[ts] = f
                except ValueError:
                    continue
        cam_ts_map[cam] = ts_list
        cam_fname_map[cam] = fname_map
        print(f'  Camera {cam}: {len(ts_list)} images')

    cam_is_ts = any(len(cam_ts_map.get(c, [])) > 0 for c in cam_names)

    infos = []
    skipped = 0
    for lidar_idx, lidar_id in enumerate(frame_ids):
        print(f'  Building frame: {lidar_id}', end='')
        try:
            lidar_ts = int(lidar_id)
        except ValueError:
            lidar_ts = None

        lidar_fname = lidar_fnames.get(lidar_id)
        
        if lidar_ts is not None:
            cam_frame_map = {}
            valid = True
            for cam in cam_names:
                ts_list = sorted(cam_fname_map.get(cam, {}).keys())
                if not ts_list:
                    valid = False
                    break
                best_ts = min(ts_list, key=lambda t: abs(t - lidar_ts))
                if abs(best_ts - lidar_ts) > 100:
                    valid = False
                    break
                cam_frame_map[cam] = cam_fname_map[cam][best_ts]
            if valid:
                first_cam = cam_names[0]
                cam_frame_id = cam_frame_map[first_cam].rsplit('.', 1)[0]
                print(f' -> camera id={cam_frame_id}')
                info = build_single_frame(data_root, cam_frame_id, cam_names, calib_root,
                                          lidar_frame_id=lidar_id, lidar_fname=lidar_fname,
                                          lidar_subdir=lidar_rel_dir, cam_frame_map=cam_frame_map)
            else:
                print(f' -> SKIP (no matching camera frame)')
                skipped += 1
                continue
        elif cam_is_ts:
            cam_frame_map = {}
            valid = True
            for cam in cam_names:
                ts_list = sorted(cam_fname_map.get(cam, {}).keys())
                if lidar_idx >= len(ts_list):
                    valid = False
                    break
                ts = ts_list[lidar_idx]
                cam_frame_map[cam] = cam_fname_map[cam][ts]
            if not valid:
                skipped += 1
                continue
            first_cam = cam_names[0]
            cam_frame_id = cam_frame_map[first_cam].rsplit('.', 1)[0]
            print(f' -> camera id={cam_frame_id} (sequential)')
            info = build_single_frame(data_root, cam_frame_id, cam_names, calib_root,
                                      lidar_frame_id=lidar_id, cam_frame_map=cam_frame_map,
                                      lidar_fname=lidar_fname, lidar_subdir=lidar_rel_dir)
        else:
            print()
            info = build_single_frame(data_root, lidar_id, cam_names, calib_root,
                                      lidar_fname=lidar_fname, lidar_subdir=lidar_rel_dir)
        infos.append(info)

    if skipped > 0:
        print(f'Skipped {skipped} frames')

    dataset = {
        'infos': infos,
        'metadata': {
            'version': 'custom',
            'cam_names': cam_names,
        },
    }

    with open(args.output, 'wb') as f:
        pickle.dump(dataset, f)

    print(f'Saved {len(infos)} frames to {args.output}')


if __name__ == '__main__':
    main()