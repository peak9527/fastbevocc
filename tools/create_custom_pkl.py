#!/usr/bin/env python3
"""
修复版本：直接从标定文件读取外参，并从 JSON 标注文件读取 GT 框
适配 SUSTechPOINTS 实际数据格式:
  - camera/<物理相机名>/<timestamp>.jpg
  - calib/camera/<物理相机名>.json  (intrinsic[9], extrinsic[16])
  - label/<timestamp>.json          (含 labels 数组)
  - lidar/<timestamp>.bin
"""

import argparse
import json
import os
import pickle
import sys
import yaml

import numpy as np
from pyquaternion import Quaternion

# 全局 LiDAR 外参路径（同一传感器，所有序列共用）
DEFAULT_LIDAR_CALIB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'calibration', 'lidar', 'helios32_top_vehicle_extrinsics.yaml')

# nuScenes 名 → 物理相机名 + calib 文件名 (默认映射，可被 --cam-map 覆盖)
CAM_PHYSICAL_MAP = {
    'CAM_FRONT':        'frontmain',
    'CAM_BACK':         'rear',
    'CAM_FRONT_LEFT':   'leftrepeater',
    'CAM_FRONT_RIGHT':  'rightrepeater',
    'CAM_BACK_LEFT':    'leftpillar',
    'CAM_BACK_RIGHT':   'rightpillar',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate FastBEV-compatible pkl from custom dataset')
    parser.add_argument(
        '--data-root',
        type=str,
        default=None,
        help='Root directory of a single-sequence dataset (legacy mode)')
    parser.add_argument(
        '--output',
        type=str,
        default='./custom_infos_val.pkl',
        help='Output pkl path')
    parser.add_argument(
        '--calib-dir',
        type=str,
        default=None,
        help='Camera calibration dir (default: {data-root}/calib/camera). Files: frontmain.json etc.')
    parser.add_argument(
        '--lidar-calib',
        type=str,
        default=DEFAULT_LIDAR_CALIB_PATH,
        help='Path to lidar extrinsics YAML')
    parser.add_argument(
        '--label-root',
        type=str,
        default=None,
        help='Root directory of label JSON files (default: {data-root}/label)')
    parser.add_argument(
        '--coord',
        type=str,
        default='flu',
        choices=['flu', 'rfu'],
        help='Coordinate system of label JSON files: flu (x=前) or rfu (x=右, y=前)')
    parser.add_argument(
        '--calib-coord',
        type=str,
        default='rfu',
        choices=['rfu', 'flu'],
        help='Coordinate system of input calibration files (extrinsic):\n'
             '  rfu (default): calib uses x=右, y=前, z=上 → auto-convert to FLU\n'
             '  flu: calib already uses x=前, y=左, z=上 → no conversion needed')
    parser.add_argument(
        '--cam-names',
        type=str,
        nargs='+',
        default=['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        help='Camera names (nuScenes convention, 6 cameras)')
    parser.add_argument(
        '--cam-dir-name',
        type=str,
        default='image',
        help='Top-level camera directory name inside data-root (default: image)')

    # === 多序列模式参数 ===
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Multi-sequence: root directory containing sequence*/image/lidar subdirs')
    parser.add_argument(
        '--label-dir',
        type=str,
        default=None,
        help='Multi-sequence: root directory containing json/<record>/<seq>/label/box_label')
    parser.add_argument(
        '--seq-list',
        type=str,
        nargs='*',
        default=None,
        help='Multi-sequence: list of sequences to process (optional).\n'
             '  Supports "record/sequence" format (e.g. record_A/sequence0).\n'
             '  If omitted, auto-scans --data-dir for all record/sequence combos '
             'that contain image/ and lidar/<lidar-name>/ subdirs.')
    parser.add_argument(
        '--lidar-name',
        type=str,
        default='bin_no_self_point',
        help='Multi-sequence: which lidar subdir to use under <seq>/lidar/ (default: bin_no_self_point)')
    parser.add_argument(
        '--cam-map',
        type=str,
        default=None,
        help='Multi-sequence: JSON file mapping {sequence: {CAM_NAME: physical_cam_dir}}')
    parser.add_argument(
        '--label-subdir',
        type=str,
        default='box_label',
        help='Multi-sequence: subdir name under <seq>/label/ (default: box_label)')
    parser.add_argument(
        '--record-map',
        type=str,
        default=None,
        help='Multi-sequence: JSON file mapping {sequence: record_name_under_data_dir}. '
             'Only needed if --seq-list uses plain sequence names and auto-scan fails. '
             'For "record/sequence" format, this is ignored.')
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


def read_calibration_json(file_path):
    """读取 SUSTechPOINTS JSON 标定文件。

    JSON 格式:
    {
      "intrinsic": [fx,0,cx, 0,fy,cy, 0,0,1],  // 3x3 row-major
      "distortion": [k1,k2,p1,p2,k3],
      "extrinsic": [R11,R12,R13,tx, R21,R22,R23,ty, R31,R32,R33,tz, 0,0,0,1]  // 4x4 row-major
    }
    返回:
      M:      3x3 内参矩阵
      R:      3x3 旋转矩阵 (camera→vehicle, RFU)
      t:      3x1 平移向量 (camera→vehicle, RFU)
      width:  图像宽度 (= cx*2)
      height: 图像高度 (= cy*2)
    """
    with open(file_path, 'r') as f:
        calib = json.load(f)

    intrinsic = calib['intrinsic']  # list of 9
    M = np.array(intrinsic, dtype=np.float32).reshape(3, 3)

    extrinsic = calib['extrinsic']  # list of 16
    Tr = np.array(extrinsic, dtype=np.float32).reshape(4, 4)
    R_rfu = Tr[:3, :3]
    t_rfu = Tr[:3, 3]

    # 从内参推断图像尺寸
    width = int(round(M[0, 2] * 2))
    height = int(round(M[1, 2] * 2))

    return M, R_rfu, t_rfu, width, height


def read_calibration_yaml_camera(file_path):
    """读取 OpenCV YAML 格式的相机标定文件。

    YAML 格式:
    %YAML:1.0
    ---
    M: !!opencv-matrix  # 3x3 内参
    Tr_cam_to_vehicle: !!opencv-matrix  # 4x4 外参
    width: 3840
    height: 2160
    coordination: rfu

    返回:
      M:      3x3 内参矩阵
      R:      3x3 旋转矩阵 (camera→vehicle, RFU)
      t:      3x1 平移向量
      width:  图像宽度
      height: 图像高度
    """
    data = read_calibration_yaml(file_path)

    M = data['M']  # 3x3 ndarray
    Tr = data['Tr_cam_to_vehicle']  # 4x4 ndarray
    R_rfu = Tr[:3, :3]
    t_rfu = Tr[:3, 3]

    width = int(data.get('width', int(round(M[0, 2] * 2))))
    height = int(data.get('height', int(round(M[1, 2] * 2))))

    return M, R_rfu, t_rfu, width, height


def _find_camera_calib_file(calib_dir, physical_name):
    """在 calib_dir 下查找指定相机的标定文件。

    依次尝试:
      1) <calib_dir>/camera/<name>.yaml
      2) <calib_dir>/camera/<name>.json
      3) <calib_dir>/<name>.yaml
      4) <calib_dir>/<name>.json
    返回 (file_path, file_type) 或 raise FileNotFoundError
    """
    candidates = [
        os.path.join(calib_dir, 'camera', f'{physical_name}.yaml'),
        os.path.join(calib_dir, 'camera', f'{physical_name}.json'),
        os.path.join(calib_dir, f'{physical_name}.yaml'),
        os.path.join(calib_dir, f'{physical_name}.json'),
    ]
    for path in candidates:
        if os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            return path, ext
    tried = '\n    '.join(candidates)
    raise FileNotFoundError(
        f'Camera calibration not found for "{physical_name}". Tried:\n    {tried}')


def _read_camera_calib(file_path, file_ext):
    """根据文件扩展名调用对应的标定解析函数。"""
    if file_ext == '.yaml':
        return read_calibration_yaml_camera(file_path)
    elif file_ext == '.json':
        return read_calibration_json(file_path)
    else:
        raise ValueError(f'Unsupported calibration file format: {file_ext}')


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


def load_json_labels(label_dir, coord='flu',
                      class_mapping=None):
    """
    读取 JSON 标注文件，转为 FastBEV gt_boxes 格式。
    自动识别两种格式：

    格式A (旧 SUSTechPOINTS):
      [{ "obj_type": "Car", "obj_id": "8",
         "psr": { "position": {"x":..., "y":..., "z":...},
                  "rotation": {"x":0, "y":0, "z":yaw},
                  "scale":    {"x":w, "y":l, "z":h} }}]

    格式B (新格式):
      { "name": "xxx.bin", "info": {...},
        "labels": [{ "category": "car",
                     "location": [x, y, z],      // [x, y, z]
                     "size":     [l, w, h],       // [length, width, height]
                     "rotation": [rx, ry, rz],    // z=yaw
                     "attribute": "..." }] }

    输出:
      gt_boxes:  np.ndarray (N, 7)  [x, y, z, w, l, h, yaw] in FLU 坐标系
      gt_names:  list[str]          类别名称列表

    Args:
        label_dir: JSON 标注文件所在目录
        coord:     'flu' 或 'rfu'，JSON 中坐标的坐标系
        class_mapping: None=使用默认映射，或自定义 dict {json_type: nuscenes_class}
    """
    if class_mapping is None:
        class_mapping = {
            # 旧格式 obj_type
            'Car': 'car', 'car': 'car',
            'Truck': 'truck', 'truck': 'truck',
            'Bus': 'bus', 'bus': 'bus',
            'Trailer': 'trailer', 'trailer': 'trailer',
            'ConstructionVehicle': 'construction_vehicle',
            'Motor': 'motorcycle', 'motor': 'motorcycle',
            'Motorcycle': 'motorcycle', 'motorcycle': 'motorcycle',
            'Bicycle': 'bicycle', 'bicycle': 'bicycle',
            'Pedestrian': 'pedestrian', 'pedestrian': 'pedestrian',
            'Rider': 'motorcycle', 'rider': 'motorcycle',
            'TrafficCone': 'traffic_cone', 'traffic_cone': 'traffic_cone',
            'Barrier': 'barrier', 'barrier': 'barrier',
            # 新格式 category（自定义类别 → nuScenes 标准类别）
            'cyclist': 'bicycle',
            'cone': 'traffic_cone',
        }

    # 合法类别（不在其中的会被跳过）
    VALID_CLASSES = {
        'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone', 'barrier',
    }

    frame_labels = {}
    all_types = set()
    fmt_detected = None

    if not os.path.isdir(label_dir):
        print(f'WARNING: label directory not found: {label_dir}')
        return frame_labels, all_types

    for fname in sorted(os.listdir(label_dir)):
        if not fname.endswith('.json'):
            continue
        frame_id = fname.replace('.json', '')
        filepath = os.path.join(label_dir, fname)

        with open(filepath, 'r') as f:
            raw_data = json.load(f)

        boxes = []
        names = []

        # ---- 自动检测格式 ----
        if isinstance(raw_data, list):
            # 格式A：根是数组
            if fmt_detected is None:
                fmt_detected = 'A'
            raw_labels = raw_data

            for obj in raw_labels:
                obj_type = obj.get('obj_type', '')
                mapped_type = class_mapping.get(obj_type, obj_type.lower())
                if mapped_type is None or mapped_type not in VALID_CLASSES:
                    continue
                all_types.add(obj_type)

                psr = obj['psr']
                x, y, z = psr['position']['x'], psr['position']['y'], psr['position']['z']
                w, l, h = psr['scale']['x'], psr['scale']['y'], psr['scale']['z']
                yaw = psr['rotation']['z']
                # A 格式下 scale: {x:w, y:l, z:h}

                if coord == 'rfu':
                    x, y = y, -x
                    yaw = yaw - np.pi / 2
                    w, l = l, w

                # gt_boxes 使用 [l, w, h] 顺序 (nuScenes 约定)，
                # 与 NuScenesDataset.get_ann_info() 中
                # LiDARInstance3DBoxes(x_size=l, y_size=w) 的期望一致
                boxes.append([x, y, z, l, w, h, yaw])
                names.append(mapped_type)

        elif isinstance(raw_data, dict) and 'labels' in raw_data:
            # 格式B：根是对象，包含 "labels" 数组
            if fmt_detected is None:
                fmt_detected = 'B'
            raw_labels = raw_data['labels']

            for obj in raw_labels:
                obj_type = obj.get('category', '')
                mapped_type = class_mapping.get(obj_type, obj_type.lower())
                if mapped_type is None or mapped_type not in VALID_CLASSES:
                    continue
                all_types.add(obj_type)

                loc = obj['location']        # [x, y, z]
                size = obj['size']           # [l, w, h]
                rot = obj['rotation']        # [rx, ry, rz]

                x, y, z = loc[0], loc[1], loc[2]
                l, w, h = size[0], size[1], size[2]   # 新格式: size=[l, w, h]
                yaw = rot[2]                            # yaw = rotation[2]

                if coord == 'rfu':
                    x, y = y, -x
                    yaw = yaw - np.pi / 2
                    w, l = l, w

                # gt_boxes 使用 [l, w, h] 顺序 (nuScenes 约定)
                boxes.append([x, y, z, l, w, h, yaw])
                names.append(mapped_type)

        else:
            print(f'WARNING: unknown JSON format in {fname}, skipping')
            continue

        if boxes:
            frame_labels[frame_id] = (
                np.array(boxes, dtype=np.float32),
                names,
            )

    fmt_name = {'A': 'old (psr)', 'B': 'new (labels[])'}.get(fmt_detected, 'unknown')
    print(f'Loaded labels from {label_dir}: {len(frame_labels)} frames '
          f'(format={fmt_name}), types: {sorted(all_types)}')
    return frame_labels, all_types


def build_single_frame(data_root, frame_id, cam_names, calib_dir,
                       lidar_calib_path, lidar_frame_id=None,
                       cam_frame_map=None, lidar_fname=None,
                       lidar_subdir='lidar', cam_dir_name='camera',
                       cam_name_map=None, labels=None, scene_token=None,
                       token_prefix='', calib_coord='rfu'):
    """
    构建单帧的 info dict。从标定文件读取准确外参。

    Args:
        calib_dir:         相机标定 JSON 所在目录
        lidar_calib_path:  LiDAR 外参 YAML 路径
        cam_dir_name:      相机根目录名 (data_root/camera/)
        cam_name_map:      dict {pkl_key: physical_dir}, None 则使用 CAM_PHYSICAL_MAP
        labels:            (gt_boxes_np7, gt_names_list) 或 None
        scene_token:       序列 token, 多序列时使用
        token_prefix:      帧 token 前缀, 多序列时防止冲突
        calib_coord:       标定文件坐标系 'rfu' (默认,自动转FLU) 或 'flu' (已FLU,不转换)
    """
    need_convert = (calib_coord == 'rfu')

    # ==================== LiDAR 外参 (固定, 全局 YAML) ====================
    if not os.path.exists(lidar_calib_path):
        raise FileNotFoundError(f'LiDAR calibration not found: {lidar_calib_path}')
    lidar_calib = read_calibration_yaml(lidar_calib_path)

    trans = lidar_calib['transform']['translation']
    rot = lidar_calib['transform']['rotation']
    lidar2ego_t_input = np.array([trans['x'], trans['y'], trans['z']], dtype=np.float32)
    lidar2ego_r_quat_input = np.array([rot['w'], rot['x'], rot['y'], rot['z']], dtype=np.float32)
    lidar2ego_r_mat_input = Quaternion(lidar2ego_r_quat_input).rotation_matrix

    if need_convert:
        # RFU → FLU
        lidar2ego_r_mat, lidar2ego_t = rfu_to_flu(lidar2ego_r_mat_input, lidar2ego_t_input)
    else:
        lidar2ego_r_mat, lidar2ego_t = lidar2ego_r_mat_input, lidar2ego_t_input
    lidar2ego_r = rotmat_to_quat(lidar2ego_r_mat)

    ego2global_t = [0.0, 0.0, 0.0]
    ego2global_r = [1.0, 0.0, 0.0, 0.0]

    # ==================== 相机外参 (YAML / JSON 标定) ====================
    cam_configs = {}
    _map = cam_name_map if cam_name_map is not None else CAM_PHYSICAL_MAP
    for cam_name in cam_names:
        physical = _map.get(cam_name)
        if not physical:
            raise ValueError(f'Unknown camera: {cam_name}')

        calib_path, calib_ext = _find_camera_calib_file(calib_dir, physical)
        M, R_input, t_input, width, height = _read_camera_calib(calib_path, calib_ext)

        if need_convert:
            # RFU → FLU
            R_flu, t_flu = rfu_to_flu(R_input, t_input)
        else:
            R_flu, t_flu = R_input, t_input
        quat_flu = rotmat_to_quat(R_flu)

        cam_configs[cam_name] = {
            'cam_intrinsic': M,
            'sensor2ego_translation': t_flu.tolist(),
            'sensor2ego_rotation': quat_flu.tolist(),
            'ego2global_translation': [0.0, 0.0, 0.0],
            'ego2global_rotation': [1.0, 0.0, 0.0, 0.0],
            'width': width,
            'height': height,
        }

    # ==================== 填写路径 ====================
    # LiDAR 路径: lidar/<id>.bin
    if lidar_fname is not None:
        lidar_path = f'{lidar_subdir}/{lidar_fname}'
    else:
        if lidar_frame_id is None:
            lidar_frame_id = frame_id
        lidar_path = f'{lidar_subdir}/{lidar_frame_id}.bin'

    # 相机路径: camera/<physical>/<timestamp>.jpg
    if cam_frame_map is not None:
        img_paths = {}
        for cam, fname in cam_frame_map.items():
            physical = _map.get(cam, cam)
            img_paths[cam] = f'{cam_dir_name}/{physical}/{fname}'
    else:
        img_paths = {}
        for cam in cam_names:
            physical = _map.get(cam, cam)
            img_paths[cam] = f'{cam_dir_name}/{physical}/{frame_id}.jpg'

    # ==================== GT 框（从 JSON 标注读取，或留空） ====================
    # labels 参数: (gt_boxes, gt_names_str) 或 None
    if labels is not None:
        gt_boxes, gt_names_str = labels
    else:
        gt_boxes = np.zeros((0, 7), dtype=np.float32)
        gt_names_str = []

    # cyclist/cone 已在 load_json_labels 中通过 class_mapping 映射为
    # bicycle/traffic_cone，gt_names_str 直接是标准 nuScenes 类别名
    class_to_idx = {
        'car': 0, 'truck': 1, 'construction_vehicle': 2, 'bus': 3, 'trailer': 4,
        'barrier': 5, 'motorcycle': 6, 'bicycle': 7, 'pedestrian': 8, 'traffic_cone': 9,
    }
    gt_names = np.array([class_to_idx.get(n, -1) for n in gt_names_str], dtype=np.int64)
    # 过滤掉未知类别
    if len(gt_names) > 0 and (gt_names == -1).any():
        valid_mask = gt_names != -1
        gt_boxes = gt_boxes[valid_mask]
        gt_names = gt_names[valid_mask]
        gt_names_str = [n for i, n in enumerate(gt_names_str) if valid_mask[i]]
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
        'token': f'{token_prefix}{frame_id}',
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
        'scene_token': scene_token if scene_token is not None else 'custom_scene',
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


def _find_label_dir_for_seq(data_root_seq, label_dir, label_subdir='box_label',
                            data_dir=None):
    """在多序列标注目录中查找 label 目录。

    支持结构：
      1. <data_root_seq>/label/<label_subdir>               (label 在序列内部)
      2. <label_dir>/<seq>/label/<label_subdir>             (label 在独立目录)
      3. <label_dir>/<record>/<seq>/label/<label_subdir>   (带 record 层)
      4. <label_dir>/<rel_path_from_data_dir>/label/<label_subdir>  (任意层级镜像)
      5. 递归搜索 <label_dir> 下任意 record 子目录中的 <seq>/label/<label_subdir>
    """
    # 优先查找序列内部的 label 目录
    internal = os.path.join(data_root_seq, 'label', label_subdir)
    if os.path.isdir(internal):
        return internal

    # 回退到外部 label_dir
    if not label_dir or not os.path.isdir(label_dir):
        # 尝试序列内部的 label 目录（无 subdir）
        internal_fallback = os.path.join(data_root_seq, 'label')
        if os.path.isdir(internal_fallback):
            return internal_fallback
        return None

    seq_name = os.path.basename(data_root_seq)
    record_name = os.path.basename(os.path.dirname(data_root_seq))

    # 直接结构: <label_dir>/<seq>/label/<label_subdir>
    direct = os.path.join(label_dir, seq_name, 'label', label_subdir)
    if os.path.isdir(direct):
        return direct

    # record 子目录: <label_dir>/<record>/<seq>/label/<label_subdir>
    candidate = os.path.join(label_dir, record_name, seq_name, 'label', label_subdir)
    if os.path.isdir(candidate):
        return candidate

    # 任意层级镜像: <label_dir>/<rel_path_from_data_dir>/label/<label_subdir>
    if data_dir is not None:
        rel = os.path.relpath(data_root_seq, data_dir)
        mirror = os.path.join(label_dir, rel, 'label', label_subdir)
        if os.path.isdir(mirror):
            return mirror

    # 递归搜索: 在 label_dir 下任意位置找到 <seq>/label/<label_subdir>
    if os.path.isdir(label_dir):
        for root, _dirs, _files in os.walk(label_dir):
            candidate = os.path.join(root, seq_name, 'label', label_subdir)
            if os.path.isdir(candidate):
                return candidate
            # 如果当前目录本身就是 seq，则优先检查当前目录下的 label
            if os.path.basename(root) == seq_name:
                candidate2 = os.path.join(root, 'label', label_subdir)
                if os.path.isdir(candidate2):
                    return candidate2
    return None


def build_sequence_infos(data_root, cam_names, cam_dir_name, calib_dir,
                         lidar_calib_path, label_dir, coord='rfu',
                         cam_name_map=None, scene_token=None, token_prefix='',
                         max_cam_ts_diff=100, lidar_subdir=None,
                         calib_coord='rfu'):
    """为单个序列构建所有帧的 info dict 列表。"""
    if not os.path.isdir(calib_dir):
        print(f'ERROR: calib dir not found: {calib_dir}')
        return []
    print(f'Camera calib dir: {calib_dir}')

    if not os.path.isfile(lidar_calib_path):
        print(f'ERROR: LiDAR calib not found: {lidar_calib_path}')
        return []
    print(f'LiDAR calib: {lidar_calib_path}')

    # 探测 lidar 目录
    lidar_abs_dir = None
    lidar_rel_dir = None
    if lidar_subdir is not None and os.path.isdir(os.path.join(data_root, lidar_subdir)):
        lidar_abs_dir = os.path.join(data_root, lidar_subdir)
        lidar_rel_dir = lidar_subdir
    if lidar_abs_dir is None:
        lidar_candidates = [
            'lidar',
            'LIDAR_TOP',
            os.path.join('lidar', 'bin_no_self_point'),
        ]
        for cand in lidar_candidates:
            cand_path = os.path.join(data_root, cand)
            if os.path.isdir(cand_path):
                lidar_abs_dir = cand_path
                lidar_rel_dir = cand
                break

    if lidar_abs_dir is None:
        print(f'ERROR: no lidar directory found in {data_root}')
        return []

    lidar_fnames = {}
    for f in os.listdir(lidar_abs_dir):
        stem, ext = os.path.splitext(f)
        if ext.lower() in ('.bin', '.pcd', '.ply'):
            lidar_fnames[stem] = f

    frame_ids = sorted(lidar_fnames.keys())
    if not frame_ids:
        print(f'ERROR: no LiDAR files found in {lidar_abs_dir}')
        return []

    print(f'Found {len(frame_ids)} LiDAR frames')

    # 加载 JSON 标注
    frame_labels, label_types = load_json_labels(label_dir, coord=coord)
    if frame_labels:
        labeled_count = sum(1 for fid in frame_ids if fid in frame_labels)
        print(f'Labeled frames: {labeled_count}/{len(frame_ids)}, '
              f'types: {sorted(label_types)}')
    else:
        print(f'No labels found in {label_dir}, generating pkl without GT boxes')

    # 扫描相机目录
    _map = cam_name_map if cam_name_map is not None else CAM_PHYSICAL_MAP
    cam_ts_map = {}
    cam_fname_map = {}
    for cam in cam_names:
        physical = _map.get(cam, cam)
        cam_dir = os.path.join(data_root, cam_dir_name, physical)
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
        print(f'  Camera {cam} ({physical}): {len(ts_list)} images')

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
        labels = frame_labels.get(lidar_id)

        if lidar_ts is not None:
            cam_frame_map = {}
            valid = True
            for cam in cam_names:
                ts_list = sorted(cam_fname_map.get(cam, {}).keys())
                if not ts_list:
                    valid = False
                    break
                best_ts = min(ts_list, key=lambda t: abs(t - lidar_ts))
                if abs(best_ts - lidar_ts) > max_cam_ts_diff:
                    valid = False
                    break
                cam_frame_map[cam] = cam_fname_map[cam][best_ts]
            if valid:
                first_cam = cam_names[0]
                cam_frame_id = cam_frame_map[first_cam].rsplit('.', 1)[0]
                n_labels = len(labels[1]) if labels else 0
                print(f' -> camera id={cam_frame_id} (labels={n_labels})')
                info = build_single_frame(
                    data_root, cam_frame_id, cam_names,
                    calib_dir, lidar_calib_path,
                    lidar_frame_id=lidar_id, lidar_fname=lidar_fname,
                    lidar_subdir=lidar_rel_dir, cam_dir_name=cam_dir_name,
                    cam_frame_map=cam_frame_map, cam_name_map=cam_name_map,
                    labels=labels, scene_token=scene_token,
                    token_prefix=token_prefix, calib_coord=calib_coord)
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
            info = build_single_frame(
                data_root, cam_frame_id, cam_names,
                calib_dir, lidar_calib_path,
                lidar_frame_id=lidar_id, cam_frame_map=cam_frame_map,
                lidar_fname=lidar_fname, lidar_subdir=lidar_rel_dir,
                cam_dir_name=cam_dir_name, cam_name_map=cam_name_map,
                labels=labels, scene_token=scene_token,
                token_prefix=token_prefix, calib_coord=calib_coord)
        else:
            print()
            info = build_single_frame(
                data_root, lidar_id, cam_names,
                calib_dir, lidar_calib_path,
                lidar_fname=lidar_fname, lidar_subdir=lidar_rel_dir,
                cam_dir_name=cam_dir_name, cam_name_map=cam_name_map,
                labels=labels, scene_token=scene_token,
                token_prefix=token_prefix, calib_coord=calib_coord)
        infos.append(info)

    if skipped > 0:
        print(f'Skipped {skipped} frames')

    return infos


def _is_sequence_dir(path, cam_dir_name, lidar_name):
    """判断一个目录是否是序列目录（包含 image 和 lidar/<lidar_name> 子目录）。"""
    return (os.path.isdir(os.path.join(path, cam_dir_name)) and
            os.path.isdir(os.path.join(path, 'lidar', lidar_name)))


def _scan_sequences(data_dir, cam_dir_name, lidar_name):
    """递归扫描 data_dir 下的所有序列目录。

    支持任意层级嵌套：<batch>/<record>/<sequence>、<record>/<sequence> 或扁平结构。
    返回 [(record, seq), ...]:
      - 扁平结构：record=None, seq=<seq>
      - 嵌套结构：record=<relative_parent>, seq=<seq>
    """
    results = []
    for dirpath, dirnames, _filenames in os.walk(data_dir):
        rel = os.path.relpath(dirpath, data_dir)
        if rel == '.':
            continue
        # 跳过非序列目录中不需要继续遍历的目录
        if os.path.basename(rel) in ('calib', 'calibration', 'image', 'lidar', 'label', 'image_undist', 'lidar_rgb', 'lidar_rgb_ply', 'seg_fixed'):
            dirnames[:] = []
            continue
        if _is_sequence_dir(dirpath, cam_dir_name, lidar_name):
            parent = os.path.dirname(rel)
            record = parent if parent else None
            seq = os.path.basename(rel)
            results.append((record, seq))
            # 找到序列目录后不再递归进入
            dirnames[:] = []
    return sorted(results, key=lambda x: (x[0] or '', x[1]))


def _load_cam_map(path):
    if path is None or not os.path.isfile(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def _load_record_map(path):
    if path is None or not os.path.isfile(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def main():
    args = parse_args()
    cam_names = args.cam_names
    cam_dir_name = args.cam_dir_name

    # 多序列模式检测
    multi_seq = args.data_dir is not None and os.path.isdir(args.data_dir)

    if not multi_seq and not args.data_root:
        print('ERROR: must specify either --data-root (single) or '
              '--data-dir (multi-sequence)')
        sys.exit(1)

    if multi_seq:
        # ====== 多序列模式 ======
        # --calib-dir 自动探测: 优先级
        #   1) 用户显式指定 --calib-dir
        #   2) <data-dir>/calib/
        #   3) <data-dir>/calib/camera/
        #   4) <parent-of-data-dir>/calib/     (e.g. yk_c1/data -> yk_c1/calib)
        #   5) <parent-of-data-dir>/calib/camera/
        #   6) <data-dir>/calibration/
        #   7) <data-dir>/calibration/camera/
        if args.calib_dir and os.path.isdir(args.calib_dir):
            calib_dir = args.calib_dir
        else:
            calib_dir = None
            parent = os.path.dirname(args.data_dir)
            calib_candidates = [
                os.path.join(args.data_dir, 'calib'),
                os.path.join(args.data_dir, 'calib', 'camera'),
                os.path.join(parent, 'calib'),
                os.path.join(parent, 'calib', 'camera'),
                os.path.join(args.data_dir, 'calibration'),
                os.path.join(args.data_dir, 'calibration', 'camera'),
            ]
            for cand in calib_candidates:
                if os.path.isdir(cand):
                    calib_dir = cand
                    print(f'Auto-detected --calib-dir: {calib_dir}')
                    break
            if calib_dir is None:
                print(f'ERROR: --calib-dir not found. Tried: {calib_candidates}. '
                      f'Specify with --calib-dir.')
                sys.exit(1)

        cam_map_global = _load_cam_map(args.cam_map)

        # --- 决定要处理的 (record, sequence) 列表 ---
        seq_entries = []
        user_specified = args.seq_list is not None and len(args.seq_list) > 0
        if user_specified:
            for item in args.seq_list:
                if '/' in item:
                    # 支持多级路径: batch/record/seq
                    seq = os.path.basename(item)
                    record = os.path.dirname(item)
                    seq_entries.append((record if record else None, seq))
                else:
                    seq_entries.append((None, item))
        else:
            # 自动扫描: 支持任意层级嵌套（扁平/record/sequence/batch/record/sequence）
            print(f'Auto-scanning --data-dir: {args.data_dir}')
            seq_entries = _scan_sequences(args.data_dir, cam_dir_name, args.lidar_name)
            for rec, sq in seq_entries:
                if rec is None:
                    print(f'  Detected sequence: {sq}')
                else:
                    print(f'  Detected sequence: {rec}/{sq}')

        if not seq_entries:
            print('ERROR: no valid sequence found. Check --data-dir layout.')
            sys.exit(1)

        print(f'\nWill process {len(seq_entries)} sequence(s):')
        for rec, sq in seq_entries:
            print(f'  {rec}/{sq}' if rec else f'  {sq}')


        all_infos = []
        for record, seq in seq_entries:
            # 数据目录: <data-dir>/<record>/<sequence>/  或  <data-dir>/<sequence>/
            if record is not None:
                data_root_seq = os.path.join(args.data_dir, record, seq)
                lidar_rel_seq = os.path.join('lidar', args.lidar_name)
                scene_token = f'{record}/{seq}'
                # token_prefix 不能包含 '/'，否则会影响后续解析
                token_prefix = f'{record}_{seq}_'.replace('/', '_')
            else:
                data_root_seq = os.path.join(args.data_dir, seq)
                lidar_rel_seq = os.path.join('lidar', args.lidar_name)
                scene_token = seq
                token_prefix = f'{seq}_'

            if not os.path.isdir(data_root_seq):
                print(f'WARNING: data dir not found: {data_root_seq}, skipping')
                continue

            # LiDAR 子目录
            lidar_abs_seq = os.path.join(data_root_seq, lidar_rel_seq)
            if not os.path.isdir(lidar_abs_seq):
                print(f'WARNING: lidar dir not found: {lidar_abs_seq}, skipping')
                continue

            # 标注目录: 优先序列内部的 label/<label_subdir>
            label_dir_seq = _find_label_dir_for_seq(
                data_root_seq, args.label_dir, args.label_subdir,
                data_dir=args.data_dir)
            if not label_dir_seq or not os.path.isdir(label_dir_seq):
                print(f'WARNING: label dir not found for {scene_token}, using None')
                label_dir_seq = None
            else:
                print(f'Label dir for {scene_token}: {label_dir_seq}')

            # 相机映射
            lookup_key = seq if record is None else seq
            cam_name_map = cam_map_global.get(lookup_key)
            if cam_name_map is None:
                cam_name_map = CAM_PHYSICAL_MAP
                print(f'Using default camera mapping for {scene_token}')
            else:
                print(f'Using custom camera mapping for {scene_token}: {cam_name_map}')

            print(f'\n=== Processing: {scene_token} ===')
            infos = build_sequence_infos(
                data_root=data_root_seq,
                cam_names=cam_names,
                cam_dir_name=cam_dir_name,
                calib_dir=calib_dir,
                lidar_calib_path=args.lidar_calib,
                label_dir=label_dir_seq,
                coord=args.coord,
                cam_name_map=cam_name_map,
                scene_token=scene_token,
                token_prefix=token_prefix,
                max_cam_ts_diff=100,
                lidar_subdir=lidar_rel_seq,
                calib_coord=args.calib_coord)
            all_infos.extend(infos)

        dataset = {
            'infos': all_infos,
            'metadata': {
                'version': 'custom',
                'cam_names': cam_names,
            },
        }

        with open(args.output, 'wb') as f:
            pickle.dump(dataset, f)

        print(f'\nSaved {len(all_infos)} frames from {len(seq_entries)} '
              f'sequences to {args.output}')

    else:
        # ====== 单序列模式（兼容旧版本） ======
        data_root = args.data_root
        calib_dir = args.calib_dir or os.path.join(data_root, 'calib', 'camera')
        label_dir = args.label_root
        if not label_dir:
            # 自动探测 label 目录
            internal = os.path.join(data_root, 'label', args.label_subdir)
            if os.path.isdir(internal):
                label_dir = internal
            else:
                label_dir = os.path.join(data_root, 'label')
        cam_name_map = _load_cam_map(args.cam_map).get('default') or None

        infos = build_sequence_infos(
            data_root=data_root,
            cam_names=cam_names,
            cam_dir_name=cam_dir_name,
            calib_dir=calib_dir,
            lidar_calib_path=args.lidar_calib,
            label_dir=label_dir,
            coord=args.coord,
            cam_name_map=cam_name_map,
            scene_token='custom_scene',
            token_prefix='',
            max_cam_ts_diff=100,
            lidar_subdir=os.path.join('lidar', args.lidar_name),
            calib_coord=args.calib_coord)

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