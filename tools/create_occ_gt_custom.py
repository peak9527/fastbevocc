#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从点云语义标签生成 occupancy 真值 labels.npz（与 nuScenes occ3d / FlashOCC 格式对齐）。

输入：
  - 由 tools/create_custom_pkl.py 生成的 infos pkl（yk_c1 等自定义数据），
    其中每帧 info 含：
      lidar_path              相对 data_root 的点云路径（6 列 bin: x,y,z,intensity,语义,实例）
      lidar2ego_rotation      [w,x,y,z] 四元数（FLU）
      lidar2ego_translation   [x,y,z]（FLU）
      cams.<CAM>.cam_intrinsic          3x3
      cams.<CAM>.sensor2ego_rotation    四元数（FLU）
      cams.<CAM>.sensor2ego_translation [x,y,z]（FLU）
  - 点云 6 列格式：x, y, z, intensity, 语义信息(第5列), 实例信息(第6列)
    语义列（如 3~29，18 类）通过 --label-map 映射到 occ 类别 1..N。
    约定（仿照 nuScenes occ3d 的 FlashOCC 18 类 = 0~16 语义 + 17=free）：
      yk_c1 体系 33 类 = 0 无标注/ignore(mask_camera 外) + 1~31 语义 + 32=free(空)。
      mask_camera 内无语义标注的体素自动标为 free（显式参与训练），
      mIoU 评估时排除 free（只统计前 num_classes-1 类）。

输出：
  - 每帧 <occ_path>/labels.npz，含：
      semantics   (Dx,Dy,Dz) uint8  体素语义类别（0=ignore；1~31 语义；32=free）
      mask_lidar  (Dx,Dy,Dz) uint8  有 LiDAR 观测的体素（评估用）
      mask_camera (Dx,Dy,Dz) uint8  相机可见的体素（训练 loss mask 用）
  - 并把 occ_path 写回 pkl（--write-back）。

用法示例：
  # 生成全部帧；label 映射见 --label-map（json: {"3242":1, ..., "0":0}）
  python tools/create_occ_gt_custom.py --infos yk_c1_infos_train.pkl \
      --data-root data/custom_data/yk_c1 --label-map label_map.json \
      --out-root data/custom_data/yk_c1/occ_gts --num-workers 8
  # auto 映射：非零 label 按数值升序自动映射 1..N（仅用于试跑，正式请提供映射表）
  python tools/create_occ_gt_custom.py --infos yk_c1_infos_train.pkl \
      --data-root data/custom_data/yk_c1 --label-map auto --out-root ... --max-frames 5
"""
import argparse
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from pyquaternion import Quaternion

# 与 nuScenes occ3d / FastBEVOCCHead 对齐的 occ 栅格
OCC_X = [-40.0, 40.0, 0.4]   # 200
OCC_Y = [-40.0, 40.0, 0.4]   # 200
OCC_Z = [-1.0, 5.4, 0.4]     # 16
OCC_SHAPE = (200, 200, 16)   # (Dx, Dy, Dz) 与 labels.npz 一致

# 仿照 nuScenes occ3d（FlashOCC 18 类 = 0~16 语义 + 17=free）：
# yk_c1 自定义体系 33 类 = 0 无标注/ignore(mask_camera 外) + 1~31 语义 + 32=free(空)。
# free 作为显式类别参与训练（模型学"空"），评估时被排除（mIoU 只统计前 num_classes-1 类）。
FREE_ID = 32


def load_points6(path):
    """读取 6 列 bin: x,y,z,intensity,语义,实例 -> (N,3) xyz, (N,) semantic(int), (N,) instance(int)"""
    raw = np.fromfile(path, dtype=np.float32)
    if len(raw) % 6 != 0:
        raise ValueError(f'not 6-float points: {path} ({len(raw)} floats)')
    arr = raw.reshape(-1, 6)
    return arr[:, :3].copy(), arr[:, 4].astype(np.int64), arr[:, 5].astype(np.int64)


def lidar2ego(points, info):
    """点云 lidar 系 -> ego 系（FLU）。info: lidar2ego_rotation [w,x,y,z], translation"""
    rot = Quaternion(info['lidar2ego_rotation'])
    t = np.array(info['lidar2ego_translation'], dtype=np.float32)
    return points @ rot.rotation_matrix.T + t


def voxel_coords(points_ego):
    """ego 系点 -> occ 体素索引 (Dx,Dy,Dz)。越界点返回 -1。"""
    xs = np.floor((points_ego[:, 0] - OCC_X[0]) / OCC_X[2]).astype(np.int64)
    ys = np.floor((points_ego[:, 1] - OCC_Y[0]) / OCC_Y[2]).astype(np.int64)
    zs = np.floor((points_ego[:, 2] - OCC_Z[0]) / OCC_Z[2]).astype(np.int64)
    valid = ((xs >= 0) & (xs < OCC_SHAPE[0]) &
             (ys >= 0) & (ys < OCC_SHAPE[1]) &
             (zs >= 0) & (zs < OCC_SHAPE[2]))
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    return np.stack([xs, ys, zs], axis=1), valid


def build_occ_from_points(points_ego, labels, label_map, free_label=None,
                          free_id=FREE_ID):
    """由 ego 系点云 + 语义标签生成 semantics / mask_lidar。

    Args:
        label_map: dict {raw_label: occ_class}
        free_label: raw label 值，若给出则该 label 的点视为 free(free_id)
        free_id: free(空)类的 occ 类别号（默认 FREE_ID=32）
    """
    semantics = np.zeros(OCC_SHAPE, dtype=np.uint8)
    mask_lidar = np.zeros(OCC_SHAPE, dtype=np.uint8)

    # 映射标签（label_map 未命中的 label -> 忽略(0)，仅计入观测 mask）
    occ_labels = np.zeros_like(labels)
    for raw, cls in label_map.items():
        occ_labels[labels == raw] = cls
    if free_label is not None:
        occ_labels[labels == free_label] = free_id

    coords, valid = voxel_coords(points_ego)
    occ_labels = occ_labels[valid]
    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]

    # 所有有点的体素均计入 mask_lidar（无论 label 是否有效）
    mask_lidar[xs, ys, zs] = 1
    # 有效标签点赋语义；同一体素多类时取首个（点云按距离排序，通常最近点为准）
    pos = occ_labels > 0
    xs_v, ys_v, zs_v = xs[pos], ys[pos], zs[pos]
    semantics[xs_v, ys_v, zs_v] = occ_labels[pos]
    return semantics, mask_lidar


_CAM_SIZE_CACHE = {}


def _cam_image_size(cam):
    """从 data_path 读取相机图片宽高（按目录缓存，同目录尺寸固定）。

    pkl 的 cams 未写 width/height，且不同相机分辨率不同
    （CAM_FRONT=3840x2160，其余 5 相机=1920x1536），读图取真实尺寸，
    否则 mask_camera 会把相机实际看不到的体素误判为可见。
    读图失败返回 (None, None)。
    """
    dp = cam.get('data_path')
    if not dp:
        return None, None
    key = os.path.dirname(dp)
    if key in _CAM_SIZE_CACHE:
        return _CAM_SIZE_CACHE[key]
    w = h = None
    p = dp if os.path.exists(dp) else None
    if p:
        img = cv2.imread(p)
        if img is not None:
            h, w = img.shape[:2]
    _CAM_SIZE_CACHE[key] = (w, h)
    return w, h


def compute_mask_camera(info, semantics):
    """相机可见 mask：体素中心投影到任一相机图像内（不考虑遮挡，与官方近似）。

    覆盖相机可见的【全部】体素（含无标注空体素），与官方 occ3d 的 mask_camera
    定义一致：mask_camera=1 表示"该体素在相机视野内（观测得到）"，
    mask_camera=0 表示"相机看不到（无法监督）"。mask 内无标注体素
    semantics=0（空/无标注），训练时作为普通类别参与 loss，模型可学会输出 0。
    注意：不再用 semantics>0 过滤——旧版只统计有标注的 1% 体素，导致模型
    从未学过"空空间"，推理时对 98% 无标注区域乱预测。
    """
    Dx, Dy, Dz = OCC_SHAPE
    xs = np.arange(Dx) * OCC_X[2] + OCC_X[0] + OCC_X[2] / 2
    ys = np.arange(Dy) * OCC_Y[2] + OCC_Y[0] + OCC_Y[2] / 2
    zs = np.arange(Dz) * OCC_Z[2] + OCC_Z[0] + OCC_Z[2] / 2
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel(), np.ones(gx.size)], axis=1)  # ego 系

    mask_camera = np.zeros(OCC_SHAPE, dtype=np.uint8)
    for cam in info['cams'].values():
        K = np.array(cam['cam_intrinsic'], dtype=np.float32)  # 3x3
        if K.shape != (3, 3):
            K = K.reshape(3, 3)
        rot = Quaternion(cam['sensor2ego_rotation']).rotation_matrix
        t = np.array(cam['sensor2ego_translation'], dtype=np.float32)
        cam2ego = np.eye(4, dtype=np.float32)
        cam2ego[:3, :3] = rot
        cam2ego[:3, 3] = t
        ego2cam = np.linalg.inv(cam2ego)
        pcam = pts[:, :3] @ ego2cam[:3, :3].T + ego2cam[:3, 3]  # (M,3) 相机系
        depth = pcam[:, 2]
        u = pcam[:, 0] / depth
        v = pcam[:, 1] / depth
        px = u * K[0, 0] + K[0, 2]
        py = v * K[1, 1] + K[1, 2]
        # 优先用 cams 里的 width/height；缺失时从 data_path 读真实尺寸
        w, h = cam.get('width'), cam.get('height')
        if w is None or h is None:
            w, h = _cam_image_size(cam)
        if w is None or h is None:
            w, h = 3840, 2160
        inside = (depth > 0.5) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
        mask_camera.ravel()[inside] = 1
    return mask_camera


def resolve_lidar_path(info, data_root, subdir):
    """按 pkl 中 lidar_path 解析实际点云路径；若指定 subdir（如 seg_fixed），
    把路径中的 'bin_no_self_point' 目录替换为该子目录。

    兼容两种 pkl：
      - lidar_path 相对 data_root（早期 create_custom_pkl.py 约定）
      - lidar_path 已含 data_root 前缀的全路径（yk_c1 现状，
        'data/custom_data/yk_c1/train/...'，此时 data_root 应传 '.'）
    若拼接后不存在而原路径存在，则直接用原路径。
    """
    raw = info['lidar_path']
    lidar_path = os.path.join(data_root, raw)
    if not os.path.exists(lidar_path) and os.path.exists(raw):
        lidar_path = raw
    if subdir and subdir != 'none':
        lidar_path = lidar_path.replace('/bin_no_self_point/', f'/{subdir}/')
    return lidar_path


def process_frame(args):
    info, data_root, out_root, label_map, free_label, subdir, free_id = args
    lidar_path = resolve_lidar_path(info, data_root, subdir)
    if not os.path.isfile(lidar_path):
        return info['token'], f'MISSING {lidar_path}'

    points, labels, _ = load_points6(lidar_path)
    points_ego = lidar2ego(points, info)
    semantics, mask_lidar = build_occ_from_points(points_ego, labels, label_map,
                                                  free_label, free_id)
    mask_camera = compute_mask_camera(info, semantics)
    # 仿照 nuScenes：mask_camera 内无语义标注的体素标为 free(空)，
    # 作为显式类别参与训练；0 只保留 mask_camera 外的不可观测体素
    # （不参与 loss / 评估，等价于 nuScenes 的 ignore）。
    semantics = semantics.copy()
    semantics[(semantics == 0) & (mask_camera > 0)] = free_id

    # occ_path 输出位置：<out_root>/<scene_token>/<token>
    scene_token = info.get('scene_token', 'unknown_scene')
    occ_path = os.path.join(out_root, scene_token.replace('/', '__'), info['token'])
    os.makedirs(occ_path, exist_ok=True)
    np.savez_compressed(os.path.join(occ_path, 'labels.npz'),
                        semantics=semantics, mask_lidar=mask_lidar,
                        mask_camera=mask_camera)
    return info['token'], f'OK n_pts={len(points)} valid={semantics.sum()}'


def build_label_map(spec):
    """spec: json 文件路径 或 'auto'。auto 将非零 label 按值升序映射 1..N。"""
    if spec == 'auto':
        return None  # 占位，见 run()
    with open(spec) as f:
        return {int(k): int(v) for k, v in json.load(f).items()}


def run(args):
    data = pickle.load(open(args.infos, 'rb'))
    infos = data['infos'] if isinstance(data, dict) and 'infos' in data else data
    if args.max_frames > 0:
        infos = infos[:args.max_frames]

    if args.label_map == 'auto':
        # 扫描帧（默认全部）的语义列值，升序映射 1..N
        raw_set = set()
        for info in infos:
            p = resolve_lidar_path(info, args.data_root, args.lidar_subdir)
            if os.path.isfile(p):
                _, sem, _ = load_points6(p)
                raw_set.update(np.unique(sem[sem > 0]).tolist())
        label_map = {int(v): i + 1 for i, v in enumerate(sorted(raw_set))}
        print(f'[auto] semantic map ({len(label_map)} classes): {label_map}')
    else:
        label_map = build_label_map(args.label_map)

    tasks = [(info, args.data_root, args.out_root, label_map, args.free_label,
              args.lidar_subdir, args.free_id) for info in infos]
    n_ok = 0
    if args.num_workers > 1:
        with ProcessPoolExecutor(args.num_workers) as ex:
            for tok, msg in ex.map(process_frame, tasks):
                if msg.startswith('OK'):
                    n_ok += 1
                else:
                    print(f'  {msg}')
    else:
        for task in tasks:
            tok, msg = process_frame(task)
            if msg.startswith('OK'):
                n_ok += 1
            else:
                print(f'  {msg}')
    print(f'generated {n_ok}/{len(infos)} frames -> {args.out_root}')

    # 写回 occ_path
    if args.write_back:
        for info in infos:
            scene_token = info.get('scene_token', 'unknown_scene')
            info['occ_path'] = os.path.join(
                args.out_root, scene_token.replace('/', '__'), info['token'])
        with open(args.infos, 'wb') as f:
            pickle.dump(data, f)
        print(f'occ_path written back to {args.infos}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--infos', required=True, help='infos pkl (create_custom_pkl.py 输出)')
    ap.add_argument('--data-root', required=True, help='pkl 中相对路径的根目录')
    ap.add_argument('--out-root', required=True, help='occ labels.npz 输出根目录')
    ap.add_argument('--label-map', default='auto',
                    help='label 映射 json {"raw": occ_class}，或 "auto"（升序自动 1..N）')
    ap.add_argument('--free-label', type=int, default=None,
                    help='若该 raw label 值表示 free 空间，映射到 --free-id')
    ap.add_argument('--free-id', type=int, default=FREE_ID,
                    help=f'free(空)类的 occ 类别号（默认 {FREE_ID}，'
                         '与 config num_classes=33 对应，mIoU 评估排除该类）')
    ap.add_argument('--lidar-subdir', default='bin_no_self_point',
                    help='点云子目录：把 pkl 中 lidar_path 的 bin_no_self_point 替换为该子目录'
                         '（bin_no_self_point / seg_fixed / none）')
    ap.add_argument('--max-frames', type=int, default=0, help='只处理前 N 帧（0=全部）')
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--write-back', action='store_true', help='把 occ_path 写回 infos pkl')
    args = ap.parse_args()
    run(args)


if __name__ == '__main__':
    main()
