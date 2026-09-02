#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统计自定义 occ 训练集的真实类别频率，用于替换 FastBEVOCCHead 内置频率表。

FastBEVOCCHead.class_balance=True 时 loss 的加权依据是
`valid_voxels = voxel_semantics[mask_camera]` 的类别分布（含 free 类）。
内置 nusc_class_frequencies 是 nuScenes 18 类，与自定义 33 类（zhito 体系：
0=ignore、1~31 语义、32=free）不一致。本脚本从 labels.npz 统计出 33 长的
真实频率数组，可直接写回 bev_occ_head.py。

口径与 loss 严格一致：
  freq_mc[i] = 训练集所有帧中，mask_camera=1 且 semantics=i 的体素总数
也顺带输出 mask_lidar 口径（评估用），仅作参考。

用法：
  python tools/compute_occ_class_freq.py --infos yk_c1_infos_train.pkl \
      --data-root . --num-classes 33 [--out freq.npy]
"""
import argparse
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def _count_one(args):
    info, data_root, num_classes = args
    path = os.path.join(info['occ_path'], 'labels.npz')
    path = path if os.path.isabs(path) else os.path.join(data_root, path)
    with np.load(path) as occ:
        semantics = occ['semantics']
        mask_lidar = occ['mask_lidar'].astype(bool)
        mask_camera = occ['mask_camera'].astype(bool)
    freq_mc = np.bincount(semantics[mask_camera], minlength=num_classes)
    freq_ml = np.bincount(semantics[mask_lidar], minlength=num_classes)
    return freq_mc, freq_ml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infos', required=True, help='训练 infos pkl 路径')
    parser.add_argument('--data-root', default='.', help='occ_path 相对根目录')
    parser.add_argument('--num-classes', type=int, default=33)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--out', default=None, help='可选：保存频率数组到 npy')
    args = parser.parse_args()

    with open(args.infos, 'rb') as f:
        infos = pickle.load(f)['infos']
    print(f'frames: {len(infos)}')

    total_mc = np.zeros(args.num_classes, dtype=np.float64)
    total_ml = np.zeros(args.num_classes, dtype=np.float64)
    tasks = [(info, args.data_root, args.num_classes) for info in infos]
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        for fmc, fml in ex.map(_count_one, tasks):
            total_mc += fmc
            total_ml += fml

    total_mc = total_mc.astype(np.int64)
    total_ml = total_ml.astype(np.int64)
    print('\nmask_camera 口径（class_balance loss 使用）:')
    print('freq_mc =', np.array2string(total_mc, separator=', '))
    print('\nmask_lidar 口径（评估用，参考）:')
    print('freq_ml =', np.array2string(total_ml, separator=', '))

    if args.out:
        np.save(args.out, total_mc)
        print(f'\nsaved -> {args.out}')

    # 直接生成可粘贴进 bev_occ_head.py 的 numpy 数组文本
    lines = ['    ' + ', '.join(f'{v}' for v in total_mc.tolist())]
    print('\n粘贴到 bev_occ_head.py 的数组内容:')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
