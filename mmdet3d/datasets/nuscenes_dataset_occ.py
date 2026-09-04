# Copyright (c) OpenMMLab. All rights reserved.
import os
import mmcv
import torch
import cv2
import numpy as np
from tqdm import tqdm

from .builder import DATASETS
from .nuscenes_dataset import NuScenesDataset
from .occ_metrics import Metric_mIoU, Metric_FScore

colors_map = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [255, 158, 0, 255],  # 1 car  orange
        [0, 0, 230, 255],    # 2 pedestrian  Blue
        [47, 79, 79, 255],   # 3 sign  Darkslategrey
        [220, 20, 60, 255],  # 4 CYCLIST  Crimson
        [255, 69, 0, 255],   # 5 traiffic_light  Orangered
        [255, 140, 0, 255],  # 6 pole  Darkorange
        [233, 150, 70, 255], # 7 construction_cone  Darksalmon
        [255, 61, 99, 255],  # 8 bycycle  Red
        [112, 128, 144, 255],# 9 motorcycle  Slategrey
        [222, 184, 135, 255],# 10 building Burlywood
        [0, 175, 0, 255],    # 11 vegetation  Green
        [165, 42, 42, 255],  # 12 trunk  nuTonomy green
        [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
        [75, 0, 75, 255], # 14 walkable, sidewalk
        [255, 0, 0, 255], # 15 unobsrvd
        [0, 0, 0, 0],  # 16 undefined
        [0, 0, 0, 0],  # 16 undefined
    ])



@DATASETS.register_module()
class NuScenesDatasetOccpancy(NuScenesDataset):
    def __init__(self, *args, occ_num_classes=18, occ_class_names=None, **kwargs):
        # 兼容 nuscenes occ3d 默认 18 类；自定义数据（如 yk_c1，zhito 标注 31 类）
        # 由配置文件传入 occ_num_classes=33（31 语义 + 0 ignore + 32 free）与
        # occ_class_names（长度=occ_num_classes），供 evaluate() 中 Metric_mIoU
        # 正确构建混淆矩阵与打印类别名。
        super(NuScenesDatasetOccpancy, self).__init__(*args, **kwargs)
        self.occ_num_classes = occ_num_classes
        self.occ_class_names = occ_class_names

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        input_dict = super(NuScenesDatasetOccpancy, self).get_data_info(index)
        # standard protocol modified from SECOND.Pytorch
        input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']
        return input_dict

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        # 兼容 FastBEVOCC 同时输出检测与占用的结果形式：
        # results 为 list[dict]，每个 dict 含 'pts_bbox' 与 'occ'；
        # 此时分离出 occ 结果并额外做 3D 目标检测评估。
        det_results = None
        if (isinstance(occ_results, (list, tuple)) and len(occ_results) > 0
                and isinstance(occ_results[0], dict)):
            det_results = [
                dict(pts_bbox=result['pts_bbox']) for result in occ_results
            ]
            # 纯 3D 检测模式（如 --cfg-options model.occ_head=None 加载纯检测
            # 权重）：结果 dict 中无 'occ' 键，跳过占用评估，仅做检测评估
            if 'occ' not in occ_results[0]:
                if self._nuscenes_db_available():
                    return NuScenesDataset.evaluate(self, det_results,
                                                    **eval_kwargs)
                self._warn_no_nuscenes_db()
                return {}
            occ_results = [result['occ'] for result in occ_results]

        self.occ_eval_metrics = Metric_mIoU(
            num_classes=self.occ_num_classes,
            use_lidar_mask=False,
            use_image_mask=True,
            class_names=self.occ_class_names)

        print('\nStarting Evaluation...')
        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]

            occ_gt = np.load(os.path.join(info['occ_path'],'labels.npz'))
            gt_semantics = occ_gt['semantics']
            mask_lidar = occ_gt['mask_lidar'].astype(bool)
            mask_camera = occ_gt['mask_camera'].astype(bool)
            # occ_pred = occ_pred
            self.occ_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)

            if index%100==0 and show_dir is not None:
                gt_vis = self.vis_occ(gt_semantics)
                pred_vis = self.vis_occ(occ_pred)
                mmcv.imwrite(np.concatenate([gt_vis, pred_vis], axis=1),
                             os.path.join(show_dir, "%d.jpg" % index))

        occ_metrics = self.occ_eval_metrics.count_miou()

        # count_miou() 返回 (class_names, per_class_IoU, cnt) 元组，
        # 需转换为 dict 才能与检测评估结果(dict)合并；键加 'occ/' 前缀避免冲突
        class_names, per_class_ious, cnt = occ_metrics
        occ_metrics = {
            f'occ/mIoU/{name}': round(iou * 100, 2)
            for name, iou in zip(class_names, per_class_ious)
        }
        # 与 count_miou() 内打印一致：mIoU 仅统计前 num_classes-1 类（不含 'free'）
        occ_metrics['occ/mIoU'] = round(
            np.nanmean(
                per_class_ious[:self.occ_eval_metrics.num_classes - 1]) * 100,
            2)

        if det_results is not None:
            # 自定义数据（如 yk_c1，metadata['version']='custom'）通常没有对应的
            # nuScenes 官方数据库，NuScenesEval 无法加载 GT 表，直接评估会抛
            # "Database version not found"。存在数据库目录时才执行检测评估。
            if not self._nuscenes_db_available():
                self._warn_no_nuscenes_db()
                return occ_metrics
            det_metrics = NuScenesDataset.evaluate(self, det_results,
                                                   **eval_kwargs)
            return {**occ_metrics, **det_metrics}
        return occ_metrics

    def _nuscenes_db_available(self):
        """检测评估(NuScenesEval)需要 nuScenes 官方数据库加载 GT 表。

        数据库目录 = {data_root}/{version}（如 ./v1.0-trainval）。
        自定义数据 version='custom' 且无对应目录时返回 False。
        """
        table_root = os.path.join(self.data_root, self.version)
        return os.path.isdir(table_root)

    def _warn_no_nuscenes_db(self):
        table_root = os.path.join(self.data_root, self.version)
        print(
            f'\n[警告] nuScenes 检测评估数据库不存在: {table_root}\n'
            f'        自定义数据无法执行官方检测评估(NuScenesEval)，'
            f'本次仅输出 occ mIoU 指标。\n'
            f'        如需检测评估，请在 {table_root} 准备对应的 '
            f'nuScenes 格式数据库(如 v1.0-trainval 目录结构)。')

    def vis_occ(self, semantics):
        # simple visualization of result in BEV
        # 栅格尺寸由数组形状自适应（当前 0.2m: 400x400x16）；
        # free(最后一类=空) 不参与"取顶"。
        free_id = self.occ_num_classes - 1
        semantics_valid = np.logical_not(semantics == free_id)
        D0, D1, Dz = semantics.shape
        d = np.arange(Dz).reshape(1, 1, Dz)
        d = np.repeat(d, D0, axis=0)
        d = np.repeat(d, D1, axis=1).astype(np.float32)
        d = d * semantics_valid
        selected = np.argmax(d, axis=2)

        selected_torch = torch.from_numpy(selected)
        semantics_torch = torch.from_numpy(semantics)

        occ_bev_torch = torch.gather(semantics_torch, dim=2,
                                     index=selected_torch.unsqueeze(-1))
        occ_bev = occ_bev_torch.numpy()

        occ_bev = occ_bev.flatten().astype(np.int32)
        # 自定义数据类别数可能 > colors_map 行数（18），越界索引防落到黑/透明
        occ_bev[occ_bev >= len(colors_map)] = 0
        occ_bev_vis = colors_map[occ_bev].astype(np.uint8)
        occ_bev_vis = occ_bev_vis.reshape(D0, D1, 4)[::-1, ::-1, :3]
        occ_bev_vis = cv2.resize(occ_bev_vis, (400, 400))
        return occ_bev_vis