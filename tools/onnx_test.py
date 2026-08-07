#!/usr/bin/env python3
"""
FastBEV ONNX 推理脚本

使用 ONNX 模型进行 3D 目标检测推理，替代原有基于 PyTorch .pth 的推理流程。

神经网络推理完全由 ONNX Runtime 执行，PyTorch 模型仅用于：
  1. 计算每帧的 FastRay 坐标 (coors_img / coors_depth) — 纯几何运算，无需权重
  2. 对 ONNX 输出进行 bbox 解码 (decode + NMS) — config 参数即可，无需权重

使用方法:
  # 仅生成格式化结果（不评估）
  python tools/onnx_test.py configs/fastbev/paper/fastbev-r50-cbgs-custom.py \
      models/fastbev_fp16_fuse.onnx \
      --format-only \
      --eval-options 'jsonfile_prefix=work_dirs/yk_c1_train/results_custom_onnx'

  # 生成结果并评估
  python tools/onnx_test.py configs/fastbev/paper/fastbev-r50-cbgs-custom.py \
      models/fastbev_fp16_fuse.onnx \
      --format-only \
      --eval bbox \
      --eval-options 'jsonfile_prefix=work_dirs/yk_c1_train/results_custom_onnx'
"""

import argparse
import sys
import time

import numpy as np
import onnxruntime as ort
import torch
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from tqdm import tqdm

from mmdet3d.core import bbox3d2result
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

try:
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg


def parse_args():
    parser = argparse.ArgumentParser(description='FastBEV ONNX Inference')
    parser.add_argument('config', help='model config file path')
    parser.add_argument('onnx', help='ONNX 模型文件路径')
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='PyTorch checkpoint 路径（可选，仅用于验证 coors/bbo '
             'decode 模块结构，不需要实际权重）')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='仅格式化为 nuScenes submission 格式，不评估')
    parser.add_argument('--eval', type=str, nargs='+',
                        help='评估指标，如 "bbox"')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='评估选项，如: jsonfile_prefix=output_dir')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU ID (default: 0)')
    parser.add_argument('--providers', type=str, nargs='+',
                        default=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                        help='ONNX Runtime execution providers (default: CUDA CPU)')
    return parser.parse_args()


def build_model_from_config(cfg, device, checkpoint_path=None):
    """构建 PyTorch 模型（仅用于 coors 计算和 bbox 解码，无需实际权重）。"""
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if checkpoint_path is not None:
        load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.to(device)
    model.eval()
    return model


def _unwrap_img_meta(metas):
    """穿透 DataContainer / list / tuple，提取单个样本的 img_meta 字典。

    mmdet collate 对不同 num_augs 有不同包裹层级，此函数持续试探解包
    直到拿到真正的 dict 为止，避免碰触 DataContainer 的 __getitem__ 行为差异。
    """
    MAX_DEPTH = 10
    for _ in range(MAX_DEPTH):
        if isinstance(metas, dict):
            return metas
        if hasattr(metas, 'data'):
            metas = metas.data
        if isinstance(metas, (list, tuple)):
            metas = metas[0] if len(metas) > 0 else metas
        else:
            try:
                metas = metas[0]
            except (TypeError, IndexError, KeyError):
                break
    raise RuntimeError(
        f'Cannot unwrap img_metas to dict, got {type(metas)}: {metas}')


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu_id}'
                          if torch.cuda.is_available() else 'cpu')

    # ---- 加载配置 ----
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    # 使用 TRT 模型类型并开启 accelerate（与 ONNX 导出时一致）
    cfg.model.type = cfg.model.type + 'TRT'
    cfg.model.img_view_transformer.accelerate = True
    cfg = compat_cfg(cfg)
    cfg.gpu_ids = [args.gpu_id]

    # ---- 构建数据集 ----
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=False,
        shuffle=False)
    print(f'Dataset: {len(dataset)} samples')

    # ---- 构建 PyTorch 模型（仅用于 coors 计算 + bbox 解码，无需权重） ----
    if args.checkpoint:
        print(f'Loading PyTorch checkpoint: {args.checkpoint}')
    else:
        print('Building model from config (no checkpoint, coors/decode only)')
    model = build_model_from_config(cfg, device, args.checkpoint)

    # 确保 CLASSES 正确设置（test.py 在 load_checkpoint 后会做这一步）
    if not hasattr(model, 'CLASSES') or model.CLASSES is None:
        model.CLASSES = dataset.CLASSES
    print(f'Model CLASSES: {model.CLASSES}')

    # 从 config 读取特征图尺寸（无需跑 image_encoder）
    vt_cfg = cfg.model.img_view_transformer
    feat_h, feat_w = vt_cfg.get('feature_size', [
        vt_cfg['image_size'][0] // vt_cfg.get('stride', 16),
        vt_cfg['image_size'][1] // vt_cfg.get('stride', 16),
    ])
    print(f'Feature map size: H={feat_h}, W={feat_w}')

    # ---- 加载 ONNX 模型 ----
    print(f'Loading ONNX model: {args.onnx}')
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = \
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        args.onnx,
        sess_options,
        providers=args.providers)

    # 诊断：打印实际使用的 execution provider（FP16 模型在 CPU 上可能异常）
    actual_providers = session.get_providers()
    active_provider = ort.get_device()
    print(f'Requested providers: {args.providers}')
    print(f'Active providers   : {actual_providers}')
    print(f'ORT device         : {active_provider}')
    if 'CUDAExecutionProvider' not in actual_providers:
        print('⚠️  WARNING: CUDAExecutionProvider NOT available! '
              'FP16 ONNX model on CPU may produce incorrect results. '
              'Install onnxruntime-gpu: pip install onnxruntime-gpu')

    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    # 诊断：打印 ONNX 模型的输入/输出详情（shape + dtype）
    print(f'ONNX inputs ({len(input_names)}):')
    onnx_input_info = {}  # name -> (shape, dtype_str)
    for inp in session.get_inputs():
        shape_str = str(list(inp.shape))
        dtype_str = inp.type
        onnx_input_info[inp.name] = (inp.shape, dtype_str)
        print(f'  {inp.name}: shape={shape_str}, dtype={dtype_str}')
    print(f'ONNX outputs: {len(output_names)} tensors')
    for out in session.get_outputs():
        print(f'  {out.name}: shape={list(out.shape)}, dtype={out.type}')

    # ---- 推理循环 ----
    results = []
    total_ort_ms = 0.0
    total_coors_ms = 0.0
    empty_count = 0  # 统计空检测帧数

    # 确定 ONNX 输入 tensor 的 numpy dtype（与导出时一致）
    onnx_np_dtypes = {}
    for inp in session.get_inputs():
        dtype_str = inp.type
        if 'float' in dtype_str:
            onnx_np_dtypes[inp.name] = np.float32
        elif 'int64' in dtype_str:
            onnx_np_dtypes[inp.name] = np.int64
        elif 'int32' in dtype_str:
            onnx_np_dtypes[inp.name] = np.int32
        else:
            onnx_np_dtypes[inp.name] = np.int64  # fallback

    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader, desc='ONNX Inference')):
            img_inputs = data['img_inputs'][0]   # 7-tuple (first aug)

            # --- 1) 计算 coors ---
            t0 = time.perf_counter()
            img_raw = img_inputs[0][0].to(device, non_blocking=True)
            # 构造 get_fastray_input 所需的 input tuple（仅用 shape）
            dummy_feat = torch.zeros(
                1, 6, 1, feat_h, feat_w, device=device)
            coors_input = [dummy_feat] + [
                d.to(device, non_blocking=True) for d in img_inputs[1:]]
            _, coors_img, coors_depth = \
                model.img_view_transformer.get_fastray_input(coors_input)
            total_coors_ms += (time.perf_counter() - t0) * 1000

            # --- 2) ONNX 推理（dtype 与 ONNX 模型导出时严格匹配）---
            ort_inputs = {
                input_names[0]: img_raw.float().contiguous().cpu().numpy()
                    .astype(onnx_np_dtypes.get(input_names[0], np.float32)),
                input_names[1]: coors_img[0].contiguous().cpu().numpy()
                    .astype(onnx_np_dtypes.get(input_names[1], np.int64)),
                input_names[2]: coors_depth[0].contiguous().cpu().numpy()
                    .astype(onnx_np_dtypes.get(input_names[2], np.int64)),
            }
            t0 = time.perf_counter()
            ort_outputs = session.run(output_names, ort_inputs)
            total_ort_ms += (time.perf_counter() - t0) * 1000

            # --- 诊断：第一个样本输出统计 ---
            if i == 0:
                print('\n--- First sample ONNX output diagnostics ---')
                for j, o in enumerate(ort_outputs):
                    print(f'  output[{j}] shape={o.shape} dtype={o.dtype} '
                          f'min={o.min():.6f} max={o.max():.6f} '
                          f'mean={o.mean():.6f} std={o.std():.6f} '
                          f'nonzero_ratio={np.count_nonzero(o) / o.size:.4f}')
                if all(np.count_nonzero(o) == 0 for o in ort_outputs):
                    print('⚠️  CRITICAL: All ONNX outputs are ZERO! '
                          'Possible causes:\n'
                          '  - ONNX Runtime is running on CPU with FP16 model '
                          '(install onnxruntime-gpu)\n'
                          '  - Input data type mismatch\n'
                          '  - ONNX model is corrupted')
                print('--- End diagnostics ---\n')

            # --- 3) 反序列化 + 解码 ---
            outs_torch = [torch.from_numpy(o).to(device) for o in ort_outputs]
            head_outs = model.result_deserialize(outs_torch)
            # 穿透 DataContainer/list/tuple 拿到单个 img_meta 字典
            meta = _unwrap_img_meta(data['img_metas'])
            bbox_list = model.pts_bbox_head.get_bboxes(
                head_outs, [meta], rescale=True)
            # bbox_list: [[bboxes, scores, labels]]

            # 统计空检测（score threshold 可能过高导致全部过滤）
            # bbox_list[0][0] 是 LiDARInstance3DBoxes 类型，用 len() 获取数量
            num_dets = len(bbox_list[0][0]) if len(bbox_list[0]) > 0 else 0
            if num_dets == 0:
                empty_count += 1

            result_entry = {
                'pts_bbox': bbox3d2result(*bbox_list[0])
            }
            results.append(result_entry)

    # ---- 耗时统计 ----
    n = len(dataset)
    print(f'\nInference done: {n} samples')
    print(f'  coors avg: {total_coors_ms / n:.1f} ms/sample')
    print(f'  ONNX  avg: {total_ort_ms / n:.1f} ms/sample')
    print(f'  total avg: {(total_coors_ms + total_ort_ms) / n:.1f} ms/sample')

    # 空检测帧统计
    if empty_count > 0:
        print(f'⚠️  WARNING: {empty_count}/{n} frames have ZERO detections '
              f'({100 * empty_count / n:.1f}%)')
        if empty_count == n:
            print('  Possible causes:')
            print('    1. ONNX Runtime using CPU with FP16 model → '
                  'install onnxruntime-gpu')
            print('    2. Score threshold too high → check test_cfg.pts '
                  'score_threshold')
            print('    3. ONNX model output format mismatch → re-export ONNX')
            print('    4. Try adding --checkpoint '
                  'work_dirs/fastbev-r50-cbgs-custom/latest.pth '
                  'to ensure proper model init')

    # ---- 格式化输出 / 评估 ----
    kwargs = {} if args.eval_options is None else args.eval_options
    if args.format_only:
        dataset.format_results(results, **kwargs)

    if args.eval:
        eval_kwargs = cfg.get('evaluation', {}).copy()
        for key in ['interval', 'tmpdir', 'start', 'gpu_collect',
                    'save_best', 'rule']:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        print(dataset.evaluate(results, **eval_kwargs))


if __name__ == '__main__':
    main()
