# Copyright (c) OpenMMLab. All rights reserved.
"""
FastBEV TensorRT 推理脚本 (严格遵循 test.py 架构).

用法 (与 dist_test.sh 完全一致):
  ./tools/trt_test.sh configs/fastbev/paper/fastbev-r50-cbgs-custom.py \
      models/fastbev_fp16_fuse.engine 2 \
      --format-only \
      --eval-options 'jsonfile_prefix=work_dirs/xxx'

神经网络推理完全由 TensorRT 执行，PyTorch 模型仅用于:
  1. 计算每帧的 FastRay 坐标 (coors_img / coors_depth)
  2. 对 TensorRT 输出进行 bbox 解码 (decode + NMS)
"""
import argparse
import os
import warnings

import mmcv
import tensorrt as trt
import torch
import torch.nn as nn
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint)

import mmdet
from mmdet3d.apis import single_gpu_test
from mmdet3d.core.bbox import bbox3d2result
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.models.detectors.base import Base3DDetector
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor

from types import MethodType as _MethodType

if mmdet.__version__ > '2.23.0':
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ═══════════════════════════════════════════════════════════════════════
# TensorRT 推理引擎封装
# ═══════════════════════════════════════════════════════════════════════

class TrtInfer:
    """TensorRT engine 推理封装 (固定 shape 快速路径 + 缓存优化)."""

    def __init__(self, engine_path, gpu_id=0):
        self.device_id = gpu_id

        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f'Failed to deserialize engine: {engine_path}')

        self.cfx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=f'cuda:{gpu_id}')
        self._addrs_bound = False

        # 收集 I/O tensor 元信息
        self.input_names = []
        self.output_names = []
        self.input_shapes = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                self.input_shapes[name] = shape
            else:
                self.output_names.append(name)

        # 检测固定 shape 引擎 → 快速路径
        self._fixed_shape = all(
            all(d != -1 for d in self.input_shapes[name])
            for name in self.input_names
        )
        if self._fixed_shape:
            for name in self.input_names:
                self.cfx.set_input_shape(name, self.input_shapes[name])

        self._alloc_buffers()

        # 缓存输出 shape (避免每帧 get_tensor_shape)
        self._output_shapes_cache = {}
        if self._fixed_shape:
            for name in self.output_names:
                self._output_shapes_cache[name] = tuple(
                    self.cfx.get_tensor_shape(name))

    def _dtype_to_torch(self, trt_dtype):
        mapping = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int64: torch.int64,
        }
        return mapping.get(trt_dtype, torch.float32)

    def _alloc_buffers(self):
        self._gpu_buffers = {}
        self._addrs_bound = False
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.cfx.get_tensor_shape(name)
            if any(d == -1 for d in shape):
                continue
            dtype = self.engine.get_tensor_dtype(name)
            nelem = trt.volume(shape)
            self._gpu_buffers[name] = torch.empty(
                nelem, dtype=self._dtype_to_torch(dtype),
                device=f'cuda:{self.device_id}')

    def _configure_shapes(self, feed_dict):
        """固定 shape 引擎直接跳过 (消除每帧 TRT API 开销)."""
        if self._fixed_shape:
            return
        need_realloc = False
        for name, tensor in feed_dict.items():
            shape = tuple(tensor.shape)
            if tuple(self.cfx.get_tensor_shape(name)) != shape:
                self.cfx.set_input_shape(name, shape)
                need_realloc = True
        if need_realloc:
            self._alloc_buffers()

    def infer(self, feed_dict):
        """执行一次推理.

        Args:
            feed_dict: {name: torch.Tensor} 输入 tensor (在 GPU 上)

        Returns:
            list[torch.Tensor]: 输出 tensor (在 GPU 上)
        """
        if self._fixed_shape:
            # 固定 shape 快速路径: 跳过 shape 查询
            for name in self.input_names:
                tensor = feed_dict[name]
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                self._gpu_buffers[name][:tensor.numel()].copy_(
                    tensor.flatten())

            if not self._addrs_bound:
                for i in range(self.engine.num_io_tensors):
                    n = self.engine.get_tensor_name(i)
                    self.cfx.set_tensor_address(
                        n, self._gpu_buffers[n].data_ptr())
                self._addrs_bound = True

            self.cfx.execute_async_v3(self.stream.cuda_stream)
            self.stream.synchronize()

            outputs = []
            for name in self.output_names:
                shape = self._output_shapes_cache[name]
                n = trt.volume(shape)
                outputs.append(self._gpu_buffers[name][:n].view(shape))
            return outputs

        # 动态 shape 路径
        self._configure_shapes(feed_dict)
        for name in self.input_names:
            tensor = feed_dict[name]
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            self._gpu_buffers[name][:tensor.numel()].copy_(tensor.flatten())

        if not self._addrs_bound:
            for i in range(self.engine.num_io_tensors):
                n = self.engine.get_tensor_name(i)
                self.cfx.set_tensor_address(
                    n, self._gpu_buffers[n].data_ptr())
            self._addrs_bound = True

        self.cfx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        outputs = []
        for name in self.output_names:
            shape = tuple(self.cfx.get_tensor_shape(name))
            n = trt.volume(shape)
            outputs.append(self._gpu_buffers[name][:n].view(shape))
        return outputs


# ═══════════════════════════════════════════════════════════════════════
# 参数解析 (与 test.py 完全一致, checkpoint → engine)
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description='FastBEV TensorRT test (and eval)')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('engine', help='TensorRT engine file (*.engine)')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options',
                      DeprecationWarning)
        args.eval_options = args.options
    return args


# ═══════════════════════════════════════════════════════════════════════
# 主函数 (严格遵循 test.py 结构)
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # 与 test.py 一致: 重新设置 gpu_ids
        _, world_size = get_dist_info()
        cfg.gpu_ids = list(range(world_size))

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # ── 构建 FastBEVTRT 模型 (提供 result_deserialize + get_fastray_input) ──
    cfg.model.type = cfg.model.type + 'TRT'
    cfg.model.img_view_transformer.accelerate = True
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    # 加载 PyTorch checkpoint (获取 CLASSES / PALETTE 元信息)
    _checkpoint = cfg.get('load_from', None)
    if _checkpoint and os.path.exists(_checkpoint):
        checkpoint = load_checkpoint(model, _checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES
        if 'PALETTE' in checkpoint.get('meta', {}):
            model.PALETTE = checkpoint['meta']['PALETTE']
        elif hasattr(dataset, 'PALETTE'):
            model.PALETTE = dataset.PALETTE
    else:
        model.CLASSES = dataset.CLASSES
        if hasattr(dataset, 'PALETTE'):
            model.PALETTE = dataset.PALETTE

    rank, _ = get_dist_info()
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # ── 加载 TensorRT engine ──
    trt_infer = TrtInfer(args.engine, gpu_id=rank)

    # ── 恢复标准 forward (FastBEVTRT 覆盖了 Base3DDetector.forward) ──
    # Base3DDetector.forward: return_loss=True → forward_train, else → forward_test
    # 与 test.py 的数据流严格一致
    model.forward = _MethodType(Base3DDetector.forward, model)

    # ── 初始化 dummy_feat (用于 get_fastray_input 坐标计算) ──
    vt_cfg = model.img_view_transformer
    if hasattr(vt_cfg, 'feature_size'):
        feat_h, feat_w = vt_cfg.feature_size
    else:
        img_h, img_w = vt_cfg.image_size
        stride = getattr(vt_cfg, 'stride', 16)
        feat_h, feat_w = img_h // stride, img_w // stride
    model._dummy_feat = torch.zeros(1, 6, 1, feat_h, feat_w,
                                    device=device, requires_grad=False)

    # ── 注入 TRT engine ──
    model._trt_infer = trt_infer

    # ── 覆盖 extract_feat: 使用 TRT engine 替代 PyTorch 特征提取 ──
    def _extract_feat_trt(self, points, img, img_metas, **kwargs):
        """TRT 推理管线: FastRay 坐标 → TRT engine → head 输出反序列化."""
        # img[0]: 原始图像 (B, N, C, H, W) , img[1:]: 6 个相机参数 tensor
        img_raw = img[0]
        # 确保 dummy_feat 与输入 tensor 在同一设备 (MMDDP 可能改变设备)
        if self._dummy_feat.device != img_raw.device:
            self._dummy_feat = self._dummy_feat.to(img_raw.device)
        coors_input = [self._dummy_feat] + list(img[1:])
        _, coors_img, coors_depth = \
            self.img_view_transformer.get_fastray_input(coors_input)
        feed = {
            self._trt_infer.input_names[0]: img_raw.float(),
            self._trt_infer.input_names[1]: coors_img[0].long(),
            self._trt_infer.input_names[2]: coors_depth[0].long(),
        }
        trt_outputs = self._trt_infer.infer(feed)
        head_outs = self.result_deserialize(trt_outputs)
        return head_outs, None, None  # (img_feats, pts_feats, depth)

    model.extract_feat = _MethodType(_extract_feat_trt, model)

    # ── 覆盖 simple_test_pts: 跳过 pts_bbox_head(x), 直接用 TRT 输出解码 ──
    def _simple_test_pts_trt(self, x, img_metas, rescale=False):
        """x 已是 TRT head 输出 (来自 _extract_feat_trt), 直接 get_bboxes."""
        bbox_list = self.pts_bbox_head.get_bboxes(
            x, img_metas, rescale=rescale)
        bbox_results = [bbox3d2result(bboxes, scores, labels)
                        for bboxes, scores, labels in bbox_list]
        return bbox_results

    model.simple_test_pts = _MethodType(_simple_test_pts_trt, model)

    if rank == 0:
        print(f'Config : {args.config}')
        print(f'Engine : {args.engine}')
        print(f'TRT inputs : {trt_infer.input_names}')
        print(f'TRT outputs: {len(trt_infer.output_names)} tensors')
        print(f'Dataset    : {len(dataset)} samples')
        print(f'Distributed: {distributed}, '
              f'world_size: {cfg.gpu_ids}')

    # ── 推理 (与 test.py 严格一致: MMDDP 包装 + multi_gpu_test) ──
    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show,
                                  args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    # ── 结果保存 & 评估 (仅 rank 0) ──
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)

        kwargs = {} if args.eval_options is None else args.eval_options

        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
