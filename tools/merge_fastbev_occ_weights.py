# Copyright (c) OpenMMLab. All rights reserved.
"""合并 FastBEV 检测权重 + FlashOCC occ 权重，生成 fastbev-occ 联合初始化权重。

背景：
    fastbev-occ-r50-cbgs.py 的模型（FastBEVOCC）= FastBEV 检测分支 + occ_head。
    两份预训练权重天然互补：
      - FastBEV 权重（work_dirs/fastbev-r50-cbgs/epoch_20_ema.pth）
        含 img_backbone/img_neck/img_view_transformer/img_bev_encoder_*/pts_bbox_head，
        不含 occ_head；
      - FlashOCC 权重（data/ckpts/flashocc-r50-256x704.pth）
        含 occ_head（final_conv/predicter，结构与 FastBEVOCCHead 一致），
        不含 pts_bbox_head（其 view transformer 是 LSS 式，与 FastBEV 的
        FastrayTransformer 不同，故不采用其检测相关权重）。

用法（方式 3，推荐：检测 + occ 双分支都初始化）：
    python tools/merge_fastbev_occ_weights.py \
        --fastbev-ckpt work_dirs/fastbev-r50-cbgs/epoch_20_ema.pth \
        --flashocc-ckpt data/ckpts/flashocc-r50-256x704.pth \
        --out work_dirs/fastbev-occ-r50-cbgs/init_merged.pth

用法（方式 2：只提取 FlashOCC 的 occ_head，检测分支随机初始化）：
    python tools/merge_fastbev_occ_weights.py --only-occ-head \
        --flashocc-ckpt data/ckpts/flashocc-r50-256x704.pth \
        --out work_dirs/fastbev-occ-r50-cbgs/occ_head_init.pth

注意：不能直接把 FlashOCC 完整权重丢给 load_from！
其 img_view_transformer.depth_net 与 FastBEV 的 FastrayTransformer
shape 不匹配（152 vs 123），load_state_dict 会 raise size mismatch，
必须用 --only-occ-head 或合并脚本提取。

生成后，在配置中设置：
    load_from = 'work_dirs/fastbev-occ-r50-cbgs/init_merged.pth'
"""
import argparse

import torch

# FastBEV 检测分支的 state_dict 前缀（全部从 FastBEV 权重继承）
DET_PREFIXES = [
    'img_backbone', 'img_neck', 'img_view_transformer',
    'img_bev_encoder_backbone', 'img_bev_encoder_neck', 'pts_bbox_head'
]
# occ 分支前缀（从 FlashOCC 权重继承）
OCC_PREFIX = 'occ_head'


def parse_args():
    parser = argparse.ArgumentParser(description='Merge FastBEV + FlashOCC weights')
    parser.add_argument(
        '--fastbev-ckpt',
        default='work_dirs/fastbev-r50-cbgs/epoch_20_ema.pth',
        help='FastBEV 检测权重路径（--only-occ-head 时可省略）')
    parser.add_argument(
        '--flashocc-ckpt',
        default='data/ckpts/flashocc-r50-256x704.pth',
        help='FlashOCC 权重路径')
    parser.add_argument(
        '--out',
        default='work_dirs/fastbev-occ-r50-cbgs/init_merged.pth',
        help='合并后权重输出路径')
    parser.add_argument(
        '--only-occ-head',
        action='store_true',
        help='只提取 FlashOCC 的 occ_head（方式 2），检测分支保持随机初始化')
    return parser.parse_args()


def main():
    args = parse_args()

    flashocc = torch.load(args.flashocc_ckpt, map_location='cpu')
    flashocc_sd = flashocc.get('state_dict', flashocc)

    # FlashOCC occ 分支（两种模式都需要）
    occ_keys = [k for k in flashocc_sd if k.startswith(OCC_PREFIX)]
    if not occ_keys:
        raise RuntimeError('FlashOCC 权重中未找到 occ_head 键')

    merged = dict()

    if args.only_occ_head:
        # 方式 2：只提取 occ_head，检测分支随机初始化
        for k in occ_keys:
            merged[k] = flashocc_sd[k]
        print(f'[merge] (only-occ-head) FlashOCC occ 键: {len(occ_keys)}')
    else:
        # 方式 3：FastBEV 检测分支 + FlashOCC occ_head
        fastbev = torch.load(args.fastbev_ckpt, map_location='cpu')
        fastbev_sd = fastbev.get('state_dict', fastbev)

        det_keys = [k for k in fastbev_sd if k.split('.')[0] in DET_PREFIXES]
        missing_det = [p for p in DET_PREFIXES if not any(k.startswith(p) for k in det_keys)]
        if missing_det:
            raise RuntimeError(f'FastBEV 权重缺少检测分支前缀: {missing_det}')
        for k in det_keys:
            merged[k] = fastbev_sd[k]
        for k in occ_keys:
            merged[k] = flashocc_sd[k]

        unused_flashocc = sorted(set(flashocc_sd) - set(merged))
        print(f'[merge] FastBEV 检测键: {len(det_keys)}, FlashOCC occ 键: {len(occ_keys)}')
        print(f'[merge] 未采用的 FlashOCC 键 ({len(unused_flashocc)}): '
              f'{unused_flashocc[:5]} ...')

    sd_keys = set(merged)
    print(f'[merge] 合并后总键数: {len(sd_keys)}')

    # occ_head 形状自检（对照 FastBEVOCCHead: out_dim=256, Dz=16, num_classes=18）
    shapes = {k.split('.')[-2]: tuple(v.shape) for k, v in merged.items()
              if k.startswith(OCC_PREFIX)}
    print(f'[merge] occ_head 形状: {shapes}')

    meta = dict(
        fastbev_ckpt=args.fastbev_ckpt,
        flashocc_ckpt=args.flashocc_ckpt,
        only_occ_head=args.only_occ_head,
        note='FastBEV detection + FlashOCC occ_head merged init',
    )
    torch.save({'state_dict': merged, 'meta': meta}, args.out)
    print(f'[merge] 已保存: {args.out}')


if __name__ == '__main__':
    main()
