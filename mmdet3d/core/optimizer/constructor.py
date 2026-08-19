# Copyright (c) OpenMMLab. All rights reserved.
"""自定义 optimizer constructor。

用于部分分支冻结（fine-tune）的训练场景：模型构建时已将冻结参数的
requires_grad 置为 False，此构造器保证优化器只包含 requires_grad=True
的可训练参数，避免为冻结参数维护优化器状态（exp_avg/exp_avg_sq）。
"""
import torch.nn as nn

from mmcv.runner.optimizer import (OPTIMIZER_BUILDERS, OPTIMIZERS,
                                   DefaultOptimizerConstructor)
from mmcv.utils import build_from_cfg


@OPTIMIZER_BUILDERS.register_module()
class RequiresGradOptimizerConstructor(DefaultOptimizerConstructor):
    """只将 requires_grad=True 的参数加入优化器。

    与模型侧的冻结逻辑配合使用，例如方案 B（冻结 FastBEV 检测分支、
    仅微调 FastBEVOCCHead 的 final_conv + predicter）。若不使用本构造器，
    mmcv 的 DefaultOptimizerConstructor 会把冻结参数（requires_grad=False）
    也放入优化器（尽管 AdamW 不会更新无梯度的参数，但会多维护一组参数）。
    """

    def __call__(self, model: nn.Module):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer_cfg['params'] = params
        return build_from_cfg(optimizer_cfg, OPTIMIZERS)
