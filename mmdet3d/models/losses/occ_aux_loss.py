# Copyright (c) OpenMMLab. All rights reserved.
"""FlashOCC 风格的 occ 辅助损失（做成可配置 loss，便于消融实验）。

- LovaszSoftmaxLoss：直接优化 mIoU，对类别不平衡/小目标友好；
- SemScalLoss：语义尺度不变损失（sem_scal_loss）；
- GeoScalLoss：几何尺度不变损失（geo_scal_loss）。

参考 FlashOCC
projects/mmdet3d_plugin/models/losses/{lovasz_softmax,semkitti_loss}.py，
这里将输入统一为扁平化形式：pred logits (P, C)、target 标签 (P,)、
可选 weight/mask (P,)，适配 FastBEVOCCHead.loss 中已有的
preds/voxel_semantics/mask_camera 形态，无需再 permute 成 5D。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from ..builder import LOSSES


def inverse_sigmoid(x, sign='A'):
    """将 (0,1) 区间标量转回 logits，避免 BCE 数值不稳定。"""
    del sign
    x = x.to(torch.float32).clamp(min=1e-6, max=1 - 1e-6)
    return -torch.log((1 / x) - 1)


def lovasz_grad(gt_sorted):
    """Computes gradient of the Lovasz extension w.r.t sorted errors.

    See Alg. 1 in the Lovasz-Softmax paper.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels):
    """Multi-class Lovasz-Softmax on flattened predictions.

    Args:
        probas (torch.Tensor): (P, C) softmax probabilities.
        labels (torch.Tensor): (P,) ground truth labels.

    0.2m 体素下 P 可达数百万（400x400x16=256 万/帧），每类的 sort 都会产生
    (P,) 级中间张量，峰值叠加明显。为控制显存：
      1) 用 torch.unique 一次找出真实出现的类别，跳过缺失类的全量扫描；
      2) 每类算完立即释放临时张量（fg/errors/perm），只保留标量 loss。
    """
    if probas.numel() == 0:
        return probas.sum() * 0.
    C = probas.size(1)
    losses = []
    present = torch.unique(labels).tolist()  # classes='present'
    for c in present:
        if c < 0 or c >= C:  # 防御：越界/ignore 标签
            continue
        fg = (labels == c)
        if fg.sum() == 0:
            continue
        class_pred = probas[:, c]
        errors = (fg.float() - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm].float()
        del fg, class_pred, errors, perm  # 及时释放，降低单类峰值
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    if not losses:
        return probas.sum() * 0.
    return torch.stack(losses).mean()


def sem_scal_loss(pred, target, ignore_index=255):
    """Semantic scalability loss（与 FlashOCC 一致：遍历前 n_classes-1 类）。

    Args:
        pred (torch.Tensor): (P, C) logits.
        target (torch.Tensor): (P,) labels.
    """
    with autocast(False):
        pred = F.softmax(pred.float(), dim=1)
        if ignore_index is not None:
            valid = target != ignore_index
            pred = pred[valid]
            target = target[valid]
        loss = 0.
        count = 0.
        n_classes = pred.shape[1]
        for i in range(n_classes - 1):
            p = pred[:, i]
            completion_target = (target == i).float()
            if completion_target.sum() > 0:
                count += 1.
                nominator = (p * completion_target).sum()
                loss_class = 0.
                if p.sum() > 0:
                    precision = nominator / (p.sum() + 1e-5)
                    loss_class += F.binary_cross_entropy_with_logits(
                        inverse_sigmoid(precision), torch.ones_like(precision))
                if completion_target.sum() > 0:
                    recall = nominator / (completion_target.sum() + 1e-5)
                    loss_class += F.binary_cross_entropy_with_logits(
                        inverse_sigmoid(recall), torch.ones_like(recall))
                if (1 - completion_target).sum() > 0:
                    specificity = ((1 - p) * (1 - completion_target)).sum() / (
                        (1 - completion_target).sum() + 1e-5)
                    loss_class += F.binary_cross_entropy_with_logits(
                        inverse_sigmoid(specificity), torch.ones_like(specificity))
                loss += loss_class
        return loss / max(count, 1.)


def geo_scal_loss(pred, target, ignore_index=255, non_empty_idx=0):
    """Geometric scalability loss。

    Args:
        pred (torch.Tensor): (P, C) logits.
        target (torch.Tensor): (P,) labels.
        non_empty_idx (int): 空体素类别的索引（nuScenes occ 中 0 为 empty）。
    """
    with autocast(False):
        pred = F.softmax(pred.float(), dim=1)
        empty_probs = pred[:, non_empty_idx]
        nonempty_probs = 1 - empty_probs
        if ignore_index is not None:
            valid = target != ignore_index
            pred = pred[valid]
            target = target[valid]
            empty_probs = empty_probs[valid]
            nonempty_probs = nonempty_probs[valid]
        nonempty_target = (target != non_empty_idx).float()
        eps = 1e-5
        intersection = (nonempty_target * nonempty_probs).sum()
        precision = intersection / (nonempty_probs.sum() + eps)
        recall = intersection / (nonempty_target.sum() + eps)
        spec = ((1 - nonempty_target) * empty_probs).sum() / (
            (1 - nonempty_target).sum() + eps)
        return (
            F.binary_cross_entropy_with_logits(
                inverse_sigmoid(precision), torch.ones_like(precision))
            + F.binary_cross_entropy_with_logits(
                inverse_sigmoid(recall), torch.ones_like(recall))
            + F.binary_cross_entropy_with_logits(
                inverse_sigmoid(spec), torch.ones_like(spec)))


@LOSSES.register_module()
class LovaszSoftmaxLoss(nn.Module):
    """Lovasz-Softmax loss（直接优化 mIoU）。

    Args:
        ignore_index (int | None): 不参与损失统计的标签，默认 255。
        loss_weight (float): 损失权重（由 head 的 aux_loss_occ 配置传入）。
    """

    def __init__(self, ignore_index=255, loss_weight=1.0):
        super(LovaszSoftmaxLoss, self).__init__()
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self.loss_name = 'loss_occ_lovasz'

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Args:
            pred (torch.Tensor): (P, C) logits。
            target (torch.Tensor): (P,) long 标签。
            weight (torch.Tensor | None): (P,) 0/1 mask（如 mask_camera）。
        """
        if weight is not None:
            valid = weight.bool()
            pred = pred[valid]
            target = target[valid]
        if self.ignore_index is not None:
            valid = target != self.ignore_index
            pred = pred[valid]
            target = target[valid]
        probas = torch.softmax(pred.float(), dim=1)
        return self.loss_weight * lovasz_softmax_flat(probas, target)


@LOSSES.register_module()
class SemScalLoss(nn.Module):
    """Semantic scalability loss（尺度不变，对场景类边界更友好）。"""

    def __init__(self, ignore_index=255, loss_weight=1.0):
        super(SemScalLoss, self).__init__()
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self.loss_name = 'loss_occ_sem_scal'

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None:
            valid = weight.bool()
            pred = pred[valid]
            target = target[valid]
        if self.ignore_index is not None:
            valid = target != self.ignore_index
            pred = pred[valid]
            target = target[valid]
        return self.loss_weight * sem_scal_loss(
            pred, target, ignore_index=None)


@LOSSES.register_module()
class GeoScalLoss(nn.Module):
    """Geometric scalability loss（空/非空几何一致性）。

    Args:
        non_empty_idx (int): 空体素类索引，默认 0（nuScenes occ 的 empty）。
            注：FlashOCC BEVOCCHead2D_V2 中传 17，可据此对齐。
    """

    def __init__(self, ignore_index=255, non_empty_idx=0, loss_weight=1.0):
        super(GeoScalLoss, self).__init__()
        self.ignore_index = ignore_index
        self.non_empty_idx = non_empty_idx
        self.loss_weight = loss_weight
        self.loss_name = 'loss_occ_geo_scal'

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None:
            valid = weight.bool()
            pred = pred[valid]
            target = target[valid]
        if self.ignore_index is not None:
            valid = target != self.ignore_index
            pred = pred[valid]
            target = target[valid]
        return self.loss_weight * geo_scal_loss(
            pred, target, ignore_index=None,
            non_empty_idx=self.non_empty_idx)
