# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from ..builder import HEADS, build_loss

nusc_class_frequencies = np.array([
    944004, 1897170, 152386, 2391677, 16957802, 724139, 189027, 2074468,
    413451, 2384460, 5916653, 175883646, 4275424, 51393615, 61411620,
    105975596, 116424404, 1892500630
])


@HEADS.register_module()
class FastBEVOCCHead(BaseModule):
    """Occupancy head for FastBEV.

    FastBEV 输出 2D BEV 特征 (B, C, Dy, Dx)，空间范围由 bev_range 决定
    （fastbev 默认 x/y 均在 [-51.2, 51.2]，分辨率 0.8m，即 128x128）。
    本 head 首先通过可微的 grid_sample 将 BEV 特征重采样到 occ 分辨率
    （默认 x/y 在 [-40, 40]，0.4m，即 200x200，与 nuScenes occ 标注对齐），
    然后经过 final_conv + predicter 输出语义占用预测
    occ_pred: (B, Dx, Dy, Dz, num_classes)。

    属性命名（final_conv / predicter / Dz / num_classes）与 FlashOCC 的
    BEVOCCHead2D 保持一致，从而可以加载 flashocc-r50-256x704.pth 中
    的 occ_head 权重作为初始化。
    """

    def __init__(self,
                 in_dim=256,
                 out_dim=256,
                 Dz=16,
                 use_mask=True,
                 num_classes=18,
                 use_predicter=True,
                 class_balance=False,
                 loss_occ=None,
                 aux_loss_occ=None,
                 bev_size=(128, 128),
                 occ_size=(200, 200),
                 bev_range=(-51.2, 51.2),
                 occ_range=(-40.0, 40.0),
                 init_cfg=None):
        super(FastBEVOCCHead, self).__init__(init_cfg=init_cfg)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz
        self.bev_size = bev_size
        self.occ_size = occ_size

        out_channels = out_dim if use_predicter else num_classes * Dz
        self.final_conv = ConvModule(
            self.in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d'))

        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes * Dz),
            )

        self.use_mask = use_mask
        self.num_classes = num_classes
        self.class_balance = class_balance
        if self.class_balance:
            class_weights = torch.from_numpy(
                1 / np.log(nusc_class_frequencies[:num_classes] + 0.001))
            self.cls_weights = class_weights
            if loss_occ is not None:
                loss_occ['class_weight'] = class_weights

        if loss_occ is None:
            loss_occ = dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        self.loss_occ = build_loss(loss_occ)

        # FlashOCC V2 风格的辅助损失（Lovász/sem_scal/geo_scal），配置开关：
        #   aux_loss_occ=[dict(type='LovaszSoftmaxLoss', loss_weight=1.0),
        #                 dict(type='SemScalLoss', loss_weight=1.0),
        #                 dict(type='GeoScalLoss', loss_weight=1.0,
        #                      non_empty_idx=0)]
        # 传 None 或空列表即只保留主 loss_occ（CrossEntropyLoss）。
        self.aux_loss_occ = []
        if aux_loss_occ is not None:
            for cfg in aux_loss_occ:
                self.aux_loss_occ.append(build_loss(dict(cfg)))

        # 预计算 BEV -> occ 的重采样网格（buffer，不参与梯度）
        self.register_buffer(
            'bev2occ_grid',
            self._build_grid(bev_size, occ_size, bev_range, occ_range),
            persistent=False)

    def _build_grid(self, bev_size, occ_size, bev_range, occ_range):
        """构建 grid_sample 采样网格。

        BEV 特征 (Dy, Dx)，每格覆盖 bev_range 范围；
        occ 输出 (Dy', Dx')，每格覆盖 occ_range 范围。
        以格子中心对齐，将 occ 输出格的格心映射回 BEV 特征坐标。
        """
        # BEV 与 occ 同分辨率同范围时无需重采样（返回 None，forward 跳过）
        if (bev_size == occ_size) and (bev_range == occ_range):
            return None

        by, bx = bev_size  # (H, W) = (Dy, Dx)
        oy, ox = occ_size
        bev_min, bev_max = bev_range
        occ_min, occ_max = occ_range

        # FastBEV（FastrayTransformer）中 BEV 格 i 的中心坐标为：
        #   grid_lower_bound + (i - 0.5) * grid_interval
        # 即像素 i 的物理中心 = bev_min + (i - 0.5) * step
        step = (bev_max - bev_min) / bx
        first_center = bev_min - step / 2

        # occ 输出格的格心坐标（米）
        xc = occ_min + (torch.arange(ox).float() + 0.5) * (occ_max - occ_min) / ox
        yc = occ_min + (torch.arange(oy).float() + 0.5) * (occ_max - occ_min) / oy

        # occ 格心 -> BEV 特征像素浮点坐标（像素 p 的物理中心 = bev_min + (p - 0.5) * step）
        ux = (xc - first_center) / step
        uy = (yc - first_center) / step

        # 像素坐标 -> grid_sample 归一化坐标（align_corners=True：像素 p <-> grid = p/(W-1)*2 - 1）
        gx = ux / (bx - 1) * 2 - 1
        gy = uy / (by - 1) * 2 - 1

        gy, gx = torch.meshgrid(gy, gx)  # (oy, ox)
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # (1, oy, ox, 2)
        return grid

    def forward(self, img_feats):
        """img_feats: (B, C, Dy, Dx) -> occ_pred: (B, Dx, Dy, Dz, num_classes)"""
        bs, C, H, W = img_feats.shape
        if self.bev2occ_grid is not None:
            # grid 是 float32 buffer；fp16/AMP 训练时需与输入特征同 dtype
            grid = self.bev2occ_grid.to(dtype=img_feats.dtype).expand(bs, -1, -1, -1)
            img_feats = F.grid_sample(
                img_feats, grid, mode='bilinear', align_corners=True)
        # (B, C, Dy, Dx) --> (B, C, Dy', Dx') --> (B, Dx', Dy', C)
        occ_pred = self.final_conv(img_feats).permute(0, 3, 2, 1)
        bs, Dx, Dy = occ_pred.shape[:3]
        if self.use_predicter:
            # (B, Dx, Dy, C) --> (B, Dx, Dy, 2*C) --> (B, Dx, Dy, Dz*n_cls)
            occ_pred = self.predicter(occ_pred)
            occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes)
        return occ_pred

    def loss(self, occ_pred, voxel_semantics, mask_camera):
        """Args:
            occ_pred: (B, Dx, Dy, Dz, num_classes)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        """
        loss = dict()
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)  # (B, Dx, Dy, Dz)
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = occ_pred.reshape(-1, self.num_classes)
            mask_camera = mask_camera.reshape(-1)

            if self.class_balance:
                valid_voxels = voxel_semantics[mask_camera.bool()]
                num_total_samples = 0
                for i in range(self.num_classes):
                    num_total_samples += \
                        (valid_voxels == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = mask_camera.sum()

            loss_occ = self.loss_occ(
                preds, voxel_semantics, mask_camera,
                avg_factor=num_total_samples)
            loss['loss_occ'] = loss_occ
            loss.update(
                self._calc_aux_losses(preds, voxel_semantics, mask_camera))
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = occ_pred.reshape(-1, self.num_classes)
            if self.class_balance:
                num_total_samples = 0
                for i in range(self.num_classes):
                    num_total_samples += \
                        (voxel_semantics == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = len(voxel_semantics)
            loss_occ = self.loss_occ(
                preds, voxel_semantics, avg_factor=num_total_samples)
            loss['loss_occ'] = loss_occ
            loss.update(self._calc_aux_losses(preds, voxel_semantics))
        return loss

    def _calc_aux_losses(self, preds, voxel_semantics, mask_camera=None):
        """计算 FlashOCC V2 风格的辅助 occ 损失。

        Args:
            preds (torch.Tensor): (P, num_classes) logits。
            voxel_semantics (torch.Tensor): (P,) long 标签。
            mask_camera (torch.Tensor | None): (P,) 0/1 相机可见 mask。
        """
        loss = dict()
        if not self.aux_loss_occ:
            return loss
        kwargs = dict()
        if mask_camera is not None:
            kwargs['weight'] = mask_camera
        for loss_fn in self.aux_loss_occ:
            loss[loss_fn.loss_name] = loss_fn(
                preds, voxel_semantics, **kwargs)
        return loss

    def get_occ(self, occ_pred, img_metas=None):
        """返回 list[np.ndarray]，每个形状 (Dx, Dy, Dz)"""
        occ_score = occ_pred.softmax(-1)  # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)  # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)
        return list(occ_res)
