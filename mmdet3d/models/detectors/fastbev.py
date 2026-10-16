# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from .. import builder
from .centerpoint import CenterPoint
from mmdet3d.models.utils.grid_mask import GridMask
from mmdet.models.backbones.resnet import ResNet


@DETECTORS.register_module()
class FastBEV(CenterPoint):
    r"""FastBEV paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2112.11790>`_

    Args:
        img_view_transformer (dict): Configuration dict of view transformer.
        img_bev_encoder_backbone (dict): Configuration dict of the BEV encoder
            backbone.
        img_bev_encoder_neck (dict): Configuration dict of the BEV encoder neck.
    """

    def __init__(self,
                 img_view_transformer,
                 img_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None,
                 use_grid_mask=False,
                 use_depth=False,
                 **kwargs):
        super(FastBEV, self).__init__(**kwargs)
        self.grid_mask = None if not use_grid_mask else \
            GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1,
                     prob=0.7)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        if img_bev_encoder_neck and img_bev_encoder_backbone:
            self.img_bev_encoder_backbone = \
                builder.build_backbone(img_bev_encoder_backbone)
            self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)
        self.use_depth = use_depth

    def image_encoder(self, img, stereo=False):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        if self.grid_mask is not None:
            imgs = self.grid_mask(imgs)
        x = self.img_backbone(imgs)
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x, stereo_feat

    @force_fp32()
    def bev_encoder(self, x, return_multi_level=False):
        """BEV 编码器。

        Args:
            x (torch.Tensor): 输入 BEV 特征 [B, C, H, W]
            return_multi_level (bool): 是否返回 backbone 多尺度特征

        Returns:
            torch.Tensor | tuple: 最终特征 或 (multi_feats, final_feat)
        """
        multi_feats = self.img_bev_encoder_backbone(x)
        if return_multi_level and isinstance(multi_feats, (list, tuple)):
            neck_input = list(multi_feats)
            x = self.img_bev_encoder_neck(neck_input)
            if type(x) in [list, tuple]:
                x = x[0]
            return list(multi_feats), x
        x = self.img_bev_encoder_neck(multi_feats)
        if type(x) in [list, tuple]:
            x = x[0]
        return x

    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        assert len(inputs) == 7
        B, N, C, H, W = inputs[0].shape
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs

        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0,  ...].unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda]

    def extract_img_feat(self, img, img_metas, return_multi_level=False, **kwargs):
        """Extract features of images."""
        img = self.prepare_inputs(img)
        x, _ = self.image_encoder(img[0])
        x, depth = self.img_view_transformer([x] + img[1:8])
        if return_multi_level:
            multi_feats, final_feat = self.bev_encoder(x, return_multi_level=True)
            return [final_feat], depth, multi_feats
        x = self.bev_encoder(x)
        return [x], depth

    def extract_feat(self, points, img, img_metas, return_multi_level=False, **kwargs):
        """Extract features from images and points."""
        if return_multi_level:
            img_feats, depth, img_multi = self.extract_img_feat(
                img, img_metas, return_multi_level=True, **kwargs)
            return (img_feats, None, depth, img_multi)
        img_feats, depth = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, depth)

    def multi_level_fuse(self, img_multi, pts_multi):
        """多层级融合：对 backbone 各 stage 输出做门控融合。

        - 非级联（cascade_fusion=False）：仅对 neck 实际消费的层级
          （self.fusion_level_ids）做融合，其余层级原样透传，保证每个
          level_fusions 模块都被消费、参数都有梯度（DDP 安全）。
        - 级联（cascade_fusion=True）：为**所有**层级做融合，且 level i 额外
          消费上一层融合结果（self.level_downs 下采样 + 对齐通道后）作为第三路
          输入。这样 neck 未直接消费的中间层（如 level1）也能经下一层回传拿到
          梯度，同时激光特征由细到粗逐级流动。

        Args:
            img_multi (list[torch.Tensor]): 图像 BEV 多尺度特征
            pts_multi (list[torch.Tensor]): 激光 BEV 多尺度特征

        Returns:
            list[torch.Tensor]: 融合后的多尺度特征，送入 neck
        """
        # 层级 id -> 门控融合模块下标
        level2idx = {lid: i for i, lid in enumerate(self.fusion_level_ids)}
        cascade = getattr(self, 'cascade_fusion', False)
        fused_multi = []
        prev_fused = None
        for lid, (img_f, pts_f) in enumerate(zip(img_multi, pts_multi)):
            if lid not in level2idx:
                # 非级联模式下 neck 不消费该层级，直接透传避免引入无梯度参数
                fused_multi.append(img_f)
                continue
            idx = level2idx[lid]
            cat_feats = [img_f, pts_f]
            if cascade and prev_fused is not None:
                # 上一层融合结果下采样到当前分辨率并对齐通道，作为第三路输入
                cat_feats.append(self.level_downs[idx - 1](prev_fused))
            gate = self.level_fusions[idx](torch.cat(cat_feats, dim=1))
            img_gate = gate[:, 0:1, :, :]
            pts_gate = gate[:, 1:2, :, :]
            fused = img_gate * img_f + pts_gate * pts_f
            if len(cat_feats) > 2:
                fused = fused + gate[:, 2:3, :, :] * cat_feats[2]
            fused_multi.append(fused)
            prev_fused = fused
        return fused_multi

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        if self.use_depth:
            gt_depth = kwargs['gt_depth']
            loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
            losses = dict(loss_depth=loss_depth)
        else:
            losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        assert False

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        img_feats, _, _ = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(img_feats)
        return outs



@DETECTORS.register_module()
class FastBEV4D(FastBEV):
    r"""FastBEV4D paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2203.17054>`_

    Args:
        pre_process (dict | None): Configuration dict of BEV pre-process net.
        align_after_view_transfromation (bool): Whether to align the BEV
            Feature after view transformation. By default, the BEV feature of
            the previous frame is aligned during the view transformation.
        num_adj (int): Number of adjacent frames.
        with_prev (bool): Whether to set the BEV feature of previous frame as
            all zero. By default, False.
    """
    def __init__(self,
                 pre_process=None,
                 align_after_view_transfromation=False,
                 num_adj=1,
                 with_prev=True,
                 pre_grad=False,
                 **kwargs):
        super(FastBEV4D, self).__init__(**kwargs)
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)
        self.align_after_view_transfromation = align_after_view_transfromation
        self.num_frame = num_adj + 1

        self.with_prev = with_prev
        self.pre_grad = pre_grad
        self.grid = None

    def gen_grid(self, input, sensor2keyegos, bda, bda_adj=None):
        n, c, h, w = input.shape
        _, v, _, _ = sensor2keyegos[0].shape
        if self.grid is None:
            # generate grid
            xs = torch.linspace(
                0, w - 1, w, dtype=input.dtype,
                device=input.device).view(1, w).expand(h, w)
            ys = torch.linspace(
                0, h - 1, h, dtype=input.dtype,
                device=input.device).view(h, 1).expand(h, w)
            grid = torch.stack((xs, ys, torch.ones_like(xs)), -1)
            self.grid = grid
        else:
            grid = self.grid
        grid = grid.view(1, h, w, 3).expand(n, h, w, 3).view(n, h, w, 3, 1)

        # get transformation from current ego frame to adjacent ego frame
        # transformation from current camera frame to current ego frame
        c02l0 = sensor2keyegos[0][:, 0:1, :, :]

        # transformation from adjacent camera frame to current ego frame
        c12l0 = sensor2keyegos[1][:, 0:1, :, :]

        # add bev data augmentation
        bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
        # bda_[:, :, :3, :3] = bda.unsqueeze(1)
        bda_[:, :, :, :] = bda.unsqueeze(1)
        bda_[:, :, 3, 3] = 1
        c02l0 = bda_.matmul(c02l0)
        if bda_adj is not None:
            bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
            bda_[:, :, :3, :3] = bda_adj.unsqueeze(1)
            bda_[:, :, 3, 3] = 1
        c12l0 = bda_.matmul(c12l0)

        # transformation from current ego frame to adjacent ego frame
        l02l1 = c02l0.matmul(torch.inverse(c12l0))[:, 0, :, :].view(
            n, 1, 1, 4, 4)
        '''
          c02l0 * inv(c12l0)
        = c02l0 * inv(l12l0 * c12l1)
        = c02l0 * inv(c12l1) * inv(l12l0)
        = l02l1 # c02l0==c12l1
        '''

        l02l1 = l02l1[:, :, :,
                      [True, True, False, True], :][:, :, :, :,
                                                    [True, True, False, True]]

        feat2bev = torch.zeros((3, 3), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.img_view_transformer.grid_interval[0]
        feat2bev[1, 1] = self.img_view_transformer.grid_interval[1]
        feat2bev[0, 2] = self.img_view_transformer.grid_lower_bound[0]
        feat2bev[1, 2] = self.img_view_transformer.grid_lower_bound[1]
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1, 3, 3)
        tf = torch.inverse(feat2bev).matmul(l02l1).matmul(feat2bev)

        # transform and normalize
        grid = tf.matmul(grid)
        normalize_factor = torch.tensor([w - 1.0, h - 1.0],
                                        dtype=input.dtype,
                                        device=input.device)
        grid = grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1,
                                                            2) * 2.0 - 1.0
        return grid

    @force_fp32()
    def shift_feature(self, input, sensor2keyegos, bda, bda_adj=None):
        grid = self.gen_grid(input, sensor2keyegos, bda, bda_adj=bda_adj)
        output = F.grid_sample(input, grid.to(input.dtype), align_corners=True)
        return output

    def prepare_bev_feat(self, img, rot, tran, intrin, post_rot, post_tran,
                         bda, mlp_input):
        x, _ = self.image_encoder(img)
        bev_feat, depth = self.img_view_transformer(
            [x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input])
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth

    def extract_img_feat_sequential(self, inputs, feat_prev):
        imgs, sensor2keyegos_curr, ego2globals_curr, intrins = inputs[:4]
        sensor2keyegos_prev, _, post_rots, post_trans, bda = inputs[4:]
        bev_feat_list = []
        mlp_input = self.img_view_transformer.get_mlp_input(
            sensor2keyegos_curr[0:1, ...], ego2globals_curr[0:1, ...],
            intrins, post_rots, post_trans, bda[0:1, ...])
        inputs_curr = (imgs, sensor2keyegos_curr[0:1, ...],
                       ego2globals_curr[0:1, ...], intrins, post_rots,
                       post_trans, bda[0:1, ...], mlp_input)
        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
        bev_feat_list.append(bev_feat)

        # align the feat_prev
        _, C, H, W = feat_prev.shape
        feat_prev = \
            self.shift_feature(feat_prev,
                               [sensor2keyegos_curr, sensor2keyegos_prev],
                               bda)
        bev_feat_list.append(feat_prev.view(1, (self.num_frame - 1) * C, H, W))

        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth

    def prepare_inputs(self, inputs, stereo=False):
        # split the inputs into each frame
        B, N, C, H, W = inputs[0].shape
        N = N // self.num_frame
        imgs = inputs[0].view(B, N, self.num_frame, C, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs[1:7]

        sensor2egos = sensor2egos.view(B, self.num_frame, N, 4, 4)
        ego2globals = ego2globals.view(B, self.num_frame, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0, 0, ...].unsqueeze(1).unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        curr2adjsensor = None
        if stereo:
            sensor2egos_cv, ego2globals_cv = sensor2egos, ego2globals
            sensor2egos_curr = \
                sensor2egos_cv[:, :self.temporal_frame, ...].double()
            ego2globals_curr = \
                ego2globals_cv[:, :self.temporal_frame, ...].double()
            sensor2egos_adj = \
                sensor2egos_cv[:, 1:self.temporal_frame + 1, ...].double()
            ego2globals_adj = \
                ego2globals_cv[:, 1:self.temporal_frame + 1, ...].double()
            curr2adjsensor = \
                torch.inverse(ego2globals_adj @ sensor2egos_adj) \
                @ ego2globals_curr @ sensor2egos_curr
            curr2adjsensor = curr2adjsensor.float()
            curr2adjsensor = torch.split(curr2adjsensor, 1, 1)
            curr2adjsensor = [p.squeeze(1) for p in curr2adjsensor]
            curr2adjsensor.extend([None for _ in range(self.extra_ref_frames)])
            assert len(curr2adjsensor) == self.num_frame

        extra = [
            sensor2keyegos,
            ego2globals,
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(B, self.num_frame, N, 3, 3),
            post_trans.view(B, self.num_frame, N, 3),
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        sensor2keyegos, ego2globals, intrins, post_rots, post_trans = extra
        return imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
               bda, curr2adjsensor

    def extract_img_feat(self,
                         img,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            return self.extract_img_feat_sequential(img, kwargs['feat_prev'])
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, _ = self.prepare_inputs(img)
        """Extract features of images."""
        bev_feat_list = []
        depth_list = []
        key_frame = True  # back propagation for key frame only
        for img, sensor2keyego, ego2global, intrin, post_rot, post_tran in zip(
                imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans):
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin, post_rot, post_tran, bda)
                inputs_curr = (img, sensor2keyego, ego2global, intrin, post_rot,
                               post_tran, bda, mlp_input)
                if key_frame:
                    bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
                else:
                    # TODO: bug here, need fixing
                    if self.pre_grad:
                        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
                    else:
                        with torch.no_grad():
                            bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
            else:
                bev_feat = torch.zeros_like(bev_feat_list[0])
                depth = None
            bev_feat_list.append(bev_feat)
            depth_list.append(depth)
            key_frame = False
        if pred_prev:
            assert self.align_after_view_transfromation
            assert sensor2keyegos[0].shape[0] == 1
            feat_prev = torch.cat(bev_feat_list[1:], dim=0)
            ego2globals_curr = \
                ego2globals[0].repeat(self.num_frame - 1, 1, 1, 1)
            sensor2keyegos_curr = \
                sensor2keyegos[0].repeat(self.num_frame - 1, 1, 1, 1)
            ego2globals_prev = torch.cat(ego2globals[1:], dim=0)
            sensor2keyegos_prev = torch.cat(sensor2keyegos[1:], dim=0)
            bda_curr = bda.repeat(self.num_frame - 1, 1, 1)
            return feat_prev, [imgs[0],
                               sensor2keyegos_curr, ego2globals_curr,
                               intrins[0],
                               sensor2keyegos_prev, ego2globals_prev,
                               post_rots[0], post_trans[0],
                               bda_curr]
        if self.align_after_view_transfromation:
            for adj_id in range(1, self.num_frame):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth_list[0]


@DETECTORS.register_module()
class FastBEVTRT(FastBEV):

    def result_serialize(self, outs):
        outs_ = []
        for out in outs:
            for key in ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']:
                outs_.append(out[0][key])
        return outs_

    def result_deserialize(self, outs):
        outs_ = []
        keys = ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']
        for head_id in range(len(outs) // 6):
            outs_head = [dict()]
            for kid, key in enumerate(keys):
                outs_head[0][key] = outs[head_id * 6 + kid]
            outs_.append(outs_head)
        return outs_

    def forward(
        self,
        img,
        coors_img,
        coors_depth=None
    ):
        img = self.img_backbone(img)
        img = self.img_neck(img)
        img = self.img_view_transformer.depth_net(img)

        channel = img.shape[1]
        img = img.permute(0, 2, 3, 1).reshape(-1, channel)
        img_0 = torch.zeros([1, img.shape[1]]).to(img.device)
        img_rest = img[1:]
        img = torch.cat([img_0, img_rest])

        if channel != self.img_view_transformer.out_channels:
            depth = img[:, :self.img_view_transformer.D].reshape(-1)
            if self.img_view_transformer.depth_act == 'sigmoid':
                depth = depth.sigmoid()
            else:
                depth = depth.softmax(dim=-1)
            x = img[:, self.img_view_transformer.D:(self.img_view_transformer.D + self.img_view_transformer.out_channels)]
            x = x[coors_img]
            depth = depth[coors_depth].unsqueeze(1)
            x = x * depth
        else:
            x = img[coors_img]
        x = x.view(1, *self.img_view_transformer.grid_size.int().tolist(), self.img_view_transformer.out_channels)
        N, X, Y, Z, C = x.shape

        if self.img_view_transformer.is_transpose:
            permute = [0, 3, 2, 1]
        else:
            permute = [0, 3, 1, 2]
        if self.img_view_transformer.fuse_type is not None:
            if self.img_view_transformer.fuse_type == 's2c':
                x = x.reshape(N, X, Y, Z*C).permute(permute)
                x = self.img_view_transformer.fuse(x)
            elif self.img_view_transformer.fuse_type == 'sum':
                x = x.sum(dim=-2).permute(permute)
            elif self.img_view_transformer.fuse_type == 'max':
                x = x.max(dim=-2)[0].permute(permute)
            else:
                raise NotImplemented
            x = self.img_view_transformer.downsample(x)

        bev_feat = self.bev_encoder(x)
        outs = self.pts_bbox_head([bev_feat])
        outs = self.result_serialize(outs)
        return outs

    def get_bev_pool_input(self, input):
        input = self.prepare_inputs(input)
        coor = self.img_view_transformer.get_lidar_coor(*input[1:7])
        return self.img_view_transformer.voxel_pooling_prepare_v2(coor)


@DETECTORS.register_module()
class FastBEVOCC(FastBEV):
    r"""FastBEV with an extra occupancy head (FlashOCC style).

    在 FastBEV 3D 目标检测的基础上，额外挂接语义占用头（FastBEVOCCHead），
    训练时同时优化检测损失与占用损失；测试时同时输出 BEV 3D 目标检测结果
    （pts_bbox）和语义占用栅格（occ）。

    分支解耦（det_downsample 非 None 时启用）：
    - BEV 特征为 200x200@0.4m（与 occ 栅格分辨率一致），occ 头直接消费原生
      0.4m 特征，不再做 0.8m->0.4m 插值上采样；
    - 检测分支先经 det_downsample（stride-2 卷积）下采样到 100x100
      （0.8m/像素），CenterHead 的 heatmap 尺寸与 target 编码保持兼容。

    Args:
        occ_head (dict | None): Configuration dict of the occupancy head.
        pts_occ_head (dict | None): 激光专用 OCC head 配置。若不为 None，
            从激光 BEV 特征直接预测 OCC，与主 OCC head 加权融合，
            提升几何边界精度。训练时两个 head 都有 loss 监督。
        occ_fusion_weight (float): 可学习融合权重的**初始值**：融合权重
            alpha = sigmoid(occ_fuse_logit)（occ_fuse_logit 为 nn.Parameter），
            训练中由主 OCC loss 的梯度自动学习，推理时按 alpha 融合主 OCC 与
            激光 OCC（alpha 初始 = occ_fusion_weight，默认 0.7）。
        det_downsample (dict | None): Configuration dict of the detection
            branch downsample layer. If None, the detection head consumes the
            original BEV feature directly. 支持可选的 'detach' 键
            （detach=True 时检测特征 detach()，用于消融"检测/occ 任务竞争"）。
        pts_voxel_layer (dict | None): 点云体素化层配置（PointPillars 风格）。
        pts_voxel_encoder (dict | None): 点云体素特征编码器配置。
        pts_middle_encoder (dict | None): 点云中间编码器配置（scatter 成伪图像）。
        pts_bev_encoder_backbone (dict | None): 激光 BEV backbone（可选，
            若为 None 则直接用 scatter 输出与图像 BEV 融合）。
        pts_bev_encoder_neck (dict | None): 激光 BEV neck（可选）。
        fusion (dict | None): BEV 融合方式配置。支持：
            - type='concat': 通道维拼接后用 1x1 conv 降维到 out_channels。
            - type='add': 逐元素相加（需两分支通道数相同）。
            - type='gated': 门控融合，对两分支分别预测门控权重再加权求和，
              配置需含 in_channels_img / in_channels_pts / out_channels /
              hidden_channels（可选，默认 64）。
            若为 None，则不启用激光分支，行为与纯视觉 FastBEVOCC 一致。
        multi_level_fusion (bool): 是否启用多层级融合。若为 True，在 BEV
            backbone 的每个 stage 输出后都做一次跨模态融合，再送入 neck。
            要求 img/pts 的 bev_encoder_backbone 都输出相同数量的多尺度特征
            （backbone_output_ids 长度一致），且对应 stage 通道数相同。
        cascade_fusion (bool): 是否启用级联门控融合。若为 True：
            1) 为 backbone 的**所有** stage 建门控融合（不再局限于 neck 消费层）；
            2) level i 的融合额外消费 level i-1 的融合结果（stride-2 卷积下采样
               并对齐通道后），使 neck 未直接消费的中间层（如 level1）也能通过
               level2 回传拿到梯度，DDP 安全，且激光信息由细到粗逐级流动。
            关闭时保持原行为：只为 neck 消费层建融合，其余层级原样透传。
        gate_bias_first (float): 门控输出层（nn.Conv2d(hid, N, 1)）bias 初值。
            第一路（图像/主分支）bias 设为 +b，其余路设为 -b；0 表示不启用
            （bias 全 0 → sigmoid=0.5，训练第 0 步就把随机初始化的新分支特征
            以 50% 权重掺进已收敛的预训练 BEV 特征，造成分布冲击）。
            推荐 2.0~3.0：初始输出 ≈ 原基线行为，训练中由网络自主学习抬高
            其他分支。作用于 level_fusions / gate_net / rev_gate_net
            （rev_gate_net 的第一路是激光分支，因此初始为激光主导）。
    """

    def __init__(self,
                 occ_head=None,
                 pts_occ_head=None,
                 occ_fusion_weight=0.7,
                 det_downsample=None,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_bev_encoder_backbone=None,
                 pts_bev_encoder_neck=None,
                 fusion=None,
                 reverse_fusion=None,
                 multi_level_fusion=False,
                 cascade_fusion=False,
                 gate_bias_first=0.0,
                 occ_alpha_half_res=False,
                 init_cfg=None,
                 **kwargs):
        super(FastBEVOCC, self).__init__(init_cfg=init_cfg, **kwargs)
        self.multi_level_fusion = multi_level_fusion
        self.cascade_fusion = cascade_fusion
        self.gate_bias_first = gate_bias_first
        # 方案B门控瘦身开关（默认关）：True 时门控在 100×100 算、bilinear 放大回 200
        self.gate_half_res = False
        self.rev_half_res = False
        self.occ_head = None
        if occ_head is not None:
            self.occ_head = builder.build_head(occ_head)
        # 激光专用 OCC head
        self.pts_occ_head = None
        if pts_occ_head is not None:
            self.pts_occ_head = builder.build_head(pts_occ_head)
            # 可学习融合权重：alpha = sigmoid(occ_fuse_logit)
            # 训练时把主 OCC 与激光 OCC 的融合结果直接送进 occ_head.loss，
            # 梯度经融合表达式回传到 occ_fuse_logit，alpha 自动学习最优权重；
            # occ_fusion_weight 仅作为 alpha 的初始值（默认 0.7 → 主 OCC 占 70%）
            import math
            init_logit = math.log(occ_fusion_weight / (1 - occ_fusion_weight))
            self.occ_fuse_logit = nn.Parameter(
                torch.tensor(init_logit, dtype=torch.float32))
            # 2D 空间 occ 融合门控：预测按 BEV 位置 (x,y) 的门控 logit，sigmoid 后在
            # (Dz, C) 维广播（occ_pred 布局为 B,Dx,Dy,Dz,C）。输入 = 纯相机 BEV || 激光
            # BEV（两路分支信号，非被加权的 logits → 无"抬置信度骗门控"的 rank-1 崩坏面）。
            # 随机初始化输出≈0 → 初始 alpha≈sigmoid(occ_fuse_logit)=occ_fusion_weight，
            # 起点行为≈原标量模型，训练中由主 OCC loss 逐步空间化（近场信激光、远场信图像）。
            from mmcv.cnn import ConvModule
            self.occ_alpha_head = nn.Sequential(
                ConvModule(
                    2 * self.pts_occ_head.in_dim, 128, 3, padding=1,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU')),
                nn.Conv2d(128, 1, 1))
            # 方案B：True 时此门控在 100×100 算、放大回 200（见 _gate2d）
            self.occ_alpha_half_res = occ_alpha_half_res
            # 诊断暂存：供 GateMonitorHook 读取
            self._occ_gate_stat = None
        self.det_downsample = None
        self.detach_det = False
        if det_downsample is not None:
            from mmcv.cnn import ConvModule
            self.detach_det = det_downsample.get('detach', False)
            self.det_downsample = nn.Sequential(
                ConvModule(
                    det_downsample['in_channels'],
                    det_downsample['out_channels'],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True))

        # ---------- 激光雷达分支（多模态融合） ----------
        self.with_lidar = fusion is not None
        self.pts_voxel_layer = None
        self.pts_voxel_encoder = None
        self.pts_middle_encoder = None
        self.pts_bev_encoder_backbone = None
        self.pts_bev_encoder_neck = None
        self.fusion = None
        self.rev_enabled = False

        if self.with_lidar:
            assert pts_voxel_layer is not None, \
                "启用 fusion 时必须提供 pts_voxel_layer"
            assert pts_voxel_encoder is not None, \
                "启用 fusion 时必须提供 pts_voxel_encoder"
            assert pts_middle_encoder is not None, \
                "启用 fusion 时必须提供 pts_middle_encoder"
            from mmdet3d.models.builder import build_voxel_encoder
            from mmdet3d.ops import Voxelization
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
            self.pts_voxel_encoder = build_voxel_encoder(pts_voxel_encoder)
            self.pts_middle_encoder = builder.build_middle_encoder(
                pts_middle_encoder)
            if pts_bev_encoder_backbone is not None:
                self.pts_bev_encoder_backbone = builder.build_backbone(
                    pts_bev_encoder_backbone)
            if pts_bev_encoder_neck is not None:
                self.pts_bev_encoder_neck = builder.build_neck(
                    pts_bev_encoder_neck)
            # 融合模块
            fusion_type = fusion.get('type', 'concat')
            if fusion_type == 'concat':
                self.fusion = nn.Conv2d(
                    fusion['in_channels'],
                    fusion['out_channels'],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False)
            elif fusion_type == 'add':
                self.fusion = nn.Identity()
            elif fusion_type == 'gated':
                # 门控融合：分别预测图像/激光分支的空间门控权重，加权求和
                from mmcv.cnn import ConvModule
                in_c_img = fusion['in_channels_img']
                in_c_pts = fusion['in_channels_pts']
                out_c = fusion['out_channels']
                hid_c = fusion.get('hidden_channels', 64)
                # 投影到相同通道数
                self.img_proj = nn.Conv2d(in_c_img, out_c, 1, bias=False)
                self.pts_proj = nn.Conv2d(in_c_pts, out_c, 1, bias=False)
                # 门控网络：concat 两分支特征 → 预测 2 通道门控权重
                # 方案B：half_res=True 时该门控在 100×100 算、放大回 200
                self.gate_half_res = fusion.get('half_res', False)
                self.gate_net = nn.Sequential(
                    ConvModule(out_c * 2, hid_c, 3, padding=1,
                               norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')),
                    nn.Conv2d(hid_c, 2, 1),
                    nn.Sigmoid(),
                )
                # 让模型从"图像主导"出发，保护预训练 BEV 特征（详见方法注释）
                self._init_gate_bias(self.gate_net)
                self.fusion = nn.Identity()  # 占位，实际融合在 fuse_bev_feats 中
            else:
                raise ValueError(f"Unsupported fusion type: {fusion_type}")
            self.fusion_type = fusion_type

            # ---- 反向融合（方案 A）：图像语义反哺激光分支 ----
            # 正向融合是「激光几何 → 图像语义分支」；这里再建一组结构对称的门控，
            # 用融合特征（已含图像语义）反过来调制激光 BEV 特征，输出送回激光分支
            # （当前用于 pts_occ_head）。激光有几何但缺语义、图像有语义但缺几何，
            # 双向互补。
            # 注意：该模块只在 pts_occ_head 存在时才会被调用，若配置了
            # reverse_fusion 但没建 pts_occ_head，其参数拿不到梯度，
            # DDP(find_unused_parameters=False) 会报 reduction 未完成。
            if reverse_fusion is not None:
                from mmcv.cnn import ConvModule
                rev_type = reverse_fusion.get('type', 'gated')
                if rev_type != 'gated':
                    raise ValueError(
                        f'Unsupported reverse fusion type: {rev_type}')
                rev_out_c = reverse_fusion['out_channels']
                rev_hid_c = reverse_fusion.get('hidden_channels', 64)
                self.rev_pts_proj = nn.Conv2d(
                    reverse_fusion['in_channels_pts'],
                    rev_out_c,
                    1,
                    bias=False)
                self.rev_img_proj = nn.Conv2d(
                    reverse_fusion['in_channels_img'],
                    rev_out_c,
                    1,
                    bias=False)
                # 方案B：rev 门控 half_res 开关
                self.rev_half_res = reverse_fusion.get('half_res', False)
                self.rev_gate_net = nn.Sequential(
                    ConvModule(
                        rev_out_c * 2,
                        rev_hid_c,
                        3,
                        padding=1,
                        norm_cfg=dict(type='BN'),
                        act_cfg=dict(type='ReLU')),
                    nn.Conv2d(rev_hid_c, 2, 1),
                    nn.Sigmoid(),
                )
                # 第一路是激光分支，因此初始为"激光主导"，图像语义由网络逐步引入
                self._init_gate_bias(self.rev_gate_net)
                self.rev_enabled = True

            # ---- 多层级融合：为 neck 实际消费的每个 stage 建门控融合单元 ----
            # 注意：只对 neck 真正消费的层级建融合模块。图像 neck（FPN_LSS）
            # 默认只用 input_feature_index 指定的层（如 (0, 2)），未被消费的
            # 层级若也建融合模块，其参数永远拿不到梯度，DDP 会直接报
            # "Expected to have finished reduction" 崩溃。
            if self.multi_level_fusion:
                from mmcv.cnn import ConvModule
                # 从 backbone 配置中读取各 stage 输出通道数
                img_ch = kwargs.get('img_bev_encoder_backbone', {}).get(
                    'num_channels', None)
                pts_ch = pts_bev_encoder_backbone.get('num_channels', None)
                assert img_ch is not None and pts_ch is not None, \
                    "多层级融合需要 backbone 配置 num_channels"
                assert len(img_ch) == len(pts_ch), \
                    f"图像/激光 backbone stage 数不一致: {len(img_ch)} vs {len(pts_ch)}"
                assert list(img_ch) == list(pts_ch), \
                    "门控融合按元素加权求和（g*img + g*pts），要求对应 stage " \
                    f"通道数一致: {img_ch} vs {pts_ch}"
                self.num_fusion_levels = len(img_ch)
                if self.cascade_fusion:
                    # 级联模式：所有层级都建门控融合。neck 未直接消费的中间层
                    # 通过"下一层消费上一层融合结果"回传梯度，DDP 安全，
                    # 且激光特征由细到粗逐级流动。
                    self.fusion_level_ids = list(range(self.num_fusion_levels))
                else:
                    # 读取 neck 实际消费的层级。注意必须从"已构建的 neck 实例"
                    # 读取：FPN_LSS 的 input_feature_index 有类默认值 (0, 2)，
                    # 配置里常常不显式写，读配置 dict 会拿到兜底值而误判。
                    neck = getattr(self, 'img_bev_encoder_neck', None)
                    if neck is not None and hasattr(neck, 'input_feature_index'):
                        consumed = neck.input_feature_index
                    else:
                        consumed = list(range(self.num_fusion_levels))
                    self.fusion_level_ids = sorted(set(int(i) for i in consumed))
                    for lid in self.fusion_level_ids:
                        assert 0 <= lid < self.num_fusion_levels, \
                            f"neck 消费的层级 {lid} 超出 backbone 输出层数"
                self.level_fusions = nn.ModuleList()
                self.level_downs = nn.ModuleList()
                hid_c = fusion.get('hidden_channels', 64)
                for idx, lid in enumerate(self.fusion_level_ids):
                    in_c = img_ch[lid] + pts_ch[lid]
                    n_gate = 2
                    if self.cascade_fusion and idx > 0:
                        # 第三路输入：上一层融合结果（stride-2 下采样 + 对齐通道）
                        in_c += img_ch[lid]
                        n_gate = 3
                        self.level_downs.append(
                            nn.Conv2d(
                                img_ch[self.fusion_level_ids[idx - 1]],
                                img_ch[lid],
                                kernel_size=3,
                                stride=2,
                                padding=1))
                    level_gate = nn.Sequential(
                        ConvModule(in_c, hid_c, 3,
                                   padding=1,
                                   norm_cfg=dict(type='BN'),
                                   act_cfg=dict(type='ReLU')),
                        nn.Conv2d(hid_c, n_gate, 1),
                        nn.Sigmoid(),
                    )
                    self._init_gate_bias(level_gate)
                    self.level_fusions.append(level_gate)

    def _init_gate_bias(self, gate_seq):
        """门控输出层 bias 初始化：第一路主导，其余路近似关闭。

        门控最后一层是 ``nn.Conv2d(hid, N, 1)``，bias 默认为全 0 →
        ``sigmoid(0) = 0.5``，训练第 0 步各个分支就各占一半权重，等于把随机
        初始化的新分支特征直接掺进已收敛的预训练 BEV 特征里，造成分布冲击
        （实测表现为检测 heatmap loss 在 warmup 阶段居高不下）。

        这里把第一路（图像/主分支）bias 设为 ``+gate_bias_first``、其余路设为
        ``-gate_bias_first``，使初始输出≈原基线行为，训练中再由网络自主学习
        抬高其他分支。``gate_bias_first=0`` 时不做任何处理（保持原行为）。

        Args:
            gate_seq (nn.Sequential): 门控网络，最后一个 nn.Conv2d 为输出层。
        """
        if not self.gate_bias_first:
            return
        conv = None
        for m in gate_seq:
            if isinstance(m, nn.Conv2d):
                conv = m
        if conv is None or conv.bias is None:
            return
        b = float(self.gate_bias_first)
        with torch.no_grad():
            conv.bias.fill_(-b)
            conv.bias[0] = b

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """对点云做体素化（与 CenterPoint.voxelize 一致）。

        Args:
            points (list[torch.Tensor]): 每帧点云，形状 (N_i, C)。

        Returns:
            voxels (torch.Tensor): (M, max_points, C)
            num_points (torch.Tensor): (M,) 每个体素内实际点数
            coors (torch.Tensor): (M, 4) [batch_idx, z, y, x]
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def extract_pts_feat(self, points, return_multi_level=False):
        """提取激光 BEV 特征。

        Args:
            points (list[torch.Tensor]): 每帧点云。
            return_multi_level (bool): 是否返回 backbone 多尺度特征列表。

        Returns:
            torch.Tensor | tuple:
                - return_multi_level=False: 最终 BEV 特征 [B, C, H, W]
                - return_multi_level=True: (multi_feats, final_feat)
                  multi_feats 是 backbone 各 stage 输出列表
        """
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        multi_feats = None
        if self.pts_bev_encoder_backbone is not None:
            x = self.pts_bev_encoder_backbone(x)
            if return_multi_level and isinstance(x, (list, tuple)):
                multi_feats = list(x)
        if self.pts_bev_encoder_neck is not None:
            x = self.pts_bev_encoder_neck(x)
            if isinstance(x, (list, tuple)):
                x = x[0]
        if return_multi_level:
            return multi_feats, x
        return x

    @staticmethod
    def _gate2d(net, inp, half_res):
        """方案B门控：在降采样分辨率算门控、再放大回原分辨率（算力优化）。

        门控是逐格软置信度先验：half_res=True 时先把输入 BEV 降到 100×100，
        nets（仍是 3×3）在低分辨率上算、再 bilinear 放大回 200×200。这样：
          - 算力约为满分辨率 3×3 的 1/4 + down/up 开销（occ/gate/rev 合计
            ~59 → ~15 GMAC）；
          - 保留 3×3 局部语义，同时上采样天然把门控图插值平滑，规避
            "满分辨率 1×1 逐格独立可能椒盐"的方案A风险。
        half_res=False（默认）时原样直通，整个模块行为不变。
        """
        B, C, H, W = inp.shape
        if not half_res:
            return net(inp)
        low = F.interpolate(
            inp, scale_factor=0.5, mode='bilinear', align_corners=False)
        g = net(low)
        if g.shape[-2:] != (H, W):
            g = F.interpolate(
                g, size=(H, W), mode='bilinear', align_corners=False)
        return g

    def fuse_bev_feats(self, img_bev, pts_bev):
        """BEV 特征融合。

        Args:
            img_bev (torch.Tensor): 图像 BEV 特征 [B, C_img, H, W]
            pts_bev (torch.Tensor): 激光 BEV 特征 [B, C_pts, H, W]

        Returns:
            torch.Tensor: 融合后 BEV 特征 [B, C_out, H, W]
        """
        if self.fusion_type == 'concat':
            x = torch.cat([img_bev, pts_bev], dim=1)
            x = self.fusion(x)
        elif self.fusion_type == 'add':
            x = img_bev + pts_bev
        elif self.fusion_type == 'gated':
            # 投影到相同通道数
            img_feat = self.img_proj(img_bev)
            pts_feat = self.pts_proj(pts_bev)
            # 预测门控权重 [B, 2, H, W]（可选方案B：half_res 时在 100×100 算+放大）
            gate = self._gate2d(
                self.gate_net,
                torch.cat([img_feat, pts_feat], dim=1),
                self.gate_half_res)
            img_gate = gate[:, 0:1, :, :]
            pts_gate = gate[:, 1:2, :, :]
            # 加权求和
            x = img_gate * img_feat + pts_gate * pts_feat
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")
        return x

    def rev_fuse_bev_feats(self, pts_bev, fused_bev):
        """反向融合：用融合特征（含图像语义）调制激光 BEV 特征。

        与 ``fuse_bev_feats`` 的门控结构对称，但输出送回激光分支，使激光的
        几何特征获得图像语义上下文（方案 A：双向融合）。未启用
        ``reverse_fusion`` 时原样返回。

        Args:
            pts_bev (torch.Tensor): 激光 BEV 特征 [B, C_pts, H, W]
            fused_bev (torch.Tensor): 融合 BEV 特征（含图像语义）
                [B, C_img, H, W]

        Returns:
            torch.Tensor: 调制后的激光 BEV 特征 [B, C_out, H, W]
        """
        if not self.rev_enabled:
            return pts_bev
        pts_feat = self.rev_pts_proj(pts_bev)
        img_feat = self.rev_img_proj(fused_bev)
        gate = self._gate2d(
            self.rev_gate_net,
            torch.cat([pts_feat, img_feat], dim=1),
            self.rev_half_res)
        pts_gate = gate[:, 0:1, :, :]
        img_gate = gate[:, 1:2, :, :]
        return pts_gate * pts_feat + img_gate * img_feat

    def _det_feats(self, img_feats):
        """检测分支特征：200x200@0.4m -> 100x100@0.8m（物理分辨率不变）。

        若 det_downsample.detach=True，在 det_downsample 之前对共享 BEV 特征做
        detach()，使检测 loss 的梯度不回流到 BEV 特征/视图变换网络——用于验证
        "检测任务抢占 BEV 特征导致 occ 变差"的假设（此时检测分支照常训练，
        仅共享特征不共享梯度）。

        注意：detach 必须在 det_downsample 输入处切断，而不是在输出处——否则
        det_downsample 自身参数收不到梯度，DDP(find_unused_parameters=False)
        会报 "Expected to have finished reduction" 崩溃。
        """
        if self.detach_det:
            img_feats = [f.detach() for f in img_feats]
        if self.det_downsample is not None:
            det_feats = [self.det_downsample(img_feats[0])]
        else:
            det_feats = img_feats
        return det_feats

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Returns:
            dict: Losses of detection branch, depth branch (if use_depth)
                and occupancy branch (if occ_head is not None).
        """
        if self.multi_level_fusion and self.with_lidar:
            # 多层级融合：提取图像+激光的多尺度特征，逐 stage 融合
            img_feats, _, depth, img_multi = self.extract_feat(
                points, img=img_inputs, img_metas=img_metas,
                return_multi_level=True, **kwargs)
            pts_multi, pts_bev_final = self.extract_pts_feat(
                points, return_multi_level=True)
            # 逐 stage 融合
            fused_multi = self.multi_level_fuse(img_multi, pts_multi)
            # 融合后的多尺度特征送入图像 neck
            fused_bev_neck = self.img_bev_encoder_neck(fused_multi)
            if isinstance(fused_bev_neck, (list, tuple)):
                fused_bev_neck = fused_bev_neck[0]
            # 最后再做一次全局门控融合（保证 fusion 模块参数被使用）
            fused_bev = self.fuse_bev_feats(fused_bev_neck, pts_bev_final)
            bev_feats = [fused_bev]
        else:
            img_feats, _, depth = self.extract_feat(
                points, img=img_inputs, img_metas=img_metas, **kwargs)
            # 多模态融合：图像 BEV + 激光 BEV（仅最后一层）
            if self.with_lidar:
                pts_bev = self.extract_pts_feat(points)
                fused_bev = self.fuse_bev_feats(img_feats[0], pts_bev)
                bev_feats = [fused_bev]
            else:
                bev_feats = img_feats

        if self.use_depth:
            gt_depth = kwargs['gt_depth']
            loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
            losses = dict(loss_depth=loss_depth)
        else:
            losses = dict()

        losses_pts = self.forward_pts_train(self._det_feats(bev_feats),
                                            gt_bboxes_3d, gt_labels_3d,
                                            img_metas, gt_bboxes_ignore)
        losses.update(losses_pts)
        if self.occ_head is not None:
            voxel_semantics = kwargs['voxel_semantics']
            mask_camera = kwargs['mask_camera']
            occ_pred = self.occ_head(bev_feats[0])

            # 激光专用 OCC head：与主 OCC 做可学习加权融合
            if self.pts_occ_head is not None and self.with_lidar:
                # 获取激光 BEV 特征（最终输出，用于 pts_occ_head）
                if self.multi_level_fusion:
                    pts_bev_for_occ = pts_bev_final
                else:
                    pts_bev_for_occ = pts_bev
                # 门控输入必须取 rev_fuse 之前的纯激光 BEV（multi/single 分支的
                # pts_bev_final / pts_bev 均未注入图像语义）。若放在下面 rev_fuse 之后，
                # pts_bev_for_occ 已被覆盖成含图像语义的融合特征，门控就测不出
                # "此处激光本身可不可信"，违背下面注释声明的设计。
                raw_pts_for_gate = pts_bev_for_occ
                # 方案 A 反向融合：用融合特征（含图像语义）调制激光分支
                pts_bev_for_occ = self.rev_fuse_bev_feats(
                    pts_bev_for_occ, bev_feats[0])
                pts_occ_pred = self.pts_occ_head(pts_bev_for_occ)
                # alpha = sigmoid(2D 空间门控 logit + 全局先验偏置 occ_fuse_logit)：
                #   - occ_alpha_head 用"纯相机 BEV || 纯激光 BEV"预测 [B,1,200,200] 门控 logit，
                #     sigmoid 后在 (Dz, C) 维广播 → 同一 BEV 格下 33 类统一加权，
                #     网络自动学到"近场信激光、远场信图像"的按位置配比；
                #   - occ_fuse_logit 保留为全局先验偏置（起点 alpha≈occ_fusion_weight），
                #     门控模块随机初始化≈0 → 起点行为≈原标量模型，训练中逐步空间化。
                #   主 OCC loss 直接作用在融合结果上，梯度回传给 occ_alpha_head 与
                #   occ_fuse_logit，自动学习最优空间配比。
                gate = self._gate2d(
                    self.occ_alpha_head,
                    torch.cat([img_feats[0], raw_pts_for_gate], dim=1),
                    self.occ_alpha_half_res)
                B, Dx, Dy, _, _ = occ_pred.shape
                alpha = torch.sigmoid(
                    gate.permute(0, 2, 3, 1).reshape(B, Dx, Dy, 1, 1)
                    + self.occ_fuse_logit)
                occ_pred = alpha * occ_pred + (1 - alpha) * pts_occ_pred
                # 诊断暂存：供 GateMonitorHook 读取
                self._occ_gate_stat = {
                    'alpha_mean': alpha.detach().mean().item(),
                    'alpha_std': alpha.detach().std().item(),
                    'spatial_gate_logit_mean': gate.detach().mean().item(),
                    'spatial_gate_logit_std': gate.detach().std().item(),
                    'occ_fuse_logit': self.occ_fuse_logit.detach().item(),
                }
                # 激光分支辅助监督（权重 0.5），保证 pts_occ_head 自身可训练
                pts_losses_occ = self.pts_occ_head.loss(
                    pts_occ_pred, voxel_semantics, mask_camera)
                for k, v in pts_losses_occ.items():
                    losses[f'pts_{k}'] = 0.5 * v
                # 把激光分支加权 loss 并入诊断 stat，供 GateMonitorHook 打印，作为
                # "激光分支在训练"的每 iter 直接证据（与日志中 pts_loss_occ 数值一致）。
                self._occ_gate_stat['pts_loss_occ'] = (
                    0.5 * pts_losses_occ['loss_occ']).detach().item()

            # 融合后的 OCC 预测参与主监督（CE + aux）
            losses_occ = self.occ_head.loss(occ_pred, voxel_semantics,
                                            mask_camera)
            losses.update(losses_occ)
        return losses

    @staticmethod
    def _has_points(points):
        """判断本帧是否有可用点云（用于激光缺失时的纯视觉推理兜底）。

        Args:
            points (list[torch.Tensor] | torch.Tensor | None): 每帧点云，
                单个张量形状为 (N_i, C)。None 表示激光缺失（驱动掉线、
                未发布点云），空张量 / numel==0 表示收到无效帧。

        Returns:
            bool: True → 走正常双模态融合前向；
                False → 走纯视觉兜底前向（跳过激光分支与 pts_occ_head）。

        Note:
            这里只做**廉价的形状判据**。NaN / 时间戳停滞 / 点数异常偏少等更
            细的有效性判定建议放在上层的健康监测模块处理：在 forward 里做
            ``torch.isnan(x).any()`` 会触发 GPU 同步，影响实时性。

            批量推理时是整批统一判定（只要有任意一帧有点云就走融合路径），
            因此建议部署时按 batch_size=1 使用，逐帧独立决策。
        """
        if points is None:
            return False
        if isinstance(points, (list, tuple)):
            if len(points) == 0:
                return False
            return any(p is not None and p.numel() > 0 for p in points)
        return points.numel() > 0

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentation.

        支持激光缺失时的**纯视觉兜底**：当 ``points`` 为 None 或空（激光缺失）
        而模型启用了融合分支时，跳过 ``extract_pts_feat`` 与 ``pts_occ_head``，
        直接用相机 BEV 特征喂检测头与 OCC 头，保证单模态仍可推理而不崩溃。

        Args:
            points (list[torch.Tensor] | torch.Tensor | None): 点云。None /
                空表示激光缺失，此时走纯视觉兜底前向。
            img_metas (list[dict]): 每帧元信息（推理时只需 'box_type_3d'）。
            img (tuple): 图像输入 7 元组。

        Returns:
            list[dict]: Each dict contains 'pts_bbox' (3D object detection
                results) and 'occ' (occupancy prediction, shape (Dx, Dy, Dz)).
        """
        # 激光可用性：None / 空点云时走纯视觉兜底（extract_pts_feat 不能接受 None）
        has_pts = self._has_points(points)
        if self.multi_level_fusion and self.with_lidar and has_pts:
            img_feats, _, _, img_multi = self.extract_feat(
                points, img=img, img_metas=img_metas,
                return_multi_level=True, **kwargs)
            pts_multi, pts_bev_final = self.extract_pts_feat(
                points, return_multi_level=True)
            fused_multi = self.multi_level_fuse(img_multi, pts_multi)
            fused_bev_neck = self.img_bev_encoder_neck(fused_multi)
            if isinstance(fused_bev_neck, (list, tuple)):
                fused_bev_neck = fused_bev_neck[0]
            fused_bev = self.fuse_bev_feats(fused_bev_neck, pts_bev_final)
            bev_feats = [fused_bev]
        else:
            img_feats, _, _ = self.extract_feat(
                points, img=img, img_metas=img_metas, **kwargs)
            if self.with_lidar and has_pts:
                pts_bev = self.extract_pts_feat(points)
                fused_bev = self.fuse_bev_feats(img_feats[0], pts_bev)
                bev_feats = [fused_bev]
            else:
                # 激光缺失（或未启用融合）：纯视觉前向，直接用相机 BEV。
                # 相机分支是完整自包含的，输出与融合 BEV 同形状（256@200x200），
                # 因此 head 可直接消费，无需构造零张量。
                bev_feats = img_feats

        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(self._det_feats(bev_feats),
                                        img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        if self.occ_head is not None:
            occ_pred = self.occ_head(bev_feats[0])
            # 激光 OCC head 融合。激光缺失时必须跳过：pts_occ_head 在全零
            # 特征上的输出是无意义的随机预测，叠加后会污染主 OCC；此时直接
            # 使用主 occ_head 在相机 BEV 上的预测即可。
            if self.pts_occ_head is not None and self.with_lidar and has_pts:
                if self.multi_level_fusion:
                    # 复用顶部多层级分支已算出的 pts_bev_final，避免重算整条激光流水线
                    # （voxelize + pillar + scatter + 3-stage backbone + neck）
                    pts_bev_for_occ = pts_bev_final
                else:
                    pts_bev_for_occ = pts_bev
                # 门控输入取 rev_fuse 之前的纯激光 BEV（与训练块一致）
                raw_pts_for_gate = pts_bev_for_occ
                # 方案 A 反向融合：用融合特征（含图像语义）调制激光分支
                pts_bev_for_occ = self.rev_fuse_bev_feats(
                    pts_bev_for_occ, bev_feats[0])
                pts_occ_pred = self.pts_occ_head(pts_bev_for_occ)
                # 与训练一致：2D 空间门控 + 全局先验偏置（布局 B,Dx,Dy,Dz,C）
                gate = self._gate2d(
                    self.occ_alpha_head,
                    torch.cat([img_feats[0], raw_pts_for_gate], dim=1),
                    self.occ_alpha_half_res)
                B, Dx, Dy, _, _ = occ_pred.shape
                alpha = torch.sigmoid(
                    gate.permute(0, 2, 3, 1).reshape(B, Dx, Dy, 1, 1)
                    + self.occ_fuse_logit)
                occ_pred = alpha * occ_pred + (1 - alpha) * pts_occ_pred
            occ_res = self.occ_head.get_occ(occ_pred)
            for result_dict, occ in zip(bbox_list, occ_res):
                result_dict['occ'] = occ
        return bbox_list

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # 与 simple_test 保持一致：激光缺失时跳过融合分支
        has_pts = self._has_points(points)
        if self.multi_level_fusion and self.with_lidar and has_pts:
            img_feats, _, _, img_multi = self.extract_feat(
                points, img=img_inputs, img_metas=img_metas,
                return_multi_level=True, **kwargs)
            pts_multi, pts_bev_final = self.extract_pts_feat(
                points, return_multi_level=True)
            fused_multi = self.multi_level_fuse(img_multi, pts_multi)
            fused_bev_neck = self.img_bev_encoder_neck(fused_multi)
            if isinstance(fused_bev_neck, (list, tuple)):
                fused_bev_neck = fused_bev_neck[0]
            fused_bev = self.fuse_bev_feats(fused_bev_neck, pts_bev_final)
            bev_feats = [fused_bev]
        else:
            img_feats, _, _ = self.extract_feat(
                points, img=img_inputs, img_metas=img_metas, **kwargs)
            if self.with_lidar and has_pts:
                pts_bev = self.extract_pts_feat(points)
                fused_bev = self.fuse_bev_feats(img_feats[0], pts_bev)
                bev_feats = [fused_bev]
            else:
                # 激光缺失：纯视觉前向
                bev_feats = img_feats
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(self._det_feats(bev_feats))
        if self.occ_head is not None:
            occ_pred = self.occ_head(bev_feats[0])
            return [outs, occ_pred]
        return outs

