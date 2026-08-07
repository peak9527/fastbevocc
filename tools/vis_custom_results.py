#!/usr/bin/env python3
"""
可视化自定义数据集的推理结果 (results_nusc.json)。

基于 tools/analysis_tools/vis.py 改造, 支持:
  - 任意数量相机 (自动检测)
  - 自定义 pkl 路径
  - 灵活布局 (1~6 相机自适应)

用法:
    python tools/vis_custom_results.py \
        work_dirs/fastbev-r50-cbgs/results_custom/pts_bbox/results_nusc.json \
        --pkl custom_infos_val.pkl \
        --save-path work_dirs/fastbev-r50-cbgs/vis_custom \
        --show-range 50 --vis-thred 0.3 --draw-gt \
        --format video --fps 10
"""

import argparse
import json
import os
import pickle

import cv2
import numpy as np
from pyquaternion.quaternion import Quaternion

from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes as LB


def check_point_in_img(points, height, width):
    valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
    valid = np.logical_and(
        valid, np.logical_and(points[:, 0] < width, points[:, 1] < height))
    return valid


def depth2color(depth):
    gray = max(0, min((depth + 2.5) / 3.0, 1.0))
    max_lumi = 200
    colors = np.array(
        [[max_lumi, 0, max_lumi], [max_lumi, 0, 0], [max_lumi, max_lumi, 0],
         [0, max_lumi, 0], [0, max_lumi, max_lumi], [0, 0, max_lumi]],
        dtype=np.float32)
    if gray == 1:
        return tuple(colors[-1].tolist())
    num_rank = len(colors) - 1
    rank = np.floor(gray * num_rank).astype(np.int)
    diff = (gray - rank / num_rank) * num_rank
    return tuple(
        (colors[rank] + (colors[rank + 1] - colors[rank]) * diff).tolist())


def lidar2img(points_lidar, camrera_info):
    points_lidar_homogeneous = \
        np.concatenate([points_lidar,
                        np.ones((points_lidar.shape[0], 1),
                                dtype=points_lidar.dtype)], axis=1)
    camera2lidar = np.eye(4, dtype=np.float32)
    camera2lidar[:3, :3] = camrera_info['sensor2lidar_rotation']
    camera2lidar[:3, 3] = camrera_info['sensor2lidar_translation']
    lidar2camera = np.linalg.inv(camera2lidar)
    points_camera_homogeneous = points_lidar_homogeneous @ lidar2camera.T
    points_camera = points_camera_homogeneous[:, :3]
    valid = np.ones((points_camera.shape[0]), dtype=bool)
    valid = np.logical_and(points_camera[:, -1] > 0.5, valid)
    points_camera = points_camera / points_camera[:, 2:3]
    camera2img = camrera_info['cam_intrinsic']
    points_img = points_camera @ camera2img.T
    points_img = points_img[:, :2]
    return points_img, valid


def get_lidar2global(infos):
    lidar2ego = np.eye(4, dtype=np.float32)
    lidar2ego[:3, :3] = Quaternion(infos['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = infos['lidar2ego_translation']
    ego2global = np.eye(4, dtype=np.float32)
    ego2global[:3, :3] = Quaternion(
        infos['ego2global_rotation']).rotation_matrix
    ego2global[:3, 3] = infos['ego2global_translation']
    return ego2global @ lidar2ego


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize inference result '
                                     'of custom dataset')
    parser.add_argument(
        'res', help='Path to the predicted result in json format')
    parser.add_argument(
        '--pkl',
        type=str,
        required=True,
        help='Path to the custom pkl file (e.g., custom_infos_val.pkl)')
    parser.add_argument(
        '--show-range',
        type=int,
        default=50,
        help='Range of visualization in BEV')
    parser.add_argument(
        '--canva-size', type=int, default=1000, help='Size of canva in pixel')
    parser.add_argument(
        '--vis-frames',
        type=int,
        default=500,
        help='Number of frames for visualization')
    parser.add_argument(
        '--scale-factor',
        type=int,
        default=4,
        help='Trade-off between image-view and bev in size of '
        'the visualized canvas')
    parser.add_argument(
        '--vis-thred',
        type=float,
        default=0.3,
        help='Threshold the predicted results')
    parser.add_argument(
        '--yaw-offset',
        type=float,
        default=np.pi / 2,
        help='Yaw offset added to predictions when converting quaternion to yaw')
    parser.add_argument('--draw-gt', action='store_true')
    parser.add_argument(
        '--save-path',
        type=str,
        default='./vis_custom_output',
        help='Path to save visualization results')
    parser.add_argument(
        '--format',
        type=str,
        default='video',
        choices=['video', 'image'],
        help='The desired format of the visualization result')
    parser.add_argument(
        '--fps', type=int, default=20, help='Frame rate of video')
    parser.add_argument(
        '--video-prefix', type=str, default='vis', help='Name of video')
    args = parser.parse_args()
    return args


color_map = {0: (255, 255, 0), 1: (0, 255, 255)}


def _build_layout(num_views, scale_factor, canva_size):
    """根据相机数量自适应构建布局参数。

    Returns:
        dict with:
          - grid_cols, grid_rows: 图像格子的列数和行数
          - img_disp_w, img_disp_h: 单幅图像显示尺寸
          - canvas_w, canvas_h: 总画布尺寸
          - views_top, views_bottom: 上下排的 view 索引 (或空)
    """
    # 图像显示区域
    if num_views <= 2:
        grid_cols = num_views
        grid_rows = 1
    elif num_views <= 4:
        grid_cols = 2
        grid_rows = (num_views + 1) // 2
    else:
        grid_cols = 3
        grid_rows = 2

    img_disp_w = int(1600 / scale_factor)
    img_disp_h = int(900 / scale_factor)
    img_panel_w = img_disp_w * grid_cols
    img_panel_h = img_disp_h * grid_rows

    # 6 相机特殊处理 (原始 nuScenes 布局: 3上+3下翻转)
    if num_views == 6:
        views_top = [0, 1, 2]
        views_bottom = [3, 4, 5]
        canvas_w = img_disp_w * 3
        canvas_h = img_disp_h * 2 + canva_size
    else:
        views_top = list(range(num_views))
        views_bottom = []
        canvas_w = max(img_panel_w, canva_size)
        canvas_h = img_panel_h + canva_size

    return {
        'grid_cols': grid_cols,
        'grid_rows': grid_rows,
        'img_disp_w': img_disp_w,
        'img_disp_h': img_disp_h,
        'canvas_w': canvas_w,
        'canvas_h': canvas_h,
        'views_top': views_top,
        'views_bottom': views_bottom,
    }


def _is_ego2global_identity(infos_list):
    """检查所有帧的 ego2global 是否都是恒等变换 (自定义数据集标志)"""
    for info in infos_list[:10]:  # 采样前 10 帧
        rot = Quaternion(info.get('ego2global_rotation', [1, 0, 0, 0]))
        trans = info.get('ego2global_translation', [0, 0, 0])
        if abs(rot.w - 1.0) > 1e-4 or abs(rot.x) > 1e-4 \
                or abs(rot.y) > 1e-4 or abs(rot.z) > 1e-4:
            return False
        if abs(trans[0]) > 1e-4 or abs(trans[1]) > 1e-4 or abs(trans[2]) > 1e-4:
            return False
    return True


def main():
    args = parse_args()
    # load predicted results
    res = json.load(open(args.res, 'r'))
    # load custom dataset info
    dataset = pickle.load(open(args.pkl, 'rb'))
    infos_list = dataset['infos']

    # auto-detect camera names from first frame
    cam_names = sorted(infos_list[0]['cams'].keys())
    num_cams = len(cam_names)
    print(f'Detected {num_cams} camera(s): {cam_names}')

    # --- 确定 yaw 偏移 ---
    # 自定义数据: ego2global=I, 预测和 GT 均在 ego/LiDAR frame
    #   gt_boxes 按 nuScenes 约定存储为 [x,y,z,l,w,h,yaw],
    #   LiDARInstance3DBoxes 中 yaw=0 对应 x 正方向(前),
    #   因此自定义数据不需要任何 yaw 偏移
    # nuScenes 数据: 预测在 global frame, 需要 π/2 偏移
    is_custom = _is_ego2global_identity(infos_list)
    if is_custom:
        print('Dataset: custom (ego2global = identity, preds/GT both in LiDAR frame)')
        yaw_offset = 0.0
        use_lidar_frame = True  # pred boxes are already in LiDAR frame, skip l2g transform
        print(f'Using yaw_offset = {yaw_offset:.4f} (FLU convention: 0 = forward)')
    else:
        print('Dataset: nuScenes (ego2global varies, predictions in global frame)')
        yaw_offset = args.yaw_offset
        use_lidar_frame = False
        print(f'Using --yaw-offset: {yaw_offset:.4f}')
    # -----------------------------------------

    # build layout
    layout = _build_layout(num_cams, args.scale_factor, args.canva_size)
    print(f'Canvas size: {layout["canvas_w"]}x{layout["canvas_h"]}')

    # prepare save path
    vis_dir = args.save_path
    os.makedirs(vis_dir, exist_ok=True)
    print('Saving visualized result to %s' % vis_dir)

    show_range = args.show_range
    canva_size = args.canva_size
    scale_factor = args.scale_factor

    if args.format == 'video':
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        vout = cv2.VideoWriter(
            os.path.join(vis_dir, '%s.mp4' % args.video_prefix), fourcc,
            args.fps, (layout['canvas_w'], layout['canvas_h']))

    draw_boxes_indexes_bev = [(0, 1), (1, 2), (2, 3), (3, 0)]
    draw_boxes_indexes_img_view = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5),
                                   (5, 6), (6, 7), (7, 4), (0, 4), (1, 5),
                                   (2, 6), (3, 7)]

    n_frames = min(args.vis_frames, len(infos_list))
    print('Start visualizing %d frames' % n_frames)

    for cnt, infos in enumerate(infos_list[:n_frames]):
        if cnt % 10 == 0:
            print('%d/%d' % (cnt, n_frames))

        token = infos['token']

        # --- collect predicted instances ---
        # results_nusc.json format (nuScenes): [x,y,z], size=[w,l,h], rotation=quat
        # mmdet3d LiDARInstance3DBoxes expects [x,y,z, x_size, y_size, z_size, yaw]
        # where x_size corresponds to nuScenes length (l), y_size to width (w).
        #
        # 注意: CenterPoint/FastBEV 在生成 results_nusc.json 时，
        # 模型预测的是 gravity-center z，但 get_bboxes() 里会错误地
        # 把 z 减去 h/2 再写入 translation[2]。
        # 因此这里的 pred z 是 gravity_center - h/2，比真实 center 偏低 h/2。
        # 需要在下方加回 h/2 恢复 gravity_center，再用 origin=(0.5,0.5,0.5) 解析。
        # GT (SUSTechPOINTS psr.position) 是 gravity-center，同样用 0.5。
        pred_res = res['results'].get(token, [])
        pred_boxes = []
        for rid in range(len(pred_res)):
            trans = pred_res[rid]['translation']
            size = pred_res[rid]['size']
            yaw = Quaternion(pred_res[rid]['rotation']).yaw_pitch_roll[0]
            pred_boxes.append([
                trans[0], trans[1], trans[2],
                size[1], size[0], size[2],  # x_size=l, y_size=w, z_size=h
                yaw + yaw_offset
            ])

        if len(pred_boxes) == 0:
            corners_lidar = np.zeros((0, 3), dtype=np.float32)
        else:
            pred_boxes = np.array(pred_boxes, dtype=np.float32)
            # 补偿 get_bboxes() 中多余的 z - h/2 操作:
            # get_bboxes() (centerpoint_head.py line 751) 把 z 减去 h/2，
            # 将模型预测的 gravity_center 转成 bottom_center 写入 results_nusc.json。
            # 这里加回 h/2 恢复 gravity_center，再用 origin=0.5 正确解析。
            pred_boxes[:, 2] = pred_boxes[:, 2] + pred_boxes[:, 5] * 0.5
            boxes = LB(pred_boxes, origin=(0.5, 0.5, 0.5))

            # DEBUG: 打印首位预测框的 z 诊断信息
            # if len(pred_boxes) > 0:
            #     pb = pred_boxes[0]
            #     gt_b = infos.get('gt_boxes')
            #     print(f'\n[DEBUG frame={cnt}] Pred box0: z_bottom={pb[2]:.3f},'
            #           f' h={pb[5]:.3f}, z_top={pb[2]+pb[5]:.3f},'
            #           f' l={pb[3]:.3f}, w={pb[4]:.3f}')
            #     if gt_b is not None and len(gt_b) > 0:
            #         gb = gt_b[0]
            #         print(f'[DEBUG frame={cnt}] GT   box0: z_gravity={gb[2]:.3f},'
            #               f' h={gb[5]:.3f}, z_bottom_gt={gb[2]-gb[5]*0.5:.3f},'
            #               f' z_top_gt={gb[2]+gb[5]*0.5:.3f},'
            #               f' l={gb[3]:.3f}, w={gb[4]:.3f}')
            #         print(f'[DEBUG frame={cnt}] z_bottom_diff: {pb[2] - (gb[2]-gb[5]*0.5):.4f}m,'
            #               f' z_top_diff: {(pb[2]+pb[5]) - (gb[2]+gb[5]*0.5):.4f}m')
            corners_tmp = boxes.corners.numpy().reshape(-1, 3)
            if not use_lidar_frame:
                # nuScenes: pred boxes are in global frame, convert back to LiDAR
                corners_tmp = np.concatenate(
                    [corners_tmp,
                     np.ones([corners_tmp.shape[0], 1])],
                    axis=1)
                l2g = get_lidar2global(infos)
                corners_lidar = corners_tmp @ np.linalg.inv(l2g).T
                corners_lidar = corners_lidar[:, :3]
            else:
                # custom: pred boxes (and GT) are already in LiDAR frame
                corners_lidar = corners_tmp

        pred_flag = np.ones((corners_lidar.shape[0] // 8, ), dtype=bool)
        scores = [
            pred_res[rid]['detection_score'] for rid in range(len(pred_res))
        ]

        # --- draw GT if requested ---
        # custom_infos_train.pkl stores gt_boxes as [x,y,z,l,w,h,yaw]
        # in FLU convention (yaw=0 means length (l) is along x/front).
        # LiDARInstance3DBoxes expects [x,y,z,x_size,y_size,z_size,yaw],
        # i.e. x_size=l, y_size=w — exactly what the pkl provides, no swap needed.
        if args.draw_gt and infos.get('gt_boxes') is not None \
                and len(infos['gt_boxes']) > 0:
            gt_boxes = infos['gt_boxes'].copy()
            corners_lidar_gt = \
                LB(gt_boxes,
                   origin=(0.5, 0.5, 0.5)).corners.numpy().reshape(-1, 3)
            corners_lidar = np.concatenate([corners_lidar, corners_lidar_gt],
                                           axis=0)
            gt_flag = np.ones((corners_lidar_gt.shape[0] // 8), dtype=bool)
            pred_flag = np.concatenate(
                [pred_flag, np.logical_not(gt_flag)], axis=0)
            scores = scores + [0 for _ in range(gt_boxes.shape[0])]
        scores = np.array(scores, dtype=np.float32)
        sort_ids = np.argsort(scores)

        # --- draw camera images with projected boxes ---
        cam_imgs = []
        for cam_name in cam_names:
            cam_info = infos['cams'][cam_name]
            try:
                img = cv2.imread(cam_info['data_path'])
                if img is None:
                    # placeholder
                    img = np.zeros((900, 1600, 3), dtype=np.uint8)
                    cv2.putText(img, f'{cam_name} (not found)', (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            except (KeyError, FileNotFoundError):
                img = np.zeros((900, 1600, 3), dtype=np.uint8)
                cv2.putText(img, f'{cam_name} (error)', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # draw 3D box projections
            if corners_lidar.shape[0] > 0:
                corners_img, valid = lidar2img(corners_lidar, cam_info)
                valid = np.logical_and(
                    valid,
                    check_point_in_img(corners_img, img.shape[0],
                                       img.shape[1]))
                valid = valid.reshape(-1, 8)
                corners_img = corners_img.reshape(-1, 8, 2).astype(np.int)
                for aid in range(valid.shape[0]):
                    for index in draw_boxes_indexes_img_view:
                        if valid[aid, index[0]] and valid[aid, index[1]]:
                            cv2.line(
                                img,
                                corners_img[aid, index[0]],
                                corners_img[aid, index[1]],
                                color=color_map[int(pred_flag[aid])],
                                thickness=2)

            # resize for display
            img_disp = cv2.resize(img, (layout['img_disp_w'],
                                        layout['img_disp_h']))
            cv2.putText(img_disp, cam_name, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cam_imgs.append(img_disp)

        # --- BEV canvas ---
        canvas = np.zeros((canva_size, canva_size, 3), dtype=np.uint8)
        # draw lidar points in BEV: forward (x) -> up, left (y) -> left
        # 注意: 图像坐标 y 向下为正, x 向右为正
        # canvas px = -y * scale + center
        # canvas py = -x * scale + center
        lidar_path = infos.get('lidar_path', '')
        if lidar_path and os.path.exists(lidar_path):
            lidar_points = np.fromfile(lidar_path, dtype=np.float32)
            try:
                for dim in [6, 4, 5, 3]:
                    if lidar_points.size % dim == 0:
                        lidar_points = lidar_points.reshape(-1, dim)
                        break
            except ValueError:
                pass
            lidar_points = lidar_points[:, :3]
            xs = lidar_points[:, 0]
            ys = lidar_points[:, 1]
            mask = (np.abs(xs) < show_range) & (np.abs(ys) < show_range)
            xs, ys = xs[mask], ys[mask]
            zs = lidar_points[mask, 2]
            scale = canva_size / (2 * show_range)
            px = ((-ys + show_range) * scale).astype(np.int32)
            py = ((-xs + show_range) * scale).astype(np.int32)
            valid = (px >= 0) & (px < canva_size) & (py >= 0) & (py < canva_size)
            px, py, zs = px[valid], py[valid], zs[valid]
            for i in range(px.shape[0]):
                color = depth2color(zs[i])
                cv2.circle(canvas, (int(px[i]), int(py[i])),
                           radius=0, color=color, thickness=1)

        # BEV 方向指示: 黄点为 ego, 黄色箭头指向前方 (画布上方)
        center = canva_size // 2
        cv2.circle(canvas, (center, center), 5, (0, 255, 255), -1)
        cv2.arrowedLine(canvas, (center, center), (center, center - 30),
                        (0, 255, 255), 2, tipLength=0.3)
        cv2.putText(canvas, 'FRONT', (center - 30, center - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # draw 3D boxes in BEV (same orientation as LiDAR points)
        # canvas px = -y, py = -x, so we rotate corners similarly
        if corners_lidar.shape[0] > 0:
            corners_lidar_bev = corners_lidar.reshape(-1, 8, 3)
            cx = corners_lidar_bev[:, :, 0]  # forward
            cy = corners_lidar_bev[:, :, 1]  # left
            rot_x = -cy
            rot_y = -cx
            bottom_idx = [0, 3, 7, 4]
            bottom_rot_x = rot_x[:, bottom_idx]
            bottom_rot_y = rot_y[:, bottom_idx]
            bottom_corners_bev = np.stack([bottom_rot_x, bottom_rot_y], axis=-1)
            bottom_corners_bev = \
                (bottom_corners_bev + show_range) / show_range / 2.0 * canva_size
            bottom_corners_bev = np.round(bottom_corners_bev).astype(np.int32)
            center_bev = np.stack([bottom_rot_x, bottom_rot_y], axis=-1).mean(axis=1)
            head_bev = np.stack([rot_x[:, [4, 7]], rot_y[:, [4, 7]]], axis=-1).mean(axis=1)
            center_canvas = \
                (center_bev + show_range) / show_range / 2.0 * canva_size
            center_canvas = center_canvas.astype(np.int32)
            head_canvas = \
                (head_bev + show_range) / show_range / 2.0 * canva_size
            head_canvas = head_canvas.astype(np.int32)

            for rid in sort_ids:
                score = scores[rid]
                if score < args.vis_thred and pred_flag[rid]:
                    continue
                score = max(min(score * 2.0, 1.0), 0.35) if pred_flag[rid] else 1.0
                color = color_map[int(pred_flag[rid])]
                for index in draw_boxes_indexes_bev:
                    cv2.line(
                        canvas,
                        bottom_corners_bev[rid, index[0]],
                        bottom_corners_bev[rid, index[1]],
                        [color[0] * score, color[1] * score, color[2] * score],
                        thickness=1)
                cv2.line(
                    canvas,
                    center_canvas[rid],
                    head_canvas[rid],
                    [color[0] * score, color[1] * score, color[2] * score],
                    1,
                    lineType=8)

        # --- fuse image-view and BEV ---
        canvas_h, canvas_w = layout['canvas_h'], layout['canvas_w']
        full_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # place camera images
        for i, img_disp in enumerate(cam_imgs):
            col = i % layout['grid_cols']
            row = i // layout['grid_cols']
            y0 = row * layout['img_disp_h']
            x0 = col * layout['img_disp_w']
            y1 = y0 + layout['img_disp_h']
            x1 = x0 + layout['img_disp_w']
            full_img[y0:y1, x0:x1, :] = img_disp

        # place BEV below (or centered if wider)
        bev_y0 = layout['img_disp_h'] * layout['grid_rows']
        bev_x0 = max(0, (canvas_w - canva_size) // 2)
        full_img[bev_y0:bev_y0 + canva_size,
                 bev_x0:bev_x0 + canva_size, :] = canvas

        if args.format == 'image':
            cv2.imwrite(os.path.join(vis_dir, '%s.jpg' % token), full_img)
        elif args.format == 'video':
            vout.write(full_img)

    if args.format == 'video':
        vout.release()
    print('Done.')


if __name__ == '__main__':
    main()
