import os
import json
import argparse
import numpy as np
from mmdet3d.apis import init_model, inference_detector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True, help='点云根目录，存放.bin/.pcd点云')
    parser.add_argument('--ckpt', required=True, help='pointpillars权重路径')
    parser.add_argument('--config', default='./configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_nus-3d.py')
    parser.add_argument('--out_label_dir', default='./auto_labels', help='自动生成3D标注json输出文件夹')
    args = parser.parse_args()

    # 初始化PointPillars模型
    model = init_model(args.config, args.ckpt, device='cuda:0')
    os.makedirs(args.out_label_dir, exist_ok=True)

    # 遍历所有点云文件
    point_cloud_files = sorted([f for f in os.listdir(args.data_root) if f.endswith('.bin') or f.endswith('.pcd')])
    for pc_name in point_cloud_files:
        pc_path = os.path.join(args.data_root, pc_name)
        # 推理单帧点云
        result, _ = inference_detector(model, pc_path)
        pred = result.pred_instances_3d

        # 提取3D框信息 x,y,z 中心 / w,l,h 尺寸 / yaw航向 / 类别 / 置信度
        labels = []
        # 一次性将所有 tensor 转到 CPU，避免循环内重复 .cpu() 调用
        boxes = pred.bboxes_3d.tensor.cpu().numpy()   # shape: (N, 7) 或 (N, 9)
        labels_3d = pred.labels_3d.cpu().numpy()
        scores_3d = pred.scores_3d.cpu().numpy()
        classes = model.dataset_meta['classes']
        num_obj = boxes.shape[0]
        for i in range(num_obj):
            score = float(scores_3d[i])
            if score < 0.3:  # 过滤低置信误检
                continue
            box = boxes[i]
            cls_id = int(labels_3d[i])
            labels.append({
                "category": classes[cls_id],
                "xyz": [float(box[0]), float(box[1]), float(box[2])],
                "wlh": [float(box[3]), float(box[4]), float(box[5])],
                "yaw": float(box[6]),
                "score": score
            })
        # 保存同文件名json标注
        out_json_path = os.path.join(args.out_label_dir, pc_name.replace('.bin', '.json').replace('.pcd', '.json'))
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(labels, f, indent=2)
    print(f"批量推理完成，标注输出至：{args.out_label_dir}")

if __name__ == "__main__":
    main()