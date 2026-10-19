"""从测试输出的 results_occ.pkl + yk_c1 val GT 重算 occ 语义 mIoU。
用法: <python> calc_occ_metric.py <results_occ.pkl> <val_infos.pkl>
Metric 逻辑与 mmdet3d/datasets/occ_metrics.py 的 Metric_mIoU 一致（use_image_mask=True）。
"""
import numpy as np
import os, sys, pickle

num_classes = 33
class_names = [
    'ignore', 'truck', 'tram', 'car', 'special_vehicle', 'cyclist',
    'bicycle', 'tricycle', 'other_non_motor_vehicle', 'pedestrian',
    'cone', 'other_object', 'barrier', 'road', 'other_ground', 'curb',
    'vegetation', 'pole', 'trunk', 'manmade', 'fence', 'lifting_lever',
    'noise_dust', 'noise_underground', 'noise_other', 'shadow', 'misc',
    'background', 'helmet', 'carton', 'bottle', 'speed_bump', 'free',
]

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def hist_info(n_cl, pred, gt):
    assert pred.shape == gt.shape
    k = (gt >= 0) & (gt < n_cl)
    labeled = np.sum(k)
    return np.bincount(
        n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
    ).reshape(n_cl, n_cl), labeled

res_path, infos_path = sys.argv[1], sys.argv[2]
results = pickle.load(open(res_path, 'rb'))
d = pickle.load(open(infos_path, 'rb'))
infos = d['infos'] if isinstance(d, dict) and 'infos' in d else d
assert len(results) == len(infos), f"results {len(results)} != infos {len(infos)}"

hist = np.zeros((num_classes, num_classes))
for i, (r, info) in enumerate(zip(results, infos)):
    occ_pred = r['occ'] if isinstance(r, dict) else r
    if i == 0:
        print('pred shape/dtype:', occ_pred.shape, occ_pred.dtype,
              'unique:', np.unique(occ_pred)[:10])
    gt = np.load(os.path.join(info['occ_path'], 'labels.npz'))
    gt_semantics = gt['semantics']
    mask_camera = gt['mask_camera'].astype(bool)
    occ_pred = np.asarray(occ_pred)
    if occ_pred.shape != gt_semantics.shape:
        # pred 可能是 4D logits [X,Y,Z,C]，取 argmax
        occ_pred = occ_pred.argmax(axis=-1)
    h, _ = hist_info(num_classes, occ_pred[mask_camera], gt_semantics[mask_camera])
    hist += h
    if (i + 1) % 1000 == 0:
        print(f'  processed {i+1}/{len(infos)}')

miou = per_class_iu(hist)
print(f'===> per class IoU of {len(infos)} samples:')
for ind_class in range(num_classes - 1):
    print(f'===> {class_names[ind_class]} - IoU = ' + str(round(miou[ind_class] * 100, 2)))
print(f'===> mIoU (前 {num_classes-1} 类, 不计 free) = ' + str(round(np.nanmean(miou[:num_classes-1]) * 100, 2)))