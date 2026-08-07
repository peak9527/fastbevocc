# FastBEV++
[FastBEV++: Fast by Algorithm, Deployable by Design](https://arxiv.org/abs/2512.08237v1)

![](./resources/fastbev_vs_fastvec++_v2.0.png)
## Get Started

Please prepare the training environment according to [BEVDet](https://github.com/HuangJunJie2017/BEVDet?tab=readme-ov-file#get-started).  

install env:

```shell
conda env create -f requirements/full_env.yaml 
```

training script:


```shell
./tools/dist_train.sh configs/fastbev/paper/fastbev-r50-cbgs.py 8
```

test script:

```shell
./tools/dist_test.sh configs/fastbev/paper/fastbev-r50-cbgs.py work_dirs/fastbev-r50-cbgs/epoch_20_ema.pth 8 --eval mAP 2>&1 | tee work_dirs/fastbev-r50-cbgs/epoch_20_ema.pth.log
```


visualization results:

```shell
# 输出 JSON 格式结果
./tools/dist_test.sh configs/fastbev/paper/fastbev-r50-cbgs.py \
  work_dirs/fastbev-r50-cbgs/epoch_20_ema.pth 2 \
  --format-only \
  --eval-options 'jsonfile_prefix=work_dirs/fastbev-r50-cbgs/results'
  
  
  #如果用最新的自己训练的模型，可以执行命令
  
 ./tools/dist_test.sh configs/fastbev/paper/fastbev-r50-cbgs.py \
  work_dirs/fastbev-r50-cbgs/latest.pth 2 \      
  --format-only \                             
  --eval-options 'jsonfile_prefix=work_dirs/fastbev-r50-cbgs/results'

# 可视化
python tools/analysis_tools/vis.py work_dirs/fastbev-r50-cbgs/results/pts_bbox/results_nusc.json \
  --root_path data/nuscenes \
  --save_path work_dirs/fastbev-r50-cbgs/vis_bev \
  --show-range 50 --vis-thred 0.3 --draw-gt --format video --fps 10
```
## 使用自定义数据进行推理和可视化
1、自定义数据生成pkl文件:

```shell
python tools/create_custom_pkl.py \
  --data-dir data/custom_data/yk_c1/data \
  --lidar-calib data/custom_data/yk_c1/calib/lidar/helios32_top_vehicle_extrinsics.yaml \
  --calib-dir data/custom_data/yk_c1/calib/camera \
  --output yk_c1_infos_val.pkl
```
2、训练自定义数据:

```shell
./tools/dist_train.sh configs/fastbev/paper/fastbev-r50-cbgs-custom.py 2
```

3、训练自定义数据:

```shell
./tools/dist_test.sh configs/fastbev/paper/fastbev-r50-cbgs-custom.py \
  work_dirs/fastbev-r50-cbgs-custom/latest.pth 2 \
  --format-only \
  --eval-options 'jsonfile_prefix=work_dirs/fastbev-r50-cbgs-custom/results_custom'
```

4、可视化推理结果:

```shell
 python tools/vis_custom_results.py \
  work_dirs/fastbev-r50-cbgs-custom/results_custom/pts_bbox/results_nusc.json \
  --pkl custom_infos_train.pkl \
  --save-path work_dirs/fastbev-r50-cbgs-custom/vis_custom_v1 \
  --show-range 50 --vis-thred 0.3 --draw-gt \
  --format video --fps 2
```
可视化结果如下：
![alt text](image.png)
## Main Results

![](./resources/fastbev++_exps.jpg)
![](./resources/fastbev++_result.png)

## Model Zoo
- [fastbev-r50-cbgs](https://drive.google.com/drive/folders/1AYtoX8XaNg8ZckFBgbo2TVmyXb26Dw2V?usp=sharing)
- [fastbev-r50-cbgs-4d](https://drive.google.com/drive/folders/1XBcftoEn_2TkpG-qQAo66Oa8_hEnfoqi?usp=sharing)
- [fastbev-r101-cbgs-4d-longterm](https://drive.google.com/drive/folders/1JLcU96Oimk7wSZLi7-FeImUGD_1fXWG1)
- [fastbev-r101-cbgs-4d-longterm-depth](https://drive.google.com/drive/folders/1RG2hFFu0germP-8sIvMYpv-K-eVER6yB)

## Inference Latency

Exported onnx: [Google Drive](https://drive.google.com/file/d/1qefFlah6PkKPtz0Zgh9aqn0rqITjU2C8/view?usp=sharing). Trt Latency based on onnx: [Google Drive](https://drive.google.com/file/d/1NJPv2_HDk6B01dZ0g0tOXIxc5zFzz4Wd/view?usp=sharing).

**FastBEV++ ONNX Architecture**
![](./resources/fastbev++_onnx_arch.jpg)

**FastBEV++ Latency**
![](./resources/Fastbev++R50_Latency.png)


## Acknowledgement

This project is not possible without multiple great open-sourced code bases. We list some notable examples below.

- [open-mmlab](https://github.com/open-mmlab)
- [CenterPoint](https://github.com/tianweiy/CenterPoint)
- [Lift-Splat-Shoot](https://github.com/nv-tlabs/lift-splat-shoot)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [BEVFusion](https://github.com/mit-han-lab/bevfusion)
- [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)
- [BEVerse](https://github.com/zhangyp15/BEVerse)  for multi-task learning.
- [BEVStereo](https://github.com/Megvii-BaseDetection/BEVStereo)  for stero depth estimation.
- [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
- [Fast-BEV](https://github.com/Sense-GVT/Fast-BEV)
