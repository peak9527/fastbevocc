import numpy as np
import cv2
import os
import re
import bisect
import yaml

# ===================== 全局配置 =====================
CAM_CALIB_PATH = "./data/calibration/camera/frontmain.yaml"
LIDAR_CALIB_PATH = "./data/calibration/lidar/helios32_top_vehicle_extrinsics.yaml"
DATA_ROOT = "data/custom_data/sequence2"
TARGET_IMG_PATH = r"data/custom_data/sequence2/CAM_FRONT/1781488486436.jpg"
SAVE_IMG = "./align_validate_RFU_only.jpg"
SCALE_TRANS = 1.0
MAX_TIME_DIFF_MS = 100
# ====================================================

def quaternion_to_matrix(qx, qy, qz, qw):
    """四元数转旋转矩阵 (scalar-last: x,y,z,w), 纯 numpy 实现"""
    R = np.zeros((3, 3))
    R[0, 0] = 1 - 2 * (qy**2 + qz**2)
    R[0, 1] = 2 * (qx * qy - qz * qw)
    R[0, 2] = 2 * (qx * qz + qy * qw)
    R[1, 0] = 2 * (qx * qy + qz * qw)
    R[1, 1] = 1 - 2 * (qx**2 + qz**2)
    R[1, 2] = 2 * (qy * qz - qx * qw)
    R[2, 0] = 2 * (qx * qz - qy * qw)
    R[2, 1] = 2 * (qy * qz + qx * qw)
    R[2, 2] = 1 - 2 * (qx**2 + qy**2)
    return R


def read_opencv_yaml(file_path):
    """读取 OpenCV 格式标定 yaml (含 !!opencv-matrix), 用于相机标定文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        raise FileNotFoundError(f"标定文件不存在 {file_path}: {str(e)}")

    # 去除 %YAML:1.0 头部
    if content.startswith('%YAML'):
        idx = content.find('---')
        if idx != -1:
            content = content[idx+3:]
        else:
            lines = content.split('\n')[1:]
            content = '\n'.join(lines)

    mat_pattern = r'!!opencv-matrix\s*\n\s*rows:\s*(\d+)\s*\n\s*cols:\s*(\d+)\s*\n\s*dt:\s*(\w+)\s*\n\s*data:\s*\[(.*?)\]'
    mat_dict = {}
    matches = list(re.finditer(mat_pattern, content, re.DOTALL))

    for idx, match in enumerate(matches):
        rows = int(match.group(1))
        cols = int(match.group(2))
        data_str = match.group(4)
        clean_str = data_str.replace("\n", "").replace(" ", "")
        str_list = clean_str.split(",")
        data = []
        for s in str_list:
            s_strip = s.strip()
            if s_strip == "" or s_strip == ".":
                continue
            data.append(float(s_strip))
        mat = np.array(data, dtype=np.float32).reshape(rows, cols)
        mat_dict[idx] = mat
        content = content.replace(match.group(0), f'__MAT_PLACE_{idx}__', 1)

    raw = yaml.safe_load(content)

    # 修复：执行递归填充，返回解析完的字典，不是返回函数
    def fill_mat(obj):
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                new_dict[k] = fill_mat(v)
            return new_dict
        elif isinstance(obj, list):
            new_list = []
            for item in obj:
                new_list.append(fill_mat(item))
            return new_list
        elif isinstance(obj, str) and '__MAT_PLACE_' in obj:
            m_idx = int(re.search(r'__MAT_PLACE_(\d+)__', obj).group(1))
            return mat_dict.get(m_idx, obj)
        return obj

    parsed_data = fill_mat(raw)
    return parsed_data

def invert_extrinsic(R, t):
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv

def load_lidar_bin(path):
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]

def get_ts_from_filename(file_path):
    """从文件名提取时间戳, 返回微秒级整数.

    文件名格式: n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151061512404.jpg
    第一组 "008" 是场景号, 最后一组 "1533151061512404" 才是微秒时间戳。
    """
    basename = os.path.basename(file_path)
    matches = re.findall(r"\d+", basename)
    if not matches:
        raise ValueError(f"无法从文件名提取时间戳: {basename}")
    return int(matches[-1])

def match_lidar_by_approx_ts(data_root, target_img_ts, max_diff_ms):
    lidar_dir = os.path.join(data_root, "lidar")
    if not os.path.exists(lidar_dir):
        lidar_dir = os.path.join(data_root, "LIDAR_TOP")
        if not os.path.exists(lidar_dir):
            raise FileNotFoundError(f"未找到lidar文件夹: {data_root}")
    lidar_ts_list = []
    lidar_path_map = {}
    for fname in os.listdir(lidar_dir):
        if fname.endswith(".bin"):
            ts = get_ts_from_filename(fname)
            lidar_ts_list.append(ts)
            lidar_path_map[ts] = os.path.join(lidar_dir, fname)
    if not lidar_ts_list:
        return None
    lidar_ts_sorted = sorted(lidar_ts_list)
    pos = bisect.bisect_left(lidar_ts_sorted, target_img_ts)
    candidate_ts = []
    if pos < len(lidar_ts_sorted):
        candidate_ts.append(lidar_ts_sorted[pos])
    if pos > 0:
        candidate_ts.append(lidar_ts_sorted[pos-1])
    best_ts = min(candidate_ts, key=lambda x: abs(x - target_img_ts))
    # 文件名中时间戳是微秒级, 除以 1000 转换为毫秒后与阈值比较
    diff_ms = abs(best_ts - target_img_ts) / 1000.0
    if diff_ms <= max_diff_ms:
        print(f"匹配时差: {diff_ms:.1f} ms (允许最大{max_diff_ms}ms)")
        return lidar_path_map[best_ts]
    else:
        print(f"最近时差{diff_ms:.1f}ms，超过阈值{max_diff_ms}ms，无匹配点云")
        return None

def load_camera_extrinsic_RFU():
    calib = read_opencv_yaml(CAM_CALIB_PATH)
    K = calib["M"].astype(np.float32)
    D = calib["D"].astype(np.float32)
    Tr = calib["Tr_cam_to_vehicle"].astype(np.float32)
    R_cam2ego_rfu = Tr[:3, :3]
    t_cam2ego_rfu = Tr[:3, 3] * SCALE_TRANS
    I_check = R_cam2ego_rfu @ R_cam2ego_rfu.T
    if not np.allclose(I_check, np.eye(3), atol=1e-3):
        raise RuntimeError("相机外参旋转矩阵非正交，标定异常")
    return K, D, R_cam2ego_rfu, t_cam2ego_rfu

def load_lidar_extrinsic_RFU():
    """直接 yaml.safe_load 读取激光雷达外参 (同 vis_custom_pkl.py 方式)"""
    with open(LIDAR_CALIB_PATH, 'r') as f:
        calib = yaml.safe_load(f)

    trans = calib["transform"]["translation"]
    rot = calib["transform"]["rotation"]
    coord = calib.get("coordination", "unknown")
    frame_id = calib["header"]["frame_id"]
    child_frame_id = calib["child_frame_id"]

    t = np.array([trans["x"], trans["y"], trans["z"]]) * SCALE_TRANS
    R = quaternion_to_matrix(rot["x"], rot["y"], rot["z"], rot["w"])

    # 构建 T_vehicle_lidar (parent=vehicle -> child=lidar), 同 vis_custom_pkl.py
    T_vehicle_lidar = np.eye(4)
    T_vehicle_lidar[:3, :3] = R
    T_vehicle_lidar[:3, 3] = t

    print(f"[Lidar Calib] {LIDAR_CALIB_PATH}")
    print(f"  parent frame:  {frame_id}")
    print(f"  child frame:   {child_frame_id}")
    print(f"  coordination:  {coord}")
    print(f"  translation:   [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
    print(f"  rotation quat: [{rot['x']:.6f}, {rot['y']:.6f}, {rot['z']:.6f}, {rot['w']:.6f}]")
    print(f"  T_vehicle_lidar (4x4):\n{R}\n{t}")

    # 从 T_vehicle_lidar 求逆得到 T_lidar_vehicle, 同 vis_custom_pkl.py 的 transform_points 语义
    T_lidar_vehicle = np.linalg.inv(T_vehicle_lidar)
    lidar2ego_rfu = T_lidar_vehicle[:3, :3]
    lidar2ego_t_rfu = T_lidar_vehicle[:3, 3]
    return lidar2ego_rfu, lidar2ego_t_rfu

def main():
    # 全程原生RFU，不做RFU→FLU转换
    K, D, cam2ego_RFU, cam2ego_tRFU = load_camera_extrinsic_RFU()
    lidar2ego_RFU, lidar2ego_tRFU = load_lidar_extrinsic_RFU()

    # RFU下 cam -> lidar
    sensor2lidar_RFU = lidar2ego_RFU @ cam2ego_RFU
    sensor2lidar_tRFU = lidar2ego_RFU @ cam2ego_tRFU + lidar2ego_tRFU

    # 匹配图片与点云
    img_ts = get_ts_from_filename(TARGET_IMG_PATH)
    print(f"图片时间戳: {img_ts}")
    lidar_path = match_lidar_by_approx_ts(DATA_ROOT, img_ts, MAX_TIME_DIFF_MS)
    if lidar_path is None:
        print("ERROR: 未找到匹配lidar bin")
        return
    img = cv2.imread(TARGET_IMG_PATH)
    if img is None:
        print(f"ERROR: 图片读取失败 {TARGET_IMG_PATH}")
        return
    print(f"输入图片: {TARGET_IMG_PATH}")
    print(f"匹配点云: {lidar_path}")

    # 图像去畸变
    h, w = img.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=1)
    img_undist = cv2.undistort(img, K, D, None, new_K)

    # lidar -> cam 逆变换用于投影
    l2s_R, l2s_t = invert_extrinsic(sensor2lidar_RFU, sensor2lidar_tRFU)
    pts_lidar = load_lidar_bin(lidar_path)
    pts_cam = (l2s_R @ pts_lidar.T).T + l2s_t
    mask_front = pts_cam[:, 2] > 0.2
    pts_valid = pts_cam[mask_front]

    # 透视投影绘制
    pts_homo = pts_valid.T
    pixels = (new_K @ pts_homo).T
    u = pixels[:, 0] / pixels[:, 2]
    v = pixels[:, 1] / pixels[:, 2]
    for ui, vi in zip(u, v):
        x = int(round(ui))
        y = int(round(vi))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img_undist, (x, y), radius=1, color=(0, 255, 0), thickness=-1)

    cv2.imwrite(SAVE_IMG, img_undist)
    print(f"\n✅ 原生RFU坐标系对齐图已保存: {SAVE_IMG}")
    print("判断：点正常横向铺开=之前是RFU2FLU矩阵旋转导致竖线；依旧竖线=雷达tf/原始标定问题")

if __name__ == "__main__":
    main()