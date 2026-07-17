import os
import json
import open3d as o3d
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless 兼容, 无需 GUI
import matplotlib.pyplot as plt

# ===================== 配置路径（改成你自己的路径）=====================
PC_DIR = "data/custom_data/sequence2/LIDAR_TOP"        # 点云bin文件夹
JSON_RAW_DIR = "data/custom_data/sequence2/auto_labels"  # 原始自动标签
JSON_SAVE_DIR = "data/custom_data/sequence2/checked_labels"  # 修正后标签保存目录
SNAPSHOT_DIR = "data/custom_data/sequence2/label_snapshots"  # headless 快照保存目录
CONF_THRESH = 0.4  # 低置信阈值
SNAPSHOT_RANGE = 80  # BEV 快照范围 (米)
# ======================================================================

os.makedirs(JSON_SAVE_DIR, exist_ok=True)

def load_bin_point_cloud(bin_path):
    """加载雷达bin点云 N*4 (x,y,z,intensity)"""
    pc_data = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_data[:, :3])
    return pcd

def create_bbox_from_obj(obj):
    """从json物体构造open3d包围盒"""
    x, y, z = obj["xyz"]
    w, l, h = obj["wlh"]
    ry = obj["yaw"]
    conf = obj.get("score", 1.0)

    box = o3d.geometry.OrientedBoundingBox()
    box.center = np.array([x, y, z])
    box.extent = np.array([w, l, h])
    # Z轴旋转
    R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, ry])
    box.R = R
    # 置信度区分颜色
    if conf < CONF_THRESH:
        box.color = [1, 0, 0]  # 红：低置信可疑框
    else:
        box.color = [0, 1, 0]  # 绿：正常框
    return box, obj

def process_single_frame(timestamp_str):
    bin_file = os.path.join(PC_DIR, f"{timestamp_str}.bin")
    json_file = os.path.join(JSON_RAW_DIR, f"{timestamp_str}.json")
    save_json_path = os.path.join(JSON_SAVE_DIR, f"{timestamp_str}.json")

    if not os.path.exists(bin_file) or not os.path.exists(json_file):
        print(f"缺失文件: {timestamp_str}")
        return

    # 读取点云
    pcd = load_bin_point_cloud(bin_file)
    # 读取标签 (兼容两种 JSON 格式: 顶层 array 或 dict with "objects" key)
    with open(json_file, "r", encoding="utf-8") as f:
        label_data = json.load(f)
    if isinstance(label_data, list):
        obj_list = label_data
        label_data = {"objects": label_data}  # 包装为 dict, 保存时统一格式
    elif isinstance(label_data, dict):
        obj_list = label_data.get("objects", [])
    else:
        obj_list = []

    # 生成所有包围盒
    bboxes = []
    obj_mapping = []
    for obj in obj_list:
        box, obj = create_bbox_from_obj(obj)
        bboxes.append(box)
        obj_mapping.append(obj)

    # 可视化窗口
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"Label Editor | TS:{timestamp_str}")
    vis.add_geometry(pcd)
    for box in bboxes:
        vis.add_geometry(box)

    # 全局缓存当前框与物体
    global current_bboxes, current_objs, label_global
    current_bboxes = bboxes
    current_objs = obj_mapping
    label_global = label_data

    # 快捷键-D 删除选中框
    def delete_selected(vis):
        picked = vis.get_picked_geometry_index()
        if 0 <= picked < len(current_bboxes):
            vis.remove_geometry(current_bboxes[picked])
            del current_bboxes[picked]
            del current_objs[picked]
        return False

    # 快捷键-S 保存修正后的json
    def save_labels(vis):
        label_global["objects"] = current_objs
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(label_global, f, indent=2)
        print(f"已保存修正标签: {save_json_path}")
        return False

    # 绑定按键
    vis.register_key_callback(ord("D"), delete_selected)
    vis.register_key_callback(ord("S"), save_labels)

    print("====操作说明====")
    print("鼠标左键：旋转视角 | 滚轮：缩放 | 右键：平移")
    print("左键点击包围盒选中 | D 删除选中框 | S 保存修改 | Q 关闭窗口切换下一帧")
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # 读取所有json时间戳
    all_ts = [f[:-5] for f in os.listdir(JSON_RAW_DIR) if f.endswith(".json")]
    all_ts.sort()
    print(f"共找到 {len(all_ts)} 帧标注")
    for ts in all_ts:
        process_single_frame(ts)