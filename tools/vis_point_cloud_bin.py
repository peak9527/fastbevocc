#!/usr/bin/env python3
"""
激光雷达点云 .bin 文件可视化脚本
支持多格式点云数据：
  - 3D (x, y, z)
  - 4D KITTI 格式 (x, y, z, intensity)
  - 5D 带 ring (x, y, z, intensity, ring)
  - 6D 带 ring+timestamp (x, y, z, intensity, ring, timestamp)
  - 7D 带所有字段
支持交互可视化及无头(headless)服务器环境截图/导出
"""

import argparse
import numpy as np
import os
import sys

# Open3D 为可选依赖，无 GUI 环境下导入可能仍会崩溃，推迟到使用时再加载
_O3D_AVAILABLE = None


def _check_o3d():
    """懒加载检查 Open3D 是否可用（避免无 GUI 环境导入时崩溃）"""
    global _O3D_AVAILABLE
    if _O3D_AVAILABLE is not None:
        return _O3D_AVAILABLE
    try:
        import open3d as o3d
        # 尝试做一个轻量操作验证渲染管线正常
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3), dtype=np.float32))
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=1, height=1, visible=False)
            vis.destroy_window()
        except Exception:
            _O3D_AVAILABLE = False
            return False
        _O3D_AVAILABLE = True
        return True
    except ImportError:
        _O3D_AVAILABLE = False
        return False


def load_bin_file(file_path: str, dim: int = None) -> np.ndarray:
    """
    加载 .bin 点云文件，支持自动检测或手动指定维度

    Args:
        file_path: bin文件路径
        dim: 每点维度（3/4/5/6/7）。None 时自动检测（优先 6D > 4D）

    Returns:
        点云数据数组 (N, dim) 或 None
    """
    try:
        raw = np.fromfile(file_path, dtype=np.float32)
    except Exception as e:
        print(f"加载文件失败: {e}")
        return None

    total = len(raw)

    # 手动指定维度
    if dim is not None:
        if total % dim != 0:
            print(f"警告：数据长度 {total} 不能被 dim={dim} 整除，将截断")
            total = (total // dim) * dim
            raw = raw[:total]
        return raw.reshape(-1, dim)

    # 自动检测：优先尝试 6D（该工程常见格式），再尝试 4D
    candidate_dims = [d for d in [6, 4, 5, 7, 3] if total % d == 0]
    if not candidate_dims:
        print(f"警告：数据长度 {total} 不能被 3/4/5/6/7 整除，默认按 4D 截断处理")
        total = (total // 4) * 4
        raw = raw[:total]
        return raw.reshape(-1, 4)

    # 启发式检测
    for d in candidate_dims:
        data = raw.reshape(-1, d)
        if d == 6:
            z_range = data[:, 2].max() - data[:, 2].min()
            ring_vals = data[:, 4]
            ring_unique = np.unique(ring_vals)
            is_ring_like = (1 <= len(ring_unique) <= 128
                            and ring_vals.min() >= 0
                            and ring_vals.max() <= 128)
            if z_range < 50 and is_ring_like:
                print(f"自动检测为 {d}D 点云（z范围={z_range:.1f}m, ring={len(ring_unique)}条线束）")
                return data
        if d == 4:
            z_range = data[:, 2].max() - data[:, 2].min()
            if z_range < 50:
                print(f"自动检测为 {d}D 点云（z范围={z_range:.1f}m）")
                return data

    # 兜底：按第一个候选维度解析
    d = candidate_dims[0]
    print(f"未能自动判定维度，按 {d}D 解析")
    return raw.reshape(-1, d)


def export_point_cloud(point_cloud: np.ndarray, output_path: str):
    """
    导出点云为 PCD/PLY 文件（纯 numpy 实现，不依赖 Open3D）

    Args:
        point_cloud: (N, dim) 数组
        output_path: .pcd 或 .ply 输出路径
    """
    if point_cloud is None or len(point_cloud) == 0:
        print("错误：点云数据为空，无法导出")
        return

    ext = os.path.splitext(output_path)[1].lower()
    n = len(point_cloud)

    if ext == '.ply':
        _export_ply(point_cloud, output_path, n)
    elif ext == '.pcd':
        _export_pcd(point_cloud, output_path, n)
    else:
        # 默认 pcd
        if not output_path.endswith('.pcd'):
            output_path += '.pcd'
        _export_pcd(point_cloud, output_path, n)

    print(f"点云已导出到: {output_path} ({n} 个点，{point_cloud.shape[1]} 维)")


def _export_ply(point_cloud, output_path, n):
    """纯 numpy 写 PLY 文件"""
    dim = point_cloud.shape[1]
    fields = ['x', 'y', 'z', 'intensity', 'ring', 'timestamp'][:dim]
    with open(output_path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        for field in fields:
            f.write(f'element vertex {n}\n')
            break
        f.write(f'element vertex {n}\n')
        for field in fields:
            f.write(f'property float {field}\n')
        f.write('end_header\n')
        for i in range(n):
            f.write(' '.join(f'{point_cloud[i, j]:.6f}' for j in range(dim)) + '\n')


def _export_pcd(point_cloud, output_path, n):
    """纯 numpy 写 ASCII PCD 文件"""
    dim = point_cloud.shape[1]
    fields = ['x', 'y', 'z', 'intensity', 'ring', 'timestamp'][:dim]
    field_str = ' '.join(fields)
    with open(output_path, 'w') as f:
        f.write('# .PCD v0.7 - Point Cloud Data file format\n')
        f.write('VERSION 0.7\n')
        f.write(f'FIELDS {field_str}\n')
        f.write(f'SIZE {" ".join(["4"] * dim)}\n')
        f.write(f'TYPE {" ".join(["F"] * dim)}\n')
        f.write(f'COUNT {" ".join(["1"] * dim)}\n')
        f.write(f'WIDTH {n}\nHEIGHT 1\n')
        f.write('VIEWPOINT 0 0 0 1 0 0 0\n')
        f.write(f'POINTS {n}\n')
        f.write('DATA ascii\n')
        for i in range(n):
            f.write(' '.join(f'{point_cloud[i, j]:.6f}' for j in range(dim)) + '\n')


def _render_headless_o3d(point_cloud: np.ndarray, output_path: str,
                         window_name: str = "Point Cloud"):
    """使用 Open3D 离屏渲染并保存截图（需要 O3D 可用）"""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3].astype(np.float32))

    # 强度着色
    if point_cloud.shape[1] >= 4:
        intensity = point_cloud[:, 3].astype(np.float32)
        denom = intensity.max() - intensity.min()
        if denom > 0:
            intensity = (intensity - intensity.min()) / denom
        else:
            intensity[:] = 0.5
        colors = np.stack([intensity, intensity * 0.6, 1.0 - intensity], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720, visible=False)
    vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    if ctr is not None:
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        ctr.set_front([-1, -1, -1])

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_path)
    vis.destroy_window()
    print(f"截图已保存到: {output_path}")


def visualize_point_cloud(point_cloud: np.ndarray,
                          window_name: str = "Point Cloud Viewer",
                          headless: bool = False,
                          output_img: str = None):
    """
    可视化点云：交互模式（需要 GUI）、离屏截图或纯文件导出

    Args:
        point_cloud: (N, dim) 数组
        window_name: 窗口名称
        headless:    True = 无头模式（导出文件，不弹窗）
        output_img:  输出文件路径
    """
    if point_cloud is None or len(point_cloud) == 0:
        print("错误：点云数据为空")
        return

    # 统计信息
    print(f"点云统计: "
          f"x=[{point_cloud[:, 0].min():.1f}, {point_cloud[:, 0].max():.1f}], "
          f"y=[{point_cloud[:, 1].min():.1f}, {point_cloud[:, 1].max():.1f}], "
          f"z=[{point_cloud[:, 2].min():.1f}, {point_cloud[:, 2].max():.1f}]")
    if point_cloud.shape[1] >= 4:
        print(f"        intensity=[{point_cloud[:, 3].min():.0f}, {point_cloud[:, 3].max():.0f}]")

    # 无头模式：直接导出文件
    if headless:
        out = output_img or "point_cloud.pcd"
        export_point_cloud(point_cloud, out)
        print("提示：可使用 CloudCompare / Open3D-GUI / pcl_viewer 等工具查看导出文件")
        return

    # 交互模式：尝试使用 Open3D
    if not _check_o3d():
        print("Open3D 不可用（无 GUI 环境），自动导出点云文件...")
        out = output_img or "point_cloud.pcd"
        export_point_cloud(point_cloud, out)
        print("提示：可将此文件拷贝到有 GUI 的机器上用 Open3D/CloudCompare 查看")
        return

    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3].astype(np.float32))
    if point_cloud.shape[1] >= 4:
        intensity = point_cloud[:, 3].astype(np.float32)
        denom = intensity.max() - intensity.min()
        if denom > 0:
            intensity = (intensity - intensity.min()) / denom
        else:
            intensity[:] = 0.5
        colors = np.stack([intensity, intensity * 0.6, 1.0 - intensity], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    try:
        vis.create_window(window_name=window_name, width=1280, height=720)
    except Exception:
        print("无法创建交互窗口（无 GUI 环境），导出点云文件...")
        out = output_img or "point_cloud.pcd"
        export_point_cloud(point_cloud, out)
        return

    vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    if ctr is not None:
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        ctr.set_front([-1, -1, -1])
    else:
        print("警告：无法获取视图控制器，使用默认视角")

    print("点云可视化窗口已打开")
    print("操作提示：")
    print("  - 鼠标左键 + 拖拽：旋转视角")
    print("  - 鼠标右键 + 拖拽：平移视角")
    print("  - 鼠标滚轮：缩放")
    print("  - 按 'Q' 键退出")

    vis.run()
    vis.destroy_window()


def save_to_pcd(point_cloud: np.ndarray, output_path: str):
    """
    将点云保存为 PCD 格式文件

    Args:
        point_cloud: 点云数据数组
        output_path: 输出文件路径
    """
    if point_cloud is None:
        print("错误：点云数据为空")
        return

    pcd = _create_o3d_pointcloud(point_cloud)
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"点云已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="激光雷达点云 .bin 文件可视化工具")
    parser.add_argument("input_file", help="输入的 .bin 点云文件路径")
    parser.add_argument("--output", "-o", help="输出文件路径（PCD 或截图）")
    parser.add_argument("--dim", "-d", type=int, default=None,
                        help="点云维度（3/4/5/6/7）。不指定则自动检测")
    parser.add_argument("--headless", action="store_true",
                        help="无头模式：直接保存截图，不弹出交互窗口")
    args = parser.parse_args()

    print(f"正在加载点云文件: {args.input_file}")

    # 加载点云数据
    point_cloud = load_bin_file(args.input_file, dim=args.dim)

    if point_cloud is None:
        return

    print(f"点云加载成功，共 {len(point_cloud)} 个点，{point_cloud.shape[1]} 维")

    # 保存为 PCD（如果指定了输出路径且不是 headless 截图模式）
    if args.output and not args.headless:
        save_to_pcd(point_cloud, args.output)

    # 可视化点云
    visualize_point_cloud(point_cloud, headless=args.headless, output_img=args.output)


if __name__ == "__main__":
    main()