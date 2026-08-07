#!/usr/bin/env python3
"""
ONNX → TensorRT Engine 构建工具 (使用 trtexec CLI)

用法:
  # 基础构建 (FP16)
  python build_trt_engine.py --onnx models/fastbev_end2end.onnx \
                             --engine models/fastbev_end2end_fp16.engine \
                             --fp16

  # INT8 量化 (需要校准数据)
  python build_trt_engine.py --onnx models/fastbev_end2end.onnx \
                             --engine models/fastbev_end2end_int8.engine \
                             --int8 --calib-dir data/calibration/

  # 指定动态 shape profile (trtexec 格式: name:min_shape,opt_shape,max_shape)
  python build_trt_engine.py --onnx models/fastbev_backbone.onnx \
                             --engine models/backbone_fp16.engine \
                             --fp16 \
                             --min-shape img:1x3x256x704 \
                             --opt-shape img:6x3x256x704 \
                             --max-shape img:6x3x256x704
"""

import argparse
import os
import re
import subprocess
import sys


def _find_trtexec():
    """查找 trtexec 二进制路径：先检查便宜的来源，最后才做昂贵的递归搜索."""
    # 1. 环境变量 TRTEXEC_BIN
    custom = os.environ.get("TRTEXEC_BIN")
    if custom and os.path.isfile(custom) and os.access(custom, os.X_OK):
        return custom

    # 2. PATH 中的 trtexec (shutil.which 几乎瞬间返回)
    import shutil
    found = shutil.which("trtexec")
    if found:
        return found

    # 3. 常见固定安装路径
    for p in ["/usr/bin/trtexec", "/usr/local/bin/trtexec",
              "/opt/tensorrt/bin/trtexec"]:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p

    # 4. 最后一招：递归搜索常见安装目录（昂贵操作，仅在前面都失败时执行）
    for base in ["/home", "/opt", "/usr/local"]:
        import glob
        for p in glob.glob(f"{base}/**/TensorRT*/bin/trtexec", recursive=True):
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p

    return None


def _resolve_trt_lib_dir(trtexec_path):
    """根据 trtexec 路径推断 TensorRT lib 目录（用于 LD_LIBRARY_PATH）."""
    # 常见布局: .../TensorRT-X.Y.Z/bin/trtexec → .../TensorRT-X.Y.Z/lib/
    bin_dir = os.path.dirname(os.path.realpath(trtexec_path))
    parent = os.path.dirname(bin_dir)

    candidates = [
        os.path.join(parent, "lib"),
        os.path.join(parent, "targets", "x86_64-linux-gnu", "lib"),
        os.path.join(os.path.dirname(parent), "lib"),
    ]
    for d in candidates:
        if os.path.isdir(d):
            # 确认目录里有 TRT 的 .so
            if any(f.endswith(".so") for f in os.listdir(d)):
                return d
    return None


def _get_trt_major_version(trtexec_path, env):
    """获取 trtexec 的 TensorRT 主版本号 (8 或 10)."""
    try:
        result = subprocess.run(
            [trtexec_path, "--help"],
            capture_output=True, text=True, timeout=30, env=env
        )
        # 匹配 "TensorRT v100800" 或 "TensorRT v8601" 格式
        m = re.search(r'TensorRT\s+v(\d+)', result.stderr + result.stdout)
        if m:
            version = int(m.group(1))  # e.g. 100800 or 8601
            return version // 10000    # e.g. 10 or 8
    except Exception:
        pass
    return 8  # 默认按 8.x 行为


def _get_input_names(trtexec_path, onnx_path, env):
    """通过 trtexec --onnx=<path> --verbose 的 dry-run 获取输入名称."""
    try:
        result = subprocess.run(
            [trtexec_path, f"--onnx={onnx_path}", "--verbose"],
            capture_output=True, text=True, timeout=180, env=env
        )
        names = []
        for line in (result.stdout + result.stderr).splitlines():
            m = re.search(r"Input\s+binding\s+for\s+(\S+)\s+with\s+dimensions", line)
            if m:
                names.append(m.group(1))
        return names
    except Exception as e:
        print(f"Warning: Failed to probe input names: {e}")
        return []


def build_engine(onnx_path, engine_path, fp16=False, int8=False,
                 min_shape=None, opt_shape=None, max_shape=None,
                 max_workspace=1024, trtexec_path=None):
    """
    通过 trtexec CLI 从 ONNX 构建 TensorRT engine.

    Args:
        onnx_path:      ONNX 模型路径
        engine_path:    输出 engine 路径
        fp16:           启用 FP16
        int8:           启用 INT8
        min_shape:      动态 shape 最小值 (trtexec 格式: name:dim1xdim2...)
        opt_shape:      动态 shape 最优值
        max_shape:      动态 shape 最大值
        max_workspace:  GPU workspace 大小 (MiB)
        trtexec_path:   自定义 trtexec 路径 (None=自动检测)
    """
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file not found: {onnx_path}")
        return False

    # 定位 trtexec
    if trtexec_path is None:
        trtexec_path = _find_trtexec()
    if trtexec_path is None or not os.path.isfile(trtexec_path):
        print("Error: trtexec not found. "
              "Install TensorRT or set TRTEXEC_BIN/--trtexec.")
        return False

    print(f"[trtexec path] {trtexec_path}")

    # 设置 LD_LIBRARY_PATH，包含 TRT 的 lib 目录
    env = os.environ.copy()
    trt_lib = _resolve_trt_lib_dir(trtexec_path)
    if trt_lib:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{trt_lib}:{existing}" if existing else trt_lib
        print(f"[trtexec lib]  {trt_lib}")

    # 创建输出目录
    out_dir = os.path.dirname(engine_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cmd = [trtexec_path, f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]

    # 精度
    if fp16:
        cmd.append("--fp16")
        print("FP16 mode enabled")
    if int8:
        cmd.append("--int8")
        print("INT8 mode enabled")

    # 动态 shape
    shape_args = _build_shape_args(trtexec_path, onnx_path, min_shape,
                                   opt_shape, max_shape, env)
    cmd.extend(shape_args)

    # builder 最高优化级别（build 时间略长，推理更快）
    trt_major = _get_trt_major_version(trtexec_path, env)
    if trt_major >= 10:
        cmd.append("--builderOptimizationLevel=5")
        print("Builder optimization level: 5 (maximum)")
    # workspace (TRT 10.x 使用 --memPoolSize, TRT 8.x 使用 --workspace)
    if trt_major >= 10:
        cmd.append(f"--memPoolSize=workspace:{max_workspace}")
    else:
        cmd.append(f"--workspace={max_workspace}")

    # 打印命令
    print(f"\n[trtexec] {' '.join(cmd)}")
    print(f"Building engine from {onnx_path}...")

    try:
        result = subprocess.run(cmd, check=False, capture_output=False, env=env)
    except FileNotFoundError:
        print(f"Error: '{trtexec_path}' not found.")
        return False
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return False

    if result.returncode != 0:
        print(f"\nError: trtexec failed with exit code {result.returncode}")
        print("Common fixes:")
        print("  1. Ensure cuDNN is installed: sudo apt install libcudnn8")
        print("  2. Set LD_LIBRARY_PATH to include TensorRT lib directory")
        print("  3. Use --trtexec to specify the correct trtexec binary")
        return False

    # 检查输出
    if not os.path.exists(engine_path):
        print(f"Error: Engine file not created: {engine_path}")
        return False

    size_mb = os.path.getsize(engine_path) / 1024 / 1024
    print(f"Engine written to: {engine_path} ({size_mb:.1f} MB)")

    # 打印 engine I/O 信息 (非关键，失败不影响)
    _print_engine_io(trtexec_path, engine_path, env)

    return True


def _build_shape_args(trtexec_path, onnx_path, min_shape, opt_shape,
                      max_shape, env):
    """构建 trtexec 动态 shape 参数."""
    args = []
    has_shapes = min_shape or opt_shape or max_shape

    if not has_shapes:
        return args

    # 探测输入名
    input_names = _get_input_names(trtexec_path, onnx_path, env)

    def _to_trtexec_fmt(raw, prefix, names):
        """将用户输入转换为 trtexec 格式: --minShapes=name:dim1xdim2...,name:dim1xdim2..."""
        if raw is None:
            return None
        # 如果已经包含 ':' (如 img:1x3x256x704)，直接用作 per-input 格式
        if ":" in raw:
            return ",".join(s.strip() for s in raw.split(","))
        # 否则用探测到的名字拼接
        shape_val = raw.replace(",", "x").strip()
        if names:
            return ",".join(f"{n}:{shape_val}" for n in names)

        print(f"Warning: Cannot resolve input names for {prefix}; "
              f"trying raw format. Consider using 'name:dim1xdim2x...' format.")
        return shape_val

    min_str = _to_trtexec_fmt(min_shape, "min_shape", input_names)
    opt_str = _to_trtexec_fmt(opt_shape, "opt_shape", input_names)
    max_str = _to_trtexec_fmt(max_shape, "max_shape", input_names)

    if min_str:
        args.append(f"--minShapes={min_str}")
    if opt_str:
        args.append(f"--optShapes={opt_str}")
    if max_str:
        args.append(f"--maxShapes={max_str}")

    if min_str or opt_str or max_str:
        print(f"Optimization profile: {min_str or 'auto'} → "
              f"{opt_str or 'auto'} → {max_str or 'auto'}")

    return args


def _print_engine_io(trtexec_path, engine_path, env):
    """通过 trtexec 加载 engine 并打印 I/O 信息."""
    try:
        result = subprocess.run(
            [trtexec_path, f"--loadEngine={engine_path}", "--verbose"],
            capture_output=True, text=True, timeout=60, env=env
        )
        lines = (result.stdout + result.stderr).splitlines()
        io_lines = [l for l in lines if re.search(
            r"(Input|Output)\s+binding", l, re.IGNORECASE)]
        if io_lines:
            print("\nEngine I/O:")
            for line in io_lines:
                print(f"  {line.strip()}")
    except Exception as e:
        print(f"Warning: Failed to inspect engine I/O: {e}")


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engine via trtexec")
    parser.add_argument("--onnx", required=True, help="ONNX model path")
    parser.add_argument("--engine", required=True, help="Output engine path")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 quantization")
    parser.add_argument("--min-shape", type=str, default=None,
                        help="Min shape (trtexec format: name:dim1xdim2x...,... or dim1,dim2,...)")
    parser.add_argument("--opt-shape", type=str, default=None,
                        help="Opt shape (trtexec format)")
    parser.add_argument("--max-shape", type=str, default=None,
                        help="Max shape (trtexec format)")
    parser.add_argument("--workspace", type=int, default=1024,
                        help="Max workspace in MB (default: 1024)")
    parser.add_argument("--trtexec", type=str, default=None,
                        help="Path to trtexec binary (auto-detect by default)")
    args = parser.parse_args()

    success = build_engine(
        args.onnx, args.engine,
        fp16=args.fp16, int8=args.int8,
        min_shape=args.min_shape, opt_shape=args.opt_shape,
        max_shape=args.max_shape, max_workspace=args.workspace,
        trtexec_path=args.trtexec
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
