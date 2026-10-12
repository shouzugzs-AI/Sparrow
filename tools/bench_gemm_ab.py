#!/usr/bin/env python3
"""
GEMM A/B 基准 —— 成员 D（性能）专用工具

做三件事：
  1) 用 vmlm-bench（tests/src/bench.rs）在**同一台机器、同一构建配置**下
     对 GEMM_NT 内核做多次采样，输出 min / median / p90（min 最抗噪）。
  2) 跑 vmlm-runtime examples/bench_gemm.vmb，给出契约表口径的 us/iter
     （这个数含 VM 装载后完整 exec 调度，与内核数分开汇报）。
  3) 用 torch.set_num_threads(1) 测 PyTorch 单核同形状 GEMM，做倍率对比。

用法:
    python bench_gemm_ab.py                 # 默认 M=N=K=512，各 5 轮
    python bench_gemm_ab.py --m 512 --n 512 --k 512 --rounds 5
    python bench_gemm_ab.py --tag baseline  # 结果追加写入 bench_ab_log.jsonl

设计要点（避免踩过的坑）：
  * 绝不用单次采样：这台 i3-12100F 上 max/min 能差 3 倍以上（后台调度 + 睿频），
    单次采样会得出与真实性能无关的结论。至少 5 轮 × 200 次。
  * 报告 min 与 median 两个值：噪声只会让耗时变长，所以 min 最接近内核的真实速度；
    median 反映「典型」情况。两者差距大 = 机器噪声大，不是内核不稳定。
  * PyTorch 也跑多轮取 min，否则对比不公平。
  * 不同 build 之间必须重新编译（cargo 会自己判断），脚本只调用 cargo 构建，
    不做 cargo clean、不删 target/。
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EX = os.path.join(ROOT, "examples")


def run(cmd, cwd=None, timeout=900):
    p = subprocess.run(
        cmd, cwd=cwd or ROOT, capture_output=True, text=True, timeout=timeout
    )
    return p.returncode, p.stdout, p.stderr


def build_bench():
    """构建内核微基准（tests crate，独立 bin，不影响 vmlm-tests）。"""
    rc, out, err = run(
        ["cargo", "build", "-p", "vmlm-tests", "--release", "--bin", "vmlm-bench"]
    )
    if rc != 0:
        print(err[-3000:])
        sys.exit("构建 vmlm-bench 失败")


def build_runtime_bench():
    """构建带 bench 特性的 runtime（计时/argv 基础设施）。"""
    rc, out, err = run(["cargo", "build", "--release", "--features", "bench"], cwd=os.path.join(ROOT, "runtime"))
    if rc != 0:
        print(err[-3000:])
        sys.exit("构建 vmlm-runtime --features bench 失败")


def kernel_samples(m, n, k, reps, rounds):
    """调用 vmlm-bench 多次，每次输出一行解析。"""
    exe = os.path.join(ROOT, "target", "release", "vmlm-bench.exe")
    mins, meds = [], []
    for _ in range(rounds):
        rc, out, err = run([exe, str(m), str(n), str(k), str(reps)])
        if rc != 0:
            print(out, err)
            sys.exit("vmlm-bench 运行失败")
        line = [l for l in out.splitlines() if l.startswith("GEMM_NT")][0]
        # GEMM_NT   512x512  x512   reps=400 median=..us min=..us p90=..us max=..us
        parts = dict(
            tok.split("=", 1) for tok in line.split() if "=" in tok
        )
        mins.append(float(parts["min"].rstrip("us")))
        meds.append(float(parts["median"].rstrip("us")))
    return mins, meds


def runtime_samples(path, reps, rounds):
    """跑 vmlm-runtime <vmb> <reps>，取每次的 avg us/iter。"""
    exe = os.path.join(ROOT, "target", "release", "vmlm-runtime.exe")
    vals = []
    for _ in range(rounds):
        rc, out, err = run([exe, path, str(reps)])
        if rc != 0:
            print(out, err)
            sys.exit("vmlm-runtime 运行失败（注意：默认特征构建不认参数，需 --features bench）")
        for l in out.splitlines():
            if l.startswith("avg us/iter"):
                vals.append(float(l.split("=")[1].strip()))
    return vals


def torch_bench(m, n, k, rounds, reps):
    """PyTorch 单核同形状：A[m,k] @ B[n,k]^T。"""
    try:
        import numpy as np
        import torch
    except ImportError as e:
        return None, str(e)
    torch.set_num_threads(1)
    rng = np.random.default_rng(7)
    A = (rng.standard_normal((m, k)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((n, k)) * 0.1).astype(np.float32)
    tA, tB = torch.from_numpy(A), torch.from_numpy(B)
    for _ in range(5):
        _ = tA @ tB.T
    mins = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(reps):
            _ = tA @ tB.T
        t1 = time.perf_counter()
        mins.append((t1 - t0) / reps * 1e6)
    return mins, None


def gflops(m, n, k, us):
    return 2.0 * m * n * k / (us / 1e6) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--reps", type=int, default=200, help="vmlm-bench 每次调用的采样数")
    ap.add_argument("--rounds", type=int, default=5, help="独立进程轮数")
    ap.add_argument("--rt-reps", type=int, default=200, help="runtime 每次 avg 的迭代数")
    ap.add_argument("--torch-reps", type=int, default=50)
    ap.add_argument("--tag", default="", help="写入 bench_ab_log.jsonl 的标签")
    ap.add_argument("--skip-torch", action="store_true")
    ap.add_argument("--skip-runtime", action="store_true")
    a = ap.parse_args()

    m, n, k = a.m, a.n, a.k
    fl = 2.0 * m * n * k
    print(f"=== GEMM A/B 基准  M={m} N={n} K={k}  ({fl/1e6:.1f} MFLOP) ===")
    print(f"     rounds={a.rounds}  bench-reps={a.reps}  rt-reps={a.rt_reps}")

    out = {"tag": a.tag, "shape": [m, n, k], "rounds": a.rounds}

    # ---- 1) 内核级 ----
    build_bench()
    kmins, kmeds = kernel_samples(m, n, k, a.reps, a.rounds)
    print("\n[1] 内核级 vmlm-bench（只测 gemm_nt，不含 VM exec）")
    print(f"    min    us: {[round(x,1) for x in kmins]}")
    print(f"    median us: {[round(x,1) for x in kmeds]}")
    print(f"    best  = {min(kmins):.1f} us  ({gflops(m,n,k,min(kmins)):.2f} GFLOPS)")
    print(f"    med-of-med = {statistics.median(kmeds):.1f} us  ({gflops(m,n,k,statistics.median(kmeds)):.2f} GFLOPS)")
    out["kernel_min_us"] = min(kmins)
    out["kernel_med_us"] = statistics.median(kmeds)
    out["kernel_min_gflops"] = gflops(m, n, k, min(kmins))
    out["kernel_med_gflops"] = gflops(m, n, k, statistics.median(kmeds))

    # ---- 2) 端到端 ----
    if not a.skip_runtime:
        build_runtime_bench()
        vmb = os.path.join(EX, "bench_gemm.vmb")
        rvals = runtime_samples(vmb, a.rt_reps, a.rounds)
        print("\n[2] 端到端 vmlm-runtime bench_gemm.vmb（含 exec 调度）")
        print(f"    avg us/iter: {[round(x,1) for x in rvals]}")
        print(f"    best = {min(rvals):.1f} us  ({gflops(m,n,k,min(rvals)):.2f} GFLOPS)")
        out["e2e_min_us"] = min(rvals)
        out["e2e_min_gflops"] = gflops(m, n, k, min(rvals))

    # ---- 3) PyTorch ----
    if not a.skip_torch:
        tv, err = torch_bench(m, n, k, a.rounds, a.torch_reps)
        if tv is None:
            print(f"\n[3] PyTorch 不可用: {err}")
        else:
            print("\n[3] PyTorch 单核 torch.set_num_threads(1)")
            print(f"    us: {[round(x,1) for x in tv]}")
            pt = min(tv)
            print(f"    best = {pt:.1f} us  ({gflops(m,n,k,pt):.2f} GFLOPS)")
            out["torch_min_us"] = pt
            out["torch_min_gflops"] = gflops(m, n, k, pt)
            if "kernel_min_us" in out:
                out["ratio_kernel_vs_torch"] = pt / out["kernel_min_us"]
                print(f"    相对倍率（内核 best / torch best 的倒数）: "
                      f"{pt / out['kernel_min_us']:.3f}x")

    log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_ab_log.jsonl")
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"\n结果已追加到 {log}")


if __name__ == "__main__":
    main()
