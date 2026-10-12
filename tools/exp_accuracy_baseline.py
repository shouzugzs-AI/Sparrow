"""exp_f 精度基线测量（Lead 维护）。

用途：给成员 H 一个**独立的**精度基准，避免只看他自己写的扫描脚本
（P8 的教训：测试工具自己有 bug 时会误导）。

本脚本用 Python 复刻 VMLM `core/src/kernels.rs` 里 `exp_f` 的算法，
逐位模拟 f32 行为，然后与 numpy 的 f64 `exp` 比较。

注意：这是**复刻**，不是调用真引擎。若与成员 H 用真引擎测出的数字不符，
以真引擎为准 —— 那说明复刻有偏差，应当修正本脚本。
"""
import numpy as np
import struct


def f32(x):
    return np.float32(x)


def exp_f_replica(x):
    """复刻当前 core/src/kernels.rs 的 exp_f 实现（逐位模拟 f32）。"""
    x = f32(x)
    if x < f32(-88.0):
        x = f32(-88.0)
    elif x > f32(88.0):
        x = f32(88.0)

    n = int(f32(f32(x * f32(1.442_695_04)) + f32(0.5)))
    r = f32(x - f32(f32(n) * f32(0.693_147_18)))
    r2 = f32(r * r)

    p = f32(1.0)
    p = f32(p + r)
    p = f32(p + f32(r2 * f32(0.5)))
    p = f32(p + f32(f32(r2 * r) * f32(0.166_666_67)))
    p = f32(p + f32(f32(r2 * r2) * f32(0.041_666_668)))
    p = f32(p + f32(f32(r2 * r2 * r) * f32(0.008_333_334)))
    p = f32(p + f32(f32(r2 * r2 * r2) * f32(0.001_388_889)))

    bits = np.uint32((n + 127) & 0xFFFFFFFF) << np.uint32(23)
    scale = struct.unpack('<f', struct.pack('<I', int(bits)))[0]
    return f32(p * f32(scale))


def scan(lo=-20.0, hi=20.0, npts=1000001):
    xs = np.linspace(lo, hi, npts)
    ref = np.exp(xs.astype(np.float64))
    got = np.array([exp_f_replica(v) for v in xs], dtype=np.float64)
    denom = np.maximum(np.abs(ref), 1e-300)
    rel = np.abs(got - ref) / denom
    i = int(np.argmax(rel))
    return rel.max(), xs[i], rel.mean()


if __name__ == '__main__':
    print('=' * 68)
    print('exp_f 精度基线（Lead 独立复刻 core/src/kernels.rs）')
    print('=' * 68)
    mx, at, mean = scan()
    print(f'采样区间 [-20, 20]，1000001 点')
    print(f'  最大相对误差 = {mx:.6e}  @ x = {at:+.6f}')
    print(f'  平均相对误差 = {mean:.6e}')
    print()
    print('对照 P8/P9 记录的值：6.51e-4 @ x=-5.199')
    print('  （若本脚本结果与之不符，以真引擎实测为准，本复刻需修正）')
    print()
    print('目标（成员 H）：最大相对误差 <= 1e-6')
