"""单算子「经引擎」隔离测试（Lead 维护）。

## 为什么需要这个

现有套件里 `vmlm-lead-verify` 是**直接调用 Rust 函数**（绕过引擎）。
而引擎路径 —— 槽解析、`dt` 取值、相邻槽约定、形状声明 —— **没有任何常驻测试**。

P9 成员 I 的经历证明了这个空白是真实的：
他的 `layernorm_bwd` 直接调函数测出 **5e-16**，但整网反向有 2% 的均匀偏差。
排查了很久才发现差异不在数学公式里，而在 `interp.rs` 的**一行槽解析**。

> **测「算子」和测「算子在引擎里的调用路径」是两件事。**

## 本脚本做什么

对单个 `LN` + `LN_BWD`：
  1. 生成常量输入（x / gamma / beta / dy）
  2. 写一个最小 `.vml`
  3. 调真正的 `vmlm-asm` 汇编
  4. 调真正的**引擎**跑（不是直接调 Rust 函数）
  5. 与 numpy f64 解析值对比

用 `[4,8]` 而不是 `[1,8]`，以覆盖 `dg`/`db` 的**跨行归约**。

## 关键设计点
- 用 `SHAPE` 把 `gamma / dgamma / dbeta` 钉成**紧邻三连**（引擎按 `gamma+1`/`gamma+2` 取值）
- 发 `LN_EPS` 显式设定 eps
- 所有路径都用绝对路径，**可从任意 cwd 调用**

用法: python tools/ln_bwd_engine_check.py
"""
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

ROOT = str(Path(__file__).resolve().parents[1])
ASM = os.path.join(ROOT, 'target', 'release', 'vmlm-asm.exe')
RUN = os.path.join(ROOT, 'target', 'release', 'vmlm-runtime.exe')

ROWS, COLS = 4, 8
EPS = 1e-6


def wr(path, arr):
    np.asarray(arr, dtype=np.float32).tofile(path)


def main():
    work = tempfile.mkdtemp(prefix='ln_iso_')
    rng = np.random.default_rng(12345)

    x = (rng.standard_normal((ROWS, COLS)) * 0.7 + 0.3).astype(np.float32)
    g = (rng.standard_normal((ROWS, COLS)) * 0.2 + 1.0).astype(np.float32)  # gamma 逐行同值
    g = np.repeat(g[:1, :], ROWS, axis=0)
    b = np.repeat((rng.standard_normal((1, COLS)) * 0.1).astype(np.float32), ROWS, axis=0)
    dy = (rng.standard_normal((ROWS, COLS)) * 0.5).astype(np.float32)
    eps = np.array([EPS], dtype=np.float32)

    # ---- numpy f64 解析 + 数值自检 ----
    xd = x.astype(np.float64)
    gd = g.astype(np.float64)
    bd = b.astype(np.float64)
    dyd = dy.astype(np.float64)

    mu = xd.mean(axis=-1, keepdims=True)
    var = xd.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + EPS)
    xs = (xd - mu) * inv
    y = xs * gd + bd

    db_ana = dyd.sum(axis=0, keepdims=True)
    dg_ana = (dyd * xs).sum(axis=0, keepdims=True)
    gdy = gd * dyd
    dx_ana = (gdy - gdy.mean(axis=-1, keepdims=True)
              - xs * (gdy * xs).mean(axis=-1, keepdims=True)) * inv

    # gamma/beta 只对第 0 行求梯度（逐行同值，引擎会把 4 行都累加）
    dg_ana_1row = dg_ana
    db_ana_1row = db_ana

    # ---- 写 .vml ----
    vml = '''.section forward
    SHAPE       x,    4, 8
    SHAPE       y,    4, 8
    SHAPE       g,    1, 8
    SHAPE       dg,   1, 8
    SHAPE       db,   1, 8
    SHAPE       dy,   4, 8
    SHAPE       dx,   4, 8
    LN_EPS      eps
    LN          y, x, g, b
    ZERO        dg
    ZERO        db
    LN_BWD      dx, dy, x, g
    PRINT       dg
    PRINT       db
    PRINT       dx
    HALT
'''
    vml_p = os.path.join(work, 'ln_iso.vml')
    with open(vml_p, 'w', encoding='utf-8', newline='\n') as f:
        f.write(vml)

    for name, arr, r, c in [('x', x, ROWS, COLS), ('g', g[:1, :], 1, COLS),
                            ('b', b[:1, :], 1, COLS), ('dy', dy, ROWS, COLS),
                            ('eps', eps, 1, 1)]:
        wr(os.path.join(work, name + '.bin'), arr)

    vmb = os.path.join(work, 'ln_iso.vmb')
    # 用正斜杠：反斜杠在命令行里可能被解释为转义（P9 踩过同类坑）
    wf = work.replace('\\', '/')
    args = [ASM, vml_p, vmb,
            f'x:{wf}/x.bin:{ROWS}:{COLS}',
            f'g:{wf}/g.bin:1:{COLS}',
            f'b:{wf}/b.bin:1:{COLS}',
            f'dy:{wf}/dy.bin:{ROWS}:{COLS}',
            f'eps:{wf}/eps.bin:1:1']
    r = subprocess.run(args, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    if r.returncode != 0 or 'OK:' not in (r.stdout or ''):
        print('ASM FAILED:', r.stdout, r.stderr)
        return 1

    # ---- 跑引擎 ----
    r2 = subprocess.run([RUN, vmb], capture_output=True, text=True,
                        encoding='utf-8', errors='replace')
    nums = []
    for tok in (r2.stdout or '').split():
        try:
            nums.append(float(tok))
        except ValueError:
            pass

    need = COLS + COLS + ROWS * COLS
    if len(nums) < need:
        print('ENGINE OUTPUT SHORT:', r2.stdout, r2.stderr)
        return 1

    e_dg = np.array(nums[:COLS])
    e_db = np.array(nums[COLS:2 * COLS])
    e_dx = np.array(nums[2 * COLS:2 * COLS + ROWS * COLS]).reshape(ROWS, COLS)

    # 引擎把 gamma 的梯度按 4 行累加（gamma 是 [1,8] 广播到 [4,8]）
    dg_ref = dg_ana.ravel()
    db_ref = db_ana.ravel()
    dx_ref = dx_ana

    def rel(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        return np.sum(np.abs(a - b)) / (np.sum(np.abs(a)) + np.sum(np.abs(b)) + 1e-300)

    print('=' * 66)
    print('单算子「经引擎」隔离测试：LN_BWD')
    print(f'形状 [{ROWS},{COLS}]  eps={EPS}')
    print('=' * 66)
    rdg, rdb, rdx = rel(dg_ref, e_dg), rel(db_ref, e_db), rel(dx_ref, e_dx)
    print(f'  dgamma  rel = {rdg:.3e}')
    print(f'  dbeta   rel = {rdb:.3e}')
    print(f'  dx      rel = {rdx:.3e}')
    print()
    ok = max(rdg, rdb, rdx) < 1e-5
    print(f'判据 < 1e-5 (f32 量级) : {"PASS" if ok else "FAIL"}')
    print()
    print('说明：本测试走**真引擎**（.vml -> vmlm-asm -> vmlm-runtime），')
    print('      覆盖槽解析 / dt 取值 / gamma+1,gamma+2 相邻约定 / 形状声明。')
    print('      它补上了 vmlm-lead-verify（直接调 Rust 函数）未覆盖的路径。')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
