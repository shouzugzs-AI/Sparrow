#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P9-J：ESERNN 训练步的**内存上限推算**（回答 Lead：Adam 状态换成 f16 / 按需分配能推到多大）。

## 为什么内存是放大的真瓶颈（而不是 `.vmb` 体积）

`Vm::load` 在装载时一次性分配 5 块区域（`core/src/interp.rs:86-116`）：

    act    = alloc(act_bytes)        # 激活区（栈/中间量），与配置相关但很小
    state  = alloc(state_bytes)
    grad   = alloc(grad_bytes)       # 与**常量池同布局** ⇒ ≈ 参数量 × 4B
    adam_m = alloc(grad_bytes)       # 引擎无条件各分配一份
    adam_v = alloc(grad_bytes)
    （每块还多要 64B 对齐余量）

⇒ **装载峰值 ≈ 常量池 + 激活/状态 + 3 × 参数量×4B**
⇒ **≈ 参数量 × 16 字节**（4 份 f32 参数 + 极小的 act/state）

> ⚠ **勘误**：我最初在汇报里写的是「15 字节/参数」，**精确值是 16.1**
> （4 份 × 4B + 激活/状态摊销）。已更正。这是 4 份副本，不是 3 份：
> 常量池本身也算一份（`grad`/`adam_m`/`adam_v` 是**额外的**三份）。

## 本脚本要算的三件事

1. **精确基数**：逐档算出 4 块参数相关区域的真实字节数
2. **f16 Adam 状态**能推到多大（adam_m/v 用 2B 而非 4B）
3. **按需分配**（只在真要训练时分配 Adam 状态）能推到多大

## 用法
    python tools\\eserNN_mem_projection.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eserNN_train_matrix import CFGS, param_count, bptt_footprint   # noqa: E402

MB = 1048576.0
#: 真实 ESERNN base 配置的真实 vocab（P8-C2 用 3379 是为了对齐 P7-C 口径）
VOCAB_C2 = 3379


def projection(cfg):
    """返回三种分配策略下的装载峰值（字节）。"""
    p = param_count(cfg) * 4          # 常量池 = 参数量 × 4B
    fp = bptt_footprint(cfg)
    act_state = fp["stack"]           # 激活/状态区（栈是主要部分）
    # 策略 A（现状）：grad + adam_m + adam_v 各一份 f32，且与常量池同布局
    A = p + act_state + 3 * p
    # 策略 B：Adam m/v 用 f16 存（省一半）
    B = p + act_state + p + 2 * (p // 2)
    # 策略 C：Adam 状态按需分配（推理/梯度检查时不分配）
    C = p + act_state + p
    # 策略 D：B + C（按需 + f16）
    D = p + act_state + p + 2 * (p // 4)
    return dict(params=param_count(cfg), const=p, act=act_state,
                A=A, B=B, C=C, D=D)


def headroom(budget_bytes, act_ratio):
    """在给定内存预算下，最大可支持的参数量。

    act_ratio：激活/状态区占参数相关内存的比例（实测 h512/l6 约 1.3MB/266MB ≈ 0.5%，
    这里保守取 1%）。
    """
    # budget = params*4 * (1 + copies) * (1 + act_ratio)
    return budget_bytes / (4.0 * (1.0 + act_ratio))


def main():
    print("=" * 92)
    print("ESERNN 训练步装载峰值内存：现状 vs 优化（按 Vm::load 的真实分配逻辑推算）")
    print("=" * 92)
    print()
    print(f"{'档位':>9} {'参数量':>13} " +
          f"{'A 现状':>10} {'B f16Adam':>11} {'C 按需':>10} {'D 两者':>10}")
    print("-" * 92)
    rows = []
    for tag, cfg in CFGS:
        r = projection(cfg)
        rows.append((tag, cfg, r))
        print(f"{tag:>9} {r['params']:>13,} "
              f"{r['A']/MB:>9.1f}M {r['B']/MB:>10.1f}M "
              f"{r['C']/MB:>9.1f}M {r['D']/MB:>9.1f}M")

    print()
    print("说明：")
    print("  A 现状   = 常量池 + 激活 + grad + adam_m + adam_v（全部 f32，与常量池同布局）")
    print("  B f16    = Adam m/v 用 f16 存储（计算时转 f32）")
    print("  C 按需   = 只在真正训练时分配 Adam 状态（推理/梯度检查不分配）")
    print("  D 两者   = f16 + 不分配 adam_v？否：D = 按需 + m/v 都 f16")
    print()

    # ---- 上限推算：以 32 位进程可用 ~2 GB / 4 GB 为预算 ----
    print("=" * 92)
    print("内存预算下的「最大可支持参数量」（设激活开销 1%）")
    print("=" * 92)
    print()
    act_ratio = 0.01
    print(f"{'策略':<18} {'2 GB 进程':>16} {'4 GB 进程':>16} {'对应 hidden（6 层, inter=3H）':>28}")
    print("-" * 92)
    # 每档的“参数量/字节”系数
    for name, copies in (("A 现状 (3 副本 f32)", 4.0),
                         ("B f16 Adam (2 副本)", 3.0),
                         ("C 按需 (1 副本)", 2.0),
                         ("D 按需+f16 (1.5 副本)", 2.5)):
        for budget_gb in (2.0, 4.0):
            pass
        n2 = headroom(2.0 * 1024 ** 3, act_ratio) / copies
        n4 = headroom(4.0 * 1024 ** 3, act_ratio) / copies
        # 反推 hidden：params ≈ (V*H) + NL*(...)，用二分找最大 H
        def max_hidden(budget_params):
            lo, hi = 64, 8192
            while lo < hi:
                mid = (lo + hi + 1) // 2
                c = dict(vocab=VOCAB_C2, hidden=mid, heads=8, head_dim=mid // 8,
                         inter=mid * 3, decay_hidden=32, eps=1e-6, T=6, layers=6)
                if param_count(c) <= budget_params:
                    lo = mid
                else:
                    hi = mid - 1
            return lo
        h2 = max_hidden(n2)
        h4 = max_hidden(n4)
        print(f"{name:<18} {n2/1e9:>13.2f} G参 {n4/1e9:>13.2f} G参 "
              f"{'h%d / h%d' % (h2, h4):>28}")

    print()
    print("结论见 examples/eserNN/train/large/README.md §9。")


if __name__ == "__main__":
    main()
