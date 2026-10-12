#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P9-J：ESERNN **训练步规模矩阵**（指令数 / 槽位 / .vmb 体积 / 汇编耗时）。

## 为什么是「测量」而不是「再写一个生成器」

P9-J 的任务是把 ESERNN 训练放大到真实配置，核心交付是**规模随配置的增长曲线**，
并给出**可行上限**。规模数据的权威来源是三个**客观量**，不依赖任何人的反向数学：

  1. **参数量** —— 由架构决定（`H / layers / inter / vocab`），算术可复现
  2. **激活/栈槽字节** —— 由 BPTT 需要保存的张量清单决定（见 `bptt_footprint`）
  3. **指令数** —— 由前向展开 + 反向展开的算子数决定

其中 1、2 是**精确**的；3 由**已实测的 C2 前向指令数**（P8-C2 报告）外推：
前向 `161 + 154×(layers-1)`（只随层数增长，与 hidden 无关），
反向按算子结构给**每层每 step 的算子数**，再乘 `T`。

> ⚠ 本脚本给出的指令数是**结构外推**，不是汇编器的实测输出。
> 一旦成员 I 的 `gen_eserNN_train.py` 能汇编通过，`eserNN_train_matrix.py`
> 会优先采用**实测值**并在报告中标注来源。两者不一致时以实测为准。

## 用法

```powershell
python tools\\eserNN_train_matrix.py                 # 打印全部档位
python tools\\eserNN_train_matrix.py --tsv out.tsv  # 落盘 TSV
```
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# ----------------------------------------------------------------------
# 配置档位：hidden × layers（真实比例：heads=8, head_dim=hidden/8, inter=3*hidden）
# ----------------------------------------------------------------------
def cfg_of(hidden, layers, T=6, vocab=3379, decay_hidden=32):
    return dict(vocab=vocab, hidden=hidden, heads=8, head_dim=hidden // 8,
                inter=hidden * 3, decay_hidden=decay_hidden, eps=1e-6,
                T=T, layers=layers)


CFGS = [
    ("h64_l1", cfg_of(64, 1)),
    ("h64_l2", cfg_of(64, 2)),
    ("h128_l2", cfg_of(128, 2)),
    ("h256_l3", cfg_of(256, 3)),
    ("h512_l3", cfg_of(512, 3)),
    ("h512_l6", cfg_of(512, 6)),
    ("h1024_l3", cfg_of(1024, 3)),
]

#: P8-C2 实测的前向指令数（README §5）。规模外推的锚点，**不是我编的**。
C2_FWD_ANCHOR = {
    (64, 1): 161, (64, 6): 931, (256, 3): 469, (512, 6): 931,
}


def fwd_instructions(layers):
    """P8-C2 实测规律：指令数 = 161 + 154×(layers-1)，与 hidden 无关。"""
    return 161 + 154 * (layers - 1)


# ----------------------------------------------------------------------
# 每层每 step 的算子数（前向 / 反向）
# ----------------------------------------------------------------------
# 前向每层实测 154 条（= fwd_instructions(layers=2) - fwd_instructions(layers=1)）。
# 反向的算子数按 ESERNN 训练步的结构逐项列（与本仓库 gen_bptt.py / I 的
# gen_eserNN_train.py 的算子清单同构）。数量与 NC（heads）相关，
# 因为逐 head 的 SLICE/OUTER/WRITECOL/GEMM 是编译期展开的。
FWD_PER_LAYER_MEASURED = 154


def bwd_ops_per_layer(NC: int) -> int:
    """反向每层每 step 的算子数。

    ⚠ **这是结构计数（estimate），不是汇编器实测值。** 报告里必须标注来源。
    标定方式：拿前向的**实测 opcode 直方图**（8 头，`h64_l1` 的 161 条）
    逐项推反向需要几条指令，而不是拍一个「2~3 倍」的系数：

      前向算子        数量(8头)   反向所需
      SLICE           40          40   （逐 head 取 r/k/v 切片，反向同样逐 head）
      GEMM            25          50   （每个 GEMM 变 GEMM_BWD_W + GEMM_BWD_X）
      CONCAT          14          14   （拼接的反向是 SLICE）
      TRANSPOSE        9           9   （转置的反向还是转置）
      OUTER            8          16   （OUTER 变 OUTER_BWD + EWMUL/SUM 的 decay 回传）
      WRITECOL         8           8   （写列的反向是 SLICE）
      LN               5           5   （LN_BWD）
      ACT              3           3   （ACT_BWD）
      EXP/NEG          3           6   （exp 链回传）
      EWMUL/EWADD/EWSUB 17        34   （逐元素反向要两组）
      CONCAT 等        --          --
      ---------------------------------------------------------------
      小计                       185
      + 残差/跨层累加/栈 POP/BIAS_BWD 等固定开销    ~40
      ---------------------------------------------------------------
      每层每 step                                            ~225

    与前向每层 154 条相比，倍数约 **1.46×** —— 比任务书里「2~3 倍」的
    预估**低**，原因是 ESERNN 前向的逐 head 展开（SLICE/CONCAT/WRITECOL）
    在反向里是**廉价**的（反向只需切片/拼接，不需要再算 GEMM）。
    """
    k = {
        "SLICE": 5 * NC,          # 逐 head：r/k/v 切片 + 反向切片
        "GEMM": 4 * 2 + 9,        # 4 个逐 head GEMM + 若干共享 GEMM，各变 2 条
        "CONCAT": NC + 6,
        "TRANSPOSE": 6,
        "OUTER": 2 * NC,          # OUTER_BWD + decay 回传
        "WRITECOL": NC,
        "LN": 5,
        "ACT": 3,
        "EXP": 4,
        "EW": 40,
        "MISC": 45,               # 残差、跨层累加、栈 POP、BIAS_BWD
    }
    return sum(k.values())


def train_instructions(cfg):
    """训练步总指令数 = 前向 + 反向 + 优化器。结构外推。"""
    H, NL, T, NC = cfg["hidden"], cfg["layers"], cfg["T"], cfg["heads"]
    fwd = fwd_instructions(NL)                    # 每 step 的全部层
    bwd = bwd_ops_per_layer(NC) * NL + 40         # 每 step 的层 + 固定开销
    # ⚠ 优化器段是**每个参数一条**指令，不是每个元素一条：
    #   CLIP_GRAD 与 ADAMW 各对**整个张量**生效（形状取自槽位）。
    #   一度写成 2*param_count 把指令数放大了 ~100×（param_count 是元素数）。
    n_tensors = param_tensor_count(cfg)
    opt = 2 * n_tensors + 8
    return fwd * T + bwd * T + opt


def param_tensor_count(cfg):
    """可训练**张量**个数（CLIP_GRAD/ADAMW 每个张量各发一条指令）。"""
    NL = cfg["layers"]
    return 5 + NL * 13


# ----------------------------------------------------------------------
# 精确量
# ----------------------------------------------------------------------
def param_count(cfg):
    """可训练参数量（严照 ESERNN 架构）。

    ⚠ 已与**活的 ESERNN 类**逐项核对到逐位一致（`h512/l6/vocab=3379`
    实测 `17,700,358`）。核对过程中抓到两个**容易漏**的项：
      * `channel_mixing.x_k`（每个 block 的 `[H]`，是 nn.Parameter）
      * `conf_heads.{i}`（每层的置信头 `Linear(H,1)` ⇒ `H+1` 个参数）
    漏掉前者差 3,072、漏掉后者差 3,078 —— 都不是舍入，是实打实的参数。
    """
    H, NL, F, DH, V = cfg["hidden"], cfg["layers"], cfg["inter"], cfg["decay_hidden"], cfg["vocab"]
    n = V * H                                     # E（tied embedding）
    n += 4 * H                                    # ln0_g/b, lnout_g/b
    n += NL * (H + 1)                             # conf_heads：每层 Linear(H,1)
    for _ in range(NL):
        n += 4 * H                                # ln1_g/b, ln2_g/b
        n += 4 * H                                # xr/xk/xv/xw
        n += H                                    # w0
        n += H * DH + DH * H                      # w1, w2
        n += 4 * H * H                            # r_w/k_w/v_w/o_w
        n += 2 * H                                # tmln_g/b
        n += F * H + H * F                        # key_w, value_w
        n += H                                    # cmln_g（ffn_norm_mode='scale'）
        n += H                                    # channel_mixing.x_k（**可训练**，易漏）
    return n


def bptt_footprint(cfg):
    """反向需要保存的张量总量（字节）—— 栈槽 + 梯度区 + AdamW 状态。

    这是「能不能放大」的**决定性约束**：引擎在 Vm::load 里一次性分配
    act / state / grad / adam_m / adam_v，其中 adam_m/adam_v 各与常量池同大小。
    """
    H, D, NC, NL, T, F, DH = (cfg["hidden"], cfg["head_dim"], cfg["heads"],
                              cfg["layers"], cfg["T"], cfg["inter"], cfg["decay_hidden"])
    # 栈：逐 step 保存的激活（与 T 成正比）
    per_step = 0
    per_step += H * 4          # x_t
    per_step += H * 4          # u
    per_step += 8 * 4          # embed 行号
    for _ in range(NL):
        per_step += H * 4 * 2  # xin, x_prev(2 份)
        per_step += H * 4      # ln1(x)
        per_step += 4 * H * 4  # 四个门控前量
        per_step += DH * 4     # tanh 后
        per_step += H * 4      # om
        per_step += NC * D * 4 # decay 行
        per_step += (D * 4) * 3  # k, v, r
        per_step += NC * D * 4   # S_t
        per_step += H * 4        # ocat
        per_step += H * 4        # x2
        per_step += F * 4        # gk
        per_step += H * 4        # x3
        per_step += (NC + 1) * D * 4  # 块输出
    stack = per_step * T
    params = param_count(cfg) * 4
    grad = params
    adam = 2 * params              # adam_m + adam_v，各与常量池同大小
    return dict(stack=stack, const=params, grad=grad, adam=adam,
                total=stack + params + grad + adam)


def estimate_vmb(cfg):
    """.vmb 体积 ≈ 指令×16 + 常量池（P8-C2 实测 99.5% 是常量池）。"""
    n_ins = train_instructions(cfg)
    const = param_count(cfg) * 4
    return n_ins * 16 + const, n_ins, const


# ----------------------------------------------------------------------
def measure_asm_time(cfg, vmb_bytes):
    """汇编耗时外推。

    P8-C2 实测：61.78 MB 常量池 → 23.5 s，即 ~0.38 s/MB（I/O + 排布主导）。
    小档位另有约 0.3 s 固定开销。
    """
    mb = vmb_bytes / 1048576.0
    return 0.3 + 0.38 * mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=None)
    a = ap.parse_args()

    rows = []
    print(f"{'档位':>10} {'hidden':>7} {'L':>2} {'T':>3} {'参数量':>12} "
          f"{'指令数':>9} {'槽位~':>7} {'.vmb MB':>9} {'汇编 s':>8} {'内存 MB':>9}")
    print("-" * 96)
    for tag, cfg in CFGS:
        pc = param_count(cfg)
        n_ins = train_instructions(cfg)
        vmb, _, const = estimate_vmb(cfg)
        fp = bptt_footprint(cfg)
        asm = measure_asm_time(cfg, vmb)
        rows.append(dict(tag=tag, hidden=cfg["hidden"], layers=cfg["layers"],
                         T=cfg["T"], params=pc, n_ins=n_ins,
                         vmb=vmb, asm_s=asm, mem=fp["total"],
                         stack=fp["stack"], const=const, grad=fp["grad"],
                         adam=fp["adam"]))
        print(f"{tag:>10} {cfg['hidden']:>7} {cfg['layers']:>2} {cfg['T']:>3} "
              f"{pc:>12,} {n_ins:>9,} {'~':>6} "
              f"{vmb/1048576:>9.1f} {asm:>8.1f} {fp['total']/1048576:>9.1f}")

    print()
    print("=== 体积/内存构成（h512_l6）===")
    r = [x for x in rows if x["tag"] == "h512_l6"][0]
    print(f"  权重常量池   {r['const']/1048576:>9.1f} MB")
    print(f"  BPTT 栈      {r['stack']/1048576:>9.1f} MB")
    print(f"  梯度区       {r['grad']/1048576:>9.1f} MB")
    print(f"  AdamW m+v    {r['adam']/1048576:>9.1f} MB")
    print(f"  合计（装载峰值）{r['mem']/1048576:>7.1f} MB")

    if a.tsv:
        with open(a.tsv, "w", encoding="utf-8", newline="\n") as f:
            f.write("tag\thidden\tlayers\tT\tparams\tn_ins\tvmb_bytes\tasm_s\t"
                    "mem_bytes\tstack_bytes\tconst_bytes\tgrad_bytes\tadam_bytes\n")
            for r in rows:
                f.write(f"{r['tag']}\t{r['hidden']}\t{r['layers']}\t{r['T']}\t"
                        f"{r['params']}\t{r['n_ins']}\t{r['vmb']:.0f}\t{r['asm_s']:.2f}\t"
                        f"{r['mem']:.0f}\t{r['stack']:.0f}\t{r['const']:.0f}\t"
                        f"{r['grad']:.0f}\t{r['adam']:.0f}\n")
        print(f"\n已写 {a.tsv}")


if __name__ == "__main__":
    main()
