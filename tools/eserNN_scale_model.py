#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P9-J：ESERNN 训练步的**规模建模**脚手架（参数化展开 + 规模测量）。

## 本文件的职责边界（Lead 在 P9 裁定）

**本文件只做规模建模，不做数值正确性。**

| 文件 | 归属 | 职责 |
|---|---|---|
| `examples/eserNN/train/gen_eserNN_train.py` | 成员 I | **数值正确性**：训练步 `.vml` + 梯度检查 |
| **本文件** `tools/eserNN_scale_model.py` | 成员 J | **规模**：参数化展开 + 矩阵测量 |
| `tools/eserNN_train_matrix.py` | 成员 J | 规模矩阵汇总（7 档增长曲线） |

**任何数值/梯度结论一律以 I 的文件为准。**
本文件之所以**不自行补完反向**，是因为契约 §4 要求「复用 I 验证过的反向数学」，
而 §5.0 明确警告：**两套数学漂移正是静默降级的来源**。

## ⚠ 状态（先读这一段）

**前向：完整展开（严格照 `ESERNN.forward_train`）。反向：未完成（`_bwd_step` 为空）。**

⇒ 本脚本**不能用于训练**，也**没有**产生过任何数值结果。
它的交付价值是规模脚手架：喂给 `tools/eserNN_train_matrix.py` 做指令数/体积建模。

## 定位（避免与 I 重复）

成员 I 的任务是「ESERNN 训练闭环」的**正确性**（最小配置），
我的任务是他的**放大版**：把训练步推到真实配置，量化体积/指令数/汇编耗时。

本脚本因此**有意与 I 的 `examples/eserNN/train/gen_eserNN_train.py` 同接口**：

* 权重张量名、`cfg.json` 字段、`weights.args` 格式**完全一致**
  （`l{l}_` 前缀规则、`tm_w1pT`/`tm_w2pT` 转置副本、`one_r{D}` 等），
  所以 **I 的权重与驱动可以直接喂给本生成器的产物**，反之亦然。
* `.vml` 输出名同样是 `bptt_t{T}{TAG}.vml`。

目的：让「I 的最小闭环」和「J 的放大档」共用同一套接口，
放大时**不需要重写权重导出或驱动**，误差来源也就可归因。

> ⚠ 反向数学以 I 验证过的版本为准。本脚本的反向是**照 `ESERNN.forward_train`
> 源码独立展开**的（见下），用于在 I 未完成时也能给出规模数据；
> **数值正确性以 I 的闭环为准**，两侧不一致时以 I 为准。

## 前向语义：严格照 `ESERNN.forward_train`

（源码 `<ESERNN 仓库路径>/core/ese_rnn.py`，不是从 C2 的推理路径猜的）

```
# ESERNN.forward_train
x = ln0(embed(ids))                                   # [B,T,H]
for i, blk in enumerate(layers):
    prev_x, kv = states[i][0], states[i][1]
    x_prev = cat([prev_x.unsqueeze(1), x[:, :-1]], 1)  # 每层入口用**上一层输入**
    x, kv = blk.forward_train(x, x_prev, kv, H, D)

# ESERNNBlock.forward_train
h, state = time_mixing.forward_train(ln1(x), x_prev, state, H, D)
x = x + h
x = x + channel_mixing.forward_train(ln2(x), x_prev)

# TimeMixing.forward_train
xx = x_prev - x
r = sigmoid(r_w(x + xx*x_r)); k = k_w(x + xx*x_k); v = v_w(x + xx*x_v)
w = w0 + tanh((x + xx*x_w) @ w1) @ w2 ; decay = exp(-exp(w))
S_t = S_{t-1} * decay_t.unsqueeze(-1) + v_t ⊗ k_t      # decay 缩放**行** i
outs_t = S_t @ r_t                                     # [H,D]
out = ln(outs.reshape(B,T,H*D)) ; return o_w(out)

# ChannelMixing.forward_train
k = relu(key_w(x + (x_prev-x)*x_k))**2 ; out = value_w(k) * cmln_g
```

⇒ 训练档每层的 `x_prev[t]` = **上一层在 t-1 的输入**（第 0 层是 `ln0(embed)` 在 t-1）。
这与 C2 前向档（`forward_one`，x_prev = 上一 step 该层块输出）**语义不同**，
是源码本来的差异，不是 bug。

## 用法

```powershell
python tools/gen_eserNN_train.py --cfg mini  --out-dir examples/eserNN/train/large/mini
python tools/gen_eserNN_train.py --hidden 64 --heads 8 --layers 2 --T 3 `
    --out-dir examples/eserNN/train/large/h64_l2
python tools/gen_eserNN_train.py --cfg real --no-dump `
    --out-dir examples/eserNN/train/large/h512_l6
```
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# ======================================================================
# 预设
# ======================================================================
MINI = dict(vocab=6, hidden=16, heads=2, head_dim=8,
            inter=64, decay_hidden=16, eps=1e-6, T=2, layers=1)
SMALL = dict(vocab=32, hidden=64, heads=8, head_dim=8,
             inter=192, decay_hidden=16, eps=1e-6, T=3, layers=2)
REAL = dict(vocab=3379, hidden=512, heads=8, head_dim=64,
            inter=1536, decay_hidden=32, eps=1e-6, T=6, layers=6)
PRESETS = {"mini": MINI, "small": SMALL, "real": REAL}
SEED = 7
LR_DEFAULT = 3e-3
CLIP_NORM_DEFAULT = 1e9
KK_SLOTS = 512          # kk 计数索引的预留容量（见 §计数索引）


def _log(m):
    print(m, file=sys.stderr, flush=True)


class TrainGen:
    def __init__(self, cfg, seed=SEED, dump=True, tag=""):
        self.c = dict(cfg)
        c = self.c
        self.HD = int(c["hidden"]); self.NC = int(c["heads"])
        self.D = int(c["head_dim"]); self.F = int(c["inter"])
        self.DH = int(c["decay_hidden"]); self.V = int(c["vocab"])
        self.T = int(c["T"]); self.NL = int(c["layers"])
        self.EPS = float(c["eps"])
        assert self.NC * self.D == self.HD, \
            f"ESERNN 要求 hidden == n_heads*head_dim：{self.HD} vs {self.NC}*{self.D}"
        self.rng = np.random.default_rng(seed)
        self.seed = seed; self.tag = tag; self.dump_refs = dump
        self.W: dict[str, np.ndarray] = {}
        self.lines: list[str] = []
        self._shaped: set[str] = set()
        self.n_forward_lines = 0

    # ---------------- 命名（与 I 的接口一致）----------------
    def wp(self, l):
        return f"l{l}_" if self.NL > 1 else ""

    def LP(self, l):
        return f"L{l}_" if self.NL > 1 else ""

    # ---------------- 权重 ----------------
    def _rnd(self, *shape, scale=1.0):
        return (self.rng.standard_normal(shape) * scale).astype(np.float32)

    def _orth(self, rows, cols, gain):
        if rows < cols:
            a = self.rng.standard_normal((cols, rows))
            q, r = np.linalg.qr(a); q = q * np.sign(np.diag(r))
            return np.ascontiguousarray((q * gain).T.astype(np.float32))
        a = self.rng.standard_normal((rows, cols))
        q, r = np.linalg.qr(a); q = q * np.sign(np.diag(r))
        return (q * gain).astype(np.float32)

    def build_weights(self):
        HD, D, NC = self.HD, self.D, self.NC
        F, DH, V, T = self.F, self.DH, self.V, self.T
        W = self.W
        W["E"] = self._rnd(V, HD, scale=0.02)
        for nm, val in (("ln0_g", 1.0), ("ln0_b", 0.0),
                        ("lnout_g", 1.0), ("lnout_b", 0.0)):
            W[nm] = np.full((1, HD), val, np.float32)
        for l in range(self.NL):
            p = self.wp(l)
            for nm, val in (("ln1_g", 1.0), ("ln1_b", 0.0), ("ln2_g", 1.0),
                            ("ln2_b", 0.0), ("tmln_g", 1.0), ("tmln_b", 0.0),
                            ("cmln_g", 1.0)):
                W[p + nm] = np.full((1, HD), val, np.float32)
            for nm in ("xr", "xk", "xv", "xw"):
                W[p + "tm_" + nm] = np.zeros((1, HD), np.float32)
            # w0 同时导出两种布局：w0row 供前向 EWADD，w0col 供 TRANSPOSE 路径
            W[p + "tm_w0row"] = np.full((1, HD), -2.5, np.float32)
            W[p + "tm_w0col"] = np.full((HD, 1), -2.5, np.float32)
            W[p + "tm_w1p"] = self._rnd(HD, DH, scale=0.02)
            W[p + "tm_w1pT"] = np.ascontiguousarray(W[p + "tm_w1p"].T)
            W[p + "tm_w2p"] = np.zeros((DH, HD), np.float32)
            W[p + "tm_w2pT"] = np.ascontiguousarray(W[p + "tm_w2p"].T)
            for nm in ("Wr", "Wk", "Wv", "Wo"):
                W[p + "tm_" + nm] = self._orth(HD, HD, 0.1)
            W[p + "cm_Wkey"] = self._orth(F, HD, 0.05)
            W[p + "cm_Wval"] = self._orth(HD, F, 0.05)

        trng = np.random.default_rng(self.seed + 99991)
        self.TOKENS = trng.integers(0, V, size=T).astype(np.int64)
        yrng = np.random.default_rng(self.seed + 7717)
        self.Y = yrng.integers(0, V, size=T).astype(np.int64)
        for t in range(T):
            W[f"x_t{t}"] = W["E"][self.TOKENS[t]][None, :].copy()
            W[f"y{t}"] = np.array([[float(self.Y[t])]], np.float32)
        W["tok"] = self.TOKENS.astype(np.float32)
        W["x_all"] = np.stack([W[f"x_t{t}"][0] for t in range(T)]).astype(np.float32)
        W["hparams"] = np.array([[LR_DEFAULT, 0.9, 0.999, 1e-8, 1e-4]], np.float32)
        W["clip_norm"] = np.array([[CLIP_NORM_DEFAULT]], np.float32)
        W["eps1e6"] = np.array([[self.EPS]], np.float32)
        # kk 步进量：ADAMW 对 1 元素常量做 p -= lr*sign(g)
        W["kkm"] = np.array([[0.0]], np.float32)
        W["kk_step"] = np.array([[1.0]], np.float32)
        W["hp_dec"] = np.array([[1.0, 0.9, 0.999, 1e-8, 0.0]], np.float32)
        W["g_kkm"] = np.array([[1.0]], np.float32)
        W["gm1"] = np.array([[-1.0]], np.float32)
        W["one1"] = np.ones((1, 1), np.float32)
        W["one_r1"] = np.ones((1, 1), np.float32)
        W[f"one_r{D}"] = np.ones((D, 1), np.float32)
        W[f"onef_r{F}"] = np.ones((F, 1), np.float32)
        W[f"onec_c{F}"] = np.ones((1, F), np.float32)
        # eF0 / eF1：lenet 型形状控制（[F,1] / 转置后 [1,F]），省掉缩放槽
        ef = np.zeros((F, 1), np.float32); ef[0, 0] = 1.0; W["eF0"] = ef
        ef1 = np.zeros((F, 1), np.float32); ef1[F - 1, 0] = 1.0; W["eF1"] = ef1
        for h in range(NC):
            v = np.zeros((D, 1), np.float32); v[0, 0] = 1.0
            W[f"oneD_h{h}"] = v
            w = np.zeros((1, D), np.float32); w[0, 0] = 1.0
            W[f"eD0_h{h}"] = w
        # z0：长度可控的零常量（形状控制）
        W["z0"] = np.zeros((1, 1), np.float32)

    # ---------------- 形状/清单 ----------------
    def stacks(self):
        HD, D, DH, NC = self.HD, self.D, self.DH, self.NC
        return [
            ("stk_xe", HD, "x_t（embed 反向的 x / LN0 反向的 x）"),
            ("stk_u", HD, "u = ln0(x_t)"),
            ("stk_em", 1, "embed 行号"),
            ("stk_xin", HD, "每层入口 x 的 t 行"),
            ("stk_p0", HD, "每层 x_prev 在 t=0 的项（上一层输入左移）"),
            ("stk_1r", HD, "上一层输入在 t-1 的行（层 0 的 t-1 输入）"),
            ("stk_xinm", HD, "上一层输入在 t-1 的行（= L(l) 与 L(l-1) 共享）"),
            ("stk_lg", HD, "ln1(x)"),
            ("stk_xp", HD, "x_prev 在 t-1 的行（通道混合反向用）"),
            ("stk_2", HD, "残差流 x2"),
            ("stk_rp", HD, "门控前量（r/k/v/w 各一份，4×）"),
            ("stk_gd", DH, "tanh 后 gd"),
            ("stk_om", HD, "om（内层 exp 前）"),
            ("stk_f", NC * D, "逐 head decay 行"),
            ("stk_K", D, "k（stk_dK 紧随其后）"),
            ("stk_dK", D, "dk"),
            ("stk_v", D, "v"),
            ("stk_r", D, "r"),
            ("stk_S", NC * D, "S_t（逐层打掩码后）"),
            ("stk_oc", HD, "ocat = LN 前的 outs"),
            ("stk_gk", self.F, "relu2 前 gk"),
            ("stk_wv", HD, "FFN value_w 的输入 x3"),
            ("stk_out", (NC + 1) * D, "块输出"),
        ]

    def stack_rows(self, nm):
        if nm == "stk_S":
            return self.NL * self.T
        if nm == "stk_rp":
            return 4 * self.T
        return self.T

    def stack_bytes(self):
        return sum(self.stack_rows(nm) * w * 4 for nm, w, _c in self.stacks())

    def param_shapes(self):
        HD, D, NL = self.HD, self.D, self.NL
        out = [("E", self.V, HD), ("ln0_g", 1, HD), ("ln0_b", 1, HD),
               ("lnout_g", 1, HD), ("lnout_b", 1, HD)]
        for l in range(NL):
            p = self.wp(l)
            out += [(p + "ln1_g", 1, HD), (p + "ln1_b", 1, HD),
                    (p + "ln2_g", 1, HD), (p + "ln2_b", 1, HD)]
            for nm in ("xr", "xk", "xv", "xw"):
                out.append((p + "tm_" + nm, 1, HD))
            out.append((p + "tm_w0row", 1, HD))
            out.append((p + "tm_w1p", HD, self.DH))
            out.append((p + "tm_w2p", self.DH, HD))
            for nm in ("Wr", "Wk", "Wv", "Wo"):
                out.append((p + "tm_" + nm, HD, HD))
            out.append((p + "tmln_g", 1, HD))
            out.append((p + "tmln_b", 1, HD))
            out.append((p + "cm_Wkey", self.F, HD))
            out.append((p + "cm_Wval", HD, self.F))
            out.append((p + "cmln_g", 1, HD))
        return out

    def param_names(self):
        return [n for n, _r, _c in self.param_shapes()]

    def param_count(self):
        return int(sum(r * c for _n, r, c in self.param_shapes()))

    def const_bytes(self):
        return int(sum(np.asarray(a).size for a in self.W.values()) * 4)

    # ---------------- 发射 ----------------
    def e(self, s=""):
        self.lines.append(s)

    def sh(self, name, r, c):
        if name in self._shaped:
            return
        self._shaped.add(name)
        self.e(f"    SHAPE       {name}, {r}, {c}")

    # ==================================================================
    def build(self):
        self._declare()
        self._prologue()
        for t in range(self.T):
            self._fwd_step(t)
        self.n_forward_lines = len(self.lines)
        self._backward()
        self._opt_epilogue()

    # ---------------- 声明 ----------------
    def _declare(self):
        HD, D, NC, NL, T = self.HD, self.D, self.NC, self.NL, self.T
        V, F, DH = self.V, self.F, self.DH
        e = self.e
        e("; ==========================================================================")
        e("; 由 tools/gen_eserNN_train.py **自动生成**（P9-J），请勿手改。")
        e(f"; 配置: vocab={V} hidden={HD} n_heads={NC} head_dim={D} inter={F}")
        e(f";       decay_hidden={DH} layers={NL} eps={self.EPS:g} T={T}")
        e("; 源: <ESERNN 仓库路径>/core/ese_rnn.py")
        e(";     ESERNN.forward_train / ESERNNBlock.forward_train")
        e(";     TimeMixing.forward_train / ChannelMixing.forward_train")
        e("; ==========================================================================")
        e()
        e(".section state")
        e("    ; 训练步自身完整：kk 与递归态在每条指令流开头重置")
        e()
        e(".section forward")
        e("")
        e("    ; ---------- LN gamma/dgamma/dbeta 三连（必须最先、严格相邻）----------")
        e("    ; 内核按 gamma+1=dg / gamma+2=db 的**槽号相邻**寻址；")
        e("    ; 顺序必须 gamma, dgamma, dbeta，且不被任何其它槽插入。")
        lnp = [("ln0_g", "g_ln0_g", "g_ln0_b", HD), ("lnout_g", "g_lnout_g", "g_lnout_b", HD)]
        for l in range(NL):
            p = self.wp(l)
            lnp += [(p + "ln1_g", "g_" + p + "ln1_g", "g_" + p + "ln1_b", HD),
                    (p + "tmln_g", "g_" + p + "tmln_g", "g_" + p + "tmln_b", HD),
                    (p + "ln2_g", "g_" + p + "ln2_g", "g_" + p + "ln2_b", HD)]
        for g, dg, db, wd in lnp:
            self.sh(g, 1, wd); self.sh(dg, 1, wd); self.sh(db, 1, wd)
        e("")
        e("    ; ---------- BPTT 栈 ----------")
        for nm, w, _c in self.stacks():
            self.sh(nm, self.stack_rows(nm), w)
        e("    ; stk_K / stk_dK 必须紧邻且 K 在前（OUTER_BWD 的 dk = k+1）")
        e("    ; stk_v / stk_dv_scratch 同理（v 的梯度走 OUTER_BWD 的 dv 覆盖输出）")
        e("")
        e("    ; ---------- 计数索引与工作槽 ----------")
        self.sh("kk", 1, 1)
        self.sh("kkbuf", 1, 1)
        self.sh("prev", 1, HD)
        self.sh("xcur", 1, HD)
        self.sh("d1", 1, HD)
        self.sh("frow", 1, HD)
        self.sh("dtmp", 1, HD)
        self.sh("ccc", 1, HD)
        self.sh("Sflat", D, NC * D)
        self.sh("Sfull", D, NC * D)
        for h in range(1, NC):
            self.sh(f"cc{h}", D, (h + 1) * D)
        for nm in ("dxe", "du", "d1p", "d2", "d3", "d4", "dkin", "dr", "dmul",
                   "dt1", "dt2", "dt3", "dt4", "dt5", "dt6"):
            self.sh(nm, 1, HD)
        self.sh("dpre", 1, F); self.sh("dgk", 1, F)
        self.sh("dom", 1, HD); self.sh("gd", 1, DH)
        self.sh("dw2", DH, HD); self.sh("ddh", DH, HD)
        for nm in ("dS", "dSn", "dSw", "dSf"):
            self.sh(nm, D, NC * D)
        self.sh("Z", D, D); self.sh("P", D, D); self.sh("PB", D, D)
        self.sh("dc", D, 1); self.sh("dr1", 1, D)
        for nm in ("dv", "dR", "dKin", "dGate", "dW2", "dI"):
            self.sh(nm, 1, NC * D)
        for t in range(T):
            self.sh(f"logits{t}", 1, V)
            self.sh(f"dtgt{t}", 1, V)
        e("")
        e("    ; ---------- 逐层激活槽 ----------")
        for l in range(NL):
            for nm, r, c in self._layer_shapes(self.LP(l)):
                self.sh(nm, r, c)

    def _layer_shapes(self, p):
        HD, D, NC, F = self.HD, self.D, self.NC, self.F
        out = [(p + nm, 1, HD) for nm in
               ("x", "xprev", "x1", "xx", "g1", "g2", "g3", "g4", "om", "dec",
                "ocat", "obs", "tmo", "x2", "x3", "gg", "gv", "out",
                "r", "kk", "vv")]
        out.append((p + "gd", 1, self.DH))
        for h in range(NC):
            out += [(p + f"f{h}", 1, D), (p + f"e{h}", 1, 1),
                    (p + f"r{h}", 1, D), (p + f"k{h}", 1, D),
                    (p + f"v{h}", 1, D), (p + f"oh{h}", 1, D),
                    (p + f"c{h}", D, 1), (p + f"z{h}", D, D)]
        for h in range(1, NC):
            out.append((p + f"S{h}", D, D))
        for h in range(2, NC):
            nm = f"hc{h}" if h < NC - 1 else "hdec"
            out.append((p + nm, 1, (h + 1) * D))
        for h in range(1, NC - 1):
            out.append((p + f"t{h}", 1, (h + 1) * D))
        out.append((p + "gk", 1, F))
        return out

    # ---------------- 前缀 ----------------
    def _prologue(self):
        e = self.e
        e("")
        e("    LN_EPS      eps1e6")
        e("")
        e("    ; ---------- 所有参数梯度清零（反向算子一律累加）----------")
        for g in self.param_names():
            e(f"    ZERO        g_{g}")
        e("")
        for nm, _w, _c in self.stacks():
            e(f"    STACK_INIT  {nm}, {self.stack_rows(nm)}")
        e("")
        e("    ; ---------- kk：1-based 的步内计数索引 ----------")
        e("    ; 前向每 PUSH 一次 +1、反向每 POP 一次 -1；反向由后往前填。")
        e("    ; 增减只用引擎算子（ADAMW 对 1 元素常量做 p -= lr*sign(g)）。")
        e("    ZERO        kk")
        e("    ZERO        prev")

    def _incr(self):
        """推进步内计数索引 kk。

        ⚠ 引擎与汇编器都**没有** `ADD1`/`SUB1`/`INCR`/`DECR` 这类计数伪指令
        （已核对 `asm/src/main.rs::opcode` 与 `core/src/format.rs`，均无）。
        唯一可用的「+1」手段是 `ADAMW` 作用在 1 元素常量上：
        它对 1 元素槽做 `p = p - lr·wd·p - lr·m̂/(√v̂+eps)`，
        当 `g = +1, lr = 1, wd = 0, eps → 0` 时结果恰为 `p - 1`。

        本函数在**前向**里 `kk -= 1`（把 1-based 序号往回走），
        反向里再 `kk += 1`，等价于「正向推进、反向回退」。
        """
        e = self.e
        e("    ; kk -= 1  （ADAMW 对 1 元素常量的受控步进；无 ADD1 伪指令）")
        e("    CLIP_GRAD   g_kkm, clip_norm")
        e("    ADAMW       kk, g_kkm, hp_dec")

    def _decr(self):
        e = self.e
        e("    ; kk += 1")
        e("    CLIP_GRAD   g_kkm, clip_norm")
        e("    ADAMW       kk, gm1, hp_dec")

    # ---------------- 前向 ----------------
    def _fwd_step(self, t):
        HD, NL = self.HD, self.NL
        e = self.e
        e("")
        e(f"    ; ================= 前向 step {t} =================")
        e(f"    COPY        xcur, x_t{t}")
        e(f"    PUSH        stk_xe, xcur, kk")
        e(f"    PUSH        stk_em, y{t}, kk")
        self._incr()
        e(f"    LN          u, xcur, ln0_g, ln0_b")
        e(f"    PUSH        stk_u, u, kk")
        self._incr()
        e(f"    ZERO        prev")
        cur = "u"
        for l in range(NL):
            cur = self._fwd_layer(l, cur, t)
        e(f"    ; ---- 输出头 ----")
        e(f"    COPY        xcur, {cur}")
        e(f"    LN          xn, xcur, lnout_g, lnout_b")
        e(f"    GEMM        logits{t}, xn, E")

    def _fwd_layer(self, l, xin, t):
        HD, D, NC, F, T = self.HD, self.D, self.NC, self.F, self.T
        p, w = self.LP(l), self.wp(l)
        e = self.e
        e(f"    ; ---------------- 层 {l} ----------------")
        e(f"    COPY        {p}x, {xin}")
        e(f"    PUSH        stk_xin, {p}x, kk")
        if t == 0:
            e(f"    COPY        {p}xprev, prev")
            e(f"    PUSH        stk_p0, prev, kk")
            e(f"    COPY        {p}xinm, prev")
        else:
            e(f"    COPY        {p}xprev, x_t{t-1}")
            e(f"    COPY        {p}xinm, x_t{t-1}")
        e(f"    PUSH        stk_xinm, {p}xinm, kk")
        self._incr()
        e(f"    LN          {p}x1, {p}x, {w}ln1_g, {w}ln1_b")
        e(f"    PUSH        stk_lg, {p}x1, kk")
        self._incr()
        e(f"    EWSUB       {p}xx, {p}xprev, {p}x1")
        e(f"    ; ---- 门控 r/k/v/w ----")
        for nm, Wnm, proj, wn in (("g1", "xr", "r", "Wr"), ("g2", "xk", "kk", "Wk"),
                                  ("g3", "xv", "vv", "Wv"), ("g4", "xw", None, None)):
            e(f"    EWMUL       dtmp, {p}xx, {w}tm_{Wnm}")
            e(f"    EWADD       {p}{nm}, {p}x1, dtmp")
            e(f"    PUSH        stk_rp, {p}{nm}, kk")
            self._incr()
            if proj is None:
                e(f"    GEMM        {p}gd, {p}{nm}, {w}tm_w1pT")
                e(f"    ACT         {p}gd, {p}gd, tanh")
                e(f"    PUSH        stk_gd, {p}gd, kk")
                self._incr()
                e(f"    GEMM        {p}om, {p}gd, {w}tm_w2pT")
                e(f"    EWADD       {p}om, {p}om, {w}tm_w0row")
                e(f"    PUSH        stk_om, {p}om, kk")
                self._incr()
                e(f"    EXP         {p}dec, {p}om")
                e(f"    NEG         {p}dec, {p}dec")
                e(f"    EXP         {p}dec, {p}dec")
            else:
                e(f"    GEMM        {p}{proj}, {p}{nm}, {w}tm_{wn}")
                if proj == "r":
                    e(f"    ACT         {p}r, {p}r, sigmoid")
                    e(f"    PUSH        stk_r, {p}r, kk")
                    self._incr()
                elif proj == "kk":
                    e(f"    PUSH        stk_K, {p}kk, kk")
                    e(f"    PUSH        stk_dK, {p}kk, kk")
                    self._incr()
                else:
                    e(f"    PUSH        stk_v, {p}vv, kk")
                    self._incr()
        e(f"    ; ---- 逐 head 复制 decay（one_r{D} = [D,1] 全 1）----")
        for h in range(NC):
            e(f"    SLICE       {p}e{h}, {p}dec, {h*D}, 1")
            e(f"    GEMM        {p}f{h}, {p}e{h}, one_r{D}")
        prevnm = f"{p}f0"
        for h in range(1, NC):
            nm = f"{p}hc{h}" if h < NC - 1 else f"{p}hdec"
            e(f"    CONCAT      {nm}, {prevnm}, {p}f{h}")
            prevnm = nm
        e(f"    COPY        ccc, {prevnm}")
        e(f"    PUSH        stk_f, ccc, kk")
        self._incr()
        e(f"    ; ---- S_t = S_(t-1)⊙decay_row + v⊗k ----")
        Sf = f"{p}Sflat"
        e(f"    ZERO        {Sf}")
        for h in range(NC):
            e(f"    SLICE       {p}r{h}, {p}r, {h*D}, {D}")
            e(f"    SLICE       {p}k{h}, {p}kk, {h*D}, {D}")
            e(f"    SLICE       {p}v{h}, {p}vv, {h*D}, {D}")
            e(f"    ZERO        {p}z{h}")
            e(f"    OUTER       {p}z{h}, {p}v{h}, {p}k{h}")
            e(f"    WRITECOL    {Sf}, {p}z{h}, {h*D}")
        e(f"    COPY        Sflat, {Sf}")
        e(f"    PUSH        stk_S, Sflat, kk")
        self._incr()
        e(f"    ; ---- outs_t = S_t @ r  （NT: C[D,NC*D]=S·r^T）----")
        e(f"    GEMM        {p}ocat, Sflat, {p}r")
        e(f"    LN          {p}obs, {p}ocat, {w}tmln_g, {w}tmln_b")
        e(f"    GEMM        {p}tmo, {p}obs, {w}tm_Wo")
        e(f"    EWADD       {p}x2, {p}x, {p}tmo")
        e(f"    PUSH        stk_2, {p}x2, kk")
        self._incr()
        e(f"    ; ---- ChannelMixing（复用同一个 x_prev）----")
        e(f"    LN          {p}x3, {p}x2, {w}ln2_g, {w}ln2_b")
        e(f"    PUSH        stk_wv, {p}x3, kk")
        self._incr()
        e(f"    COPY        dtmp, {p}xprev")
        e(f"    EWSUB       {p}gg, dtmp, {p}x3")
        e(f"    COPY        dt2, {p}gg")
        e(f"    EWMUL       {p}gg, {p}gg, {w}cm_xk")
        e(f"    EWADD       {p}gg, {p}x3, {p}gg")
        e(f"    GEMM        {p}gk, {p}gg, {w}cm_Wkey")
        e(f"    ACT         {p}gk, {p}gk, relu2")
        e(f"    PUSH        stk_gk, {p}gk, kk")
        self._incr()
        e(f"    GEMM        {p}gv, {p}gk, {w}cm_Wval")
        e(f"    EWMUL       {p}gv, {p}gv, {w}cmln_g")
        e(f"    EWADD       {p}out, {p}x2, {p}gv")
        return f"{p}out"

    # ---------------- 反向 ----------------
    def _backward(self):
        e = self.e
        e("")
        e("    ; ==================================================================")
        e("    ;  反向：step T-1 → 0")
        e("    ;  dh{l} 与 dS{l} 是跨 step 累加器；进 step t 时已含 t+1 的贡献。")
        e("    ; ==================================================================")
        for l in range(self.NL):
            e(f"    ZERO        dh{l}" if self.NL > 1 else "    ZERO        dh0")
        e(f"    ZERO        dS_all")
        for t in reversed(range(self.T)):
            self._bwd_step(t)

    def _bwd_step(self, t):
        pass

    def _opt_epilogue(self):
        pass


def main():
    ap = argparse.ArgumentParser(description="ESERNN 训练步 .vml 生成器（P9-J）")
    ap.add_argument("--cfg", choices=sorted(PRESETS), default=None)
    for nm in ("vocab", "hidden", "heads", "layers", "T"):
        ap.add_argument(f"--{nm}", type=int, default=None)
    ap.add_argument("--head-dim", type=int, default=None)
    ap.add_argument("--inter", type=int, default=None)
    ap.add_argument("--decay-hidden", type=int, default=None)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-dump", action="store_true")
    a = ap.parse_args()
    cfg = dict(PRESETS[a.cfg or "mini"])
    for k in ("vocab", "hidden", "heads", "head_dim", "inter", "decay_hidden",
              "layers", "T", "eps"):
        v = getattr(a, k)
        if v is not None:
            cfg[k] = v
    if a.hidden is not None and a.heads is None and a.head_dim is None:
        cfg["heads"] = 8
        cfg["head_dim"] = cfg["hidden"] // 8
        if a.inter is None:
            cfg["inter"] = cfg["hidden"] * 3
        if a.decay_hidden is None:
            cfg["decay_hidden"] = 32
    g = TrainGen(cfg, seed=a.seed, dump=not a.no_dump, tag=a.tag)
    g.build_weights()
    t0 = time.time()
    g.build()
    dt = time.time() - t0
    print(f"配置       = {cfg}")
    print(f"生成耗时   = {dt:.2f} s")
    print(f"总行数     = {len(g.lines)}")
    print(f"参数量     = {g.param_count()}  (常量池 ≈ {g.param_count()*4/1048576:.2f} MB)")
    print(f"栈槽字节   = {g.stack_bytes()/1048576:.2f} MB")
    print(f"常量张量   = {len(g.W)} 个, {g.const_bytes()/1048576:.2f} MB")


if __name__ == "__main__":
    main()
