#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""成员 C2：ESERNN `.vml` 生成器（参数化）。

## 为什么要有这个脚本

P7-C 手写的 `step.vml` 只覆盖 mini 配置（hidden=64 / 1 层），而真实 ESERNN 是
`hidden=512 / 6 层 / n_heads=8 / head_dim=64 / intermediate=1536`。
手写放大的**唯一硬阻塞**是：`.vml` 里 `SLICE` 的列偏移/宽度、`SHAPE` 的常量、
逐 head / 逐层的槽位名全是**硬编码数字**（见 P7-C README §9.4）。

本脚本把这些全部参数化，对 `H / n_heads / head_dim / n_layers / T / vocab /
intermediate / decay_hidden` 做循环展开，输出：

  * `step.vml`（或 `--out` 指定的文件名）—— 指令流
  * `weights.args`     —— 汇编器权重注入参数（相对路径，必须先 cd 到输出目录）
  * `ref_logits.txt` / `ref_S.bin` / `ref_xprev.bin` / `ref_x_steps.bin` 等参考
  * 所有 `*.bin` 权重张量（原始 f32 小端）

## 与 P7 手写版的等价性

`--cfg mini` 时，本脚本输出的指令流与 P7-C 手写的 `step.vml` **逐条同构**
（同样的指令序列、同样的槽位名、同样的数值），只是由循环展开生成。
`--check-vs-step` 会把两者做**指令级 diff**，确保回归成立。

## 关键约定（照抄 P7-C 的实测结论，不重新发明）

1. **默认 NT 的 GEMM**：`C[M,N] = A[M,K] · B[N,K]^T`，B 按 `nn.Linear` 的
   `[out,in]` 存放。`nn` 标志**实测不可用**（README §6），本脚本一律不用；
   凡需要 `A·B`（B 为 `[K,N]`）先 `TRANSPOSE`。
2. **`SLICE dst, src, startCol, width`**：4 操作数，宽度决定 dst 的列数。
3. **`LN dst, src, gamma, bias`**：4 操作数；输出必须与 src 同形状。
4. **`LN_EPS eps_slot`** 必须在最前面发一条，把 eps 从引擎默认 1e-5 改成
   真实 ESERNN 的 `cfg.rms_eps = 1e-6`。不发的话 mini 下差 2.8e-2。
5. **输入槽必须显式 `SHAPE`**（`x` / `logits`），否则形状推断链断在 0x0，
   越界写坏内存。
6. **`M=1` 的 GEMM 链条易断**，所有中间槽显式声明形状最稳（本脚本自动补）。

## 参数化要点

* **逐 head 循环**：`st_S` 是 `[D, n_heads*D]`（head 水平拼接），
  第 h 个 head 占列 `h*D .. h*D+D`。所有 `SLICE`/`WRITECOL` 的偏移都是 `h*D`，
  **宽度是 `D`** —— 这就是必须脚本生成的那部分。
* **逐层循环**：每层一套槽位名（`L{l}_xxx`）与一套权重（`l{l}_xxx`），
  `st_S{l}` / `st_xprev{l}` 是**每层独立的持久状态**。
  层 l 的输入是层 l-1 的输出；`x_prev` 每层独立（源码 `states[i][0]`）。
* **logits 只在最后一层之后算一次**（源码 `F.linear(self.ln_out(x), E)` 在层循环外）。

## 用法

```powershell
python tools/gen_eserNN.py --cfg mini   --out-dir examples/eserNN/gen_mini
python tools/gen_eserNN.py --hidden 128 --layers 2 --out-dir examples/eserNN/gen_h128_l2
python tools/gen_eserNN.py --hidden 512 --layers 6 --vocab 3379 `
    --inter 1536 --decay-hidden 32 --out-dir examples/eserNN/gen_h512_l6
```
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# ======================================================================
# 配置
# ======================================================================
#: mini 配置 —— 与 P7-C `make_eserNN.py` 的 CFG **逐字段一致**，
#: 这样生成器的回归才能和手写 `step.vml` 对齐。
MINI = dict(
    vocab=32, hidden=64, heads=8, head_dim=8,
    inter=128, decay_hidden=16, eps=1e-6, T=6, layers=1,
)

#: 真实 ESERNN 配置（`ESERNNConfig.base_24m`：hidden=512 / 6 层 / 8 头 / D=64）。
#: vocab 用 3379（P7-C README §9 的口径；真实源码默认 16384，仅影响 E 的形状）。
REAL = dict(
    vocab=3379, hidden=512, heads=8, head_dim=64,
    inter=1536, decay_hidden=32, eps=1e-6, T=6, layers=6,
)

EPS_ENGINE = 1e-5          # 引擎 OP_LN 的默认 eps（对照用）
SEED = 20240927


class Gen:
    """按配置参数化展开 ESERNN 单步前向。"""

    def __init__(self, cfg, ckpt=None, seed=SEED):
        self.c = dict(cfg)
        c = self.c
        self.HD = c["hidden"]
        self.NC = c["heads"]
        self.D = c["head_dim"]
        self.F = c["inter"]
        self.DH = c["decay_hidden"]
        self.V = c["vocab"]
        self.T = c["T"]
        self.NL = c["layers"]
        self.EPS = c["eps"]
        assert self.NC * self.D == self.HD, \
            f"真实 ESERNN 要求 hidden == n_heads*head_dim，收到 {self.HD} vs {self.NC}*{self.D}"
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.W: dict[str, np.ndarray] = {}
        self.lines: list[str] = []
        self.ckpt = ckpt

    # ---------------- 权重 ----------------
    def wp(self, l, name):
        """权重张量名：单层不带层前缀（与 P7-C 手写 `step.vml` 对齐）。"""
        return (f"l{l}_" if self.NL > 1 else "") + name

    def sp(self, l):
        """状态槽名：单层不带层后缀。"""
        return f"st_S{l}" if self.NL > 1 else "st_S"

    def xpp(self, l):
        """x_prev 状态槽名：单层不带层后缀。"""
        return f"st_xprev{l}" if self.NL > 1 else "st_xprev"

    def _rnd(self, *shape, scale=1.0):
        return (self.rng.standard_normal(shape) * scale).astype(np.float32)

    def _orth(self, rows, cols, gain):
        """等价 torch.nn.init.orthogonal_(W) * gain。

        torch 对 [rows, cols]：若 rows < cols 先转置 → QR 取 Q → 再转置回来。
        这里显式复刻，否则 [H,F] (H<F) 的 value_w 会拿到不同矩阵。
        """
        if rows < cols:
            a = self.rng.standard_normal((cols, rows))
            q, r = np.linalg.qr(a)
            q = q * np.sign(np.diag(r))
            return np.ascontiguousarray((q * gain).T.astype(np.float32))
        a = self.rng.standard_normal((rows, cols))
        q, r = np.linalg.qr(a)
        q = q * np.sign(np.diag(r))
        return (q * gain).astype(np.float32)

    def build_weights(self):
        """按真实 `ese_rnn.py` 的初始化造权重（或从 checkpoint 加载）。"""
        HD, D, NC = self.HD, self.D, self.NC
        F, DH, V = self.F, self.DH, self.V
        W = self.W
        ck = self.ckpt

        if ck is None:
            W["E"] = self._rnd(V, HD, scale=0.02)
            # --- ESERNN.ln0 / ln_out ---
            W["ln0_g"] = np.ones((1, HD), np.float32)
            W["ln0_b"] = np.zeros((1, HD), np.float32)
            W["lnout_g"] = np.ones((1, HD), np.float32)
            W["lnout_b"] = np.zeros((1, HD), np.float32)
            # --- 每层 ---
            for l in range(self.NL):
                p = self.wp(l, "")
                W[p + "ln1_g"] = np.ones((1, HD), np.float32)
                W[p + "ln1_b"] = np.zeros((1, HD), np.float32)
                W[p + "ln2_g"] = np.ones((1, HD), np.float32)
                W[p + "ln2_b"] = np.zeros((1, HD), np.float32)
                # TimeMixing 门控：真实是 zeros
                for nm in ("xr", "xk", "xv", "xw"):
                    W[p + "tm_" + nm] = np.zeros((1, HD), np.float32)
                # w0 默认 decay_init_w0 = -2.5 → decay ≈ 0.921
                W[p + "tm_w0col"] = np.full((HD, 1), -2.5, np.float32)
                # ⚠ w1/w2 是 nn.Parameter，布局与 nn.Linear **相反**：
                #   w1p=[H,DH] / w2p=[DH,H]。VMLM 的默认 NT 要转置副本。
                W[p + "tm_w1p"] = self._rnd(HD, DH, scale=0.02)
                W[p + "tm_w1pT"] = np.ascontiguousarray(W[p + "tm_w1p"].T)
                W[p + "tm_w2p"] = np.zeros((DH, HD), np.float32)
                W[p + "tm_w2pT"] = np.ascontiguousarray(W[p + "tm_w2p"].T)
                W[p + "tm_Wr"] = self._orth(HD, HD, 0.1)
                W[p + "tm_Wk"] = self._orth(HD, HD, 0.1)
                W[p + "tm_Wv"] = self._orth(HD, HD, 0.1)
                W[p + "tm_Wo"] = self._orth(HD, HD, 0.1)
                W[p + "tmln_g"] = np.ones((1, HD), np.float32)
                W[p + "tmln_b"] = np.zeros((1, HD), np.float32)
                W[p + "cm_xk"] = np.zeros((1, HD), np.float32)
                W[p + "cm_Wkey"] = self._orth(F, HD, 0.05)
                W[p + "cm_Wval"] = self._orth(HD, F, 0.05)
                W[p + "cmln_g"] = np.ones((1, HD), np.float32)
        else:
            self._load_from_ckpt(ck)

        # --- 输入序列 ---
        # ⚠ token 序列必须**只依赖 (vocab, T, seed)**，不能受 n_layers 影响：
        # 否则不同层数的档位会拿到不同输入，跨档对比就失去意义（也让我一度
        # 误判成层间 bug）。这里用独立的 rng，不消耗 `self.rng` 的权重序列。
        trng = np.random.default_rng(self.seed + 99991)
        self.TOKENS = trng.integers(0, V, size=self.T).astype(np.int32)
        self.X = W["E"][self.TOKENS]                      # [T, HD]
        W["tok"] = self.TOKENS.astype(np.float32)
        for t in range(self.T):
            W[f"x_t{t}"] = self.X[t:t + 1].copy()
        W["x_all"] = self.X.copy()

        # --- 汇编辅助常量 ---
        W["one_r1"] = np.ones((1, 1), np.float32)
        W["one_col"] = np.ones((1, 1), np.float32)   # GEMM 广播源 [1,1]
        W["eps1e6"] = np.array([[self.EPS]], np.float32)
        # one_r{D}: [D,1] 全 1 —— GEMM(标量[1,1], one_rD) ⇒ [1,D] 把标量复制 D 份。
        # 名字带 head_dim 后缀，便于在多个档位之间对比 `.vml` 时一眼看出宽度。
        W[f"one_r{D}"] = np.ones((self.D, 1), np.float32)

    def _load_from_ckpt(self, sd):
        """从真实 checkpoint 的 state_dict 加载（`nn.Parameter` 转置副本在此产生）。"""
        import torch  # noqa: F401  (只是确认可用)
        HD, D, NC = self.HD, self.D, self.NC
        W = self.W

        def g(name):
            return np.ascontiguousarray(sd[name].detach().cpu().numpy().astype(np.float32))

        emb = g("embed_tokens.weight") if "embed_tokens.weight" in sd else None
        if emb is None:
            raise KeyError("checkpoint 里没有 embed_tokens.weight")
        W["E"] = emb
        for src, dst in (("ln0.weight", "ln0_g"), ("ln0.bias", "ln0_b"),
                         ("ln_out.weight", "lnout_g"), ("ln_out.bias", "lnout_b")):
            W[dst] = g(src).reshape(1, -1)
        for l in range(self.NL):
            p, s = self.wp(l, ""), f"layers.{l}."
            for nm, d2 in (("ln1.weight", "ln1_g"), ("ln1.bias", "ln1_b"),
                           ("ln2.weight", "ln2_g"), ("ln2.bias", "ln2_b")):
                W[p + d2] = g(s + nm).reshape(1, -1)
            tm = s + "time_mixing."
            # ⚠ checkpoint 里叫 x_r/x_k/x_v/x_w，我的 W 字典里也叫 tm_xr/tm_xk/…
            #   （去掉了下划线）。用 nm[2:] 会得到 "r" 而不是 "x_r" —— 一度
            #   让我在 crosscheck 里拿到 KeyError，这里显式配对最稳。
            for ck_name, short in (("x_r", "xr"), ("x_k", "xk"),
                                   ("x_v", "xv"), ("x_w", "xw")):
                W[p + "tm_" + short] = g(tm + ck_name).reshape(1, -1)
            W[p + "tm_w0col"] = g(tm + "w0").reshape(-1, 1)
            # w1/w2 是 nn.Parameter：权威布局 [H,DH] / [DH,H]，VMLM 要转置副本
            W[p + "tm_w1p"] = g(tm + "w1")
            W[p + "tm_w1pT"] = np.ascontiguousarray(W[p + "tm_w1p"].T)
            W[p + "tm_w2p"] = g(tm + "w2")
            W[p + "tm_w2pT"] = np.ascontiguousarray(W[p + "tm_w2p"].T)
            for nm, d2 in (("r_w.weight", "tm_Wr"), ("k_w.weight", "tm_Wk"),
                           ("v_w.weight", "tm_Wv"), ("o_w.weight", "tm_Wo")):
                W[p + d2] = g(tm + nm)
            W[p + "tmln_g"] = g(tm + "ln.weight").reshape(1, -1)
            W[p + "tmln_b"] = g(tm + "ln.bias").reshape(1, -1)
            cm = s + "channel_mixing."
            W[p + "cm_xk"] = g(cm + "x_k").reshape(1, -1)
            W[p + "cm_Wkey"] = g(cm + "key_w.weight")
            W[p + "cm_Wval"] = g(cm + "value_w.weight")
            lnk = cm + "ln"
            if lnk in sd:
                W[p + "cmln_g"] = g(lnk).reshape(1, -1)
            else:
                W[p + "cmln_g"] = np.ones((1, HD), np.float32)

    # ---------------- 指令流 ----------------
    def emit(self, s):
        self.lines.append(s)

    # ---------------- LN bias 槽号的历史坑（已由 Lead 在 core/asm 修复）----------------
    # `LN dst, src, gamma, bias` 是 4 操作数，放不进 dst/a/b 三个 u32，所以第 4 个
    # （**bias 槽号**）被编进 16 字节指令的 `dt` 字段。
    #
    # 【曾经的硬阻塞】P8 早期 `dt` 是 `u8`，汇编器写的是
    #     dt = self.slot(regs[3]) as u8;     // Rust 的 `as` **静默回绕**
    # 于是槽号 ≥ 256 的 bias 会指向一个完全无关的槽位，LayerNorm 用错 bias，
    # 数值**静默出错**。实测（3 层 hidden=256，376 槽）：
    #     l2_tmln_b 槽号 358 -> dt=102 = L0_oh1   ✗
    #     l2_ln2_b  槽号 364 -> dt=108 = L0_oh7   ✗
    #     lnout_b   槽号 375 -> dt=119 = L0_tmo   ✗
    # 对照：2 层（254 槽）0 条越界 → PASS 1.4e-6。
    # ⇒ **层数无关**，只跟「有没有 LN bias 槽号 ≥ 256」有关（截断指纹）。
    #
    # 【现状】Lead 已改成 `struct Ins { op:u8, fl:u8, dt:u16, dst:u32, a:u32, b:u32 }`
    # （仍 16 字节，无填充），槽位上限提到 65535，并给三处转换加了超界 panic。
    # ⇒ 本生成器**不需要**任何槽位顺序规避，用自然的分配即可。

    # ---------------- 显式形状声明 ----------------
    def all_shapes(self):
        """列出**每一个**激活槽的显式形状：[(槽名, rows, cols), ...]。

        【为什么必须全部显式声明】
        汇编器 `asm/src/main.rs::infer_shapes` 只迭代 **4 轮**（源码注释：
        「指令流是 DAG，几轮足够」）。但 ESERNN 的层内依赖链有 150+ 条指令，
        跨层更深 —— 4 轮**根本传播不到**第 2 层及以后的槽位。
        后果：那些槽位形状停留在 0x0 或错误值，GEMM/LN 用错的 m/n/k，
        数值静默出错（实测 hidden=256/3 层时 step0 就错 6.5e-2，
        而 2 层只错 1.4e-6 —— 正是「第 3 层刚好超出 4 轮传播能力」的特征）。
        显式 SHAPE 是 `force_shape`，直接 pin 住形状，与迭代轮数无关。
        """
        HD, D, NC = self.HD, self.D, self.NC
        F = self.F
        out = []
        # 全局
        out.append(("x", 1, HD))
        out.append(("logits", 1, self.V))
        out.append(("u", 1, HD))
        out.append(("xn", 1, HD))
        for l in range(self.NL):
            p = f"L{l}_" if self.NL > 1 else ""
            # --- [1, HD] 的向量 ---
            # ⚠ 不要在这里放 hdec：它的宽度是 NC*D == HD，但下面 hc 循环里
            #   同名条目会再声明一次，顺序不同会导致自相矛盾（我踩过：
            #   多声明一次 hdec 让 3 层误差从 6.5e-2 涨到 7.5e-1）。
            for nm in ("x1", "xx", "g1", "r", "g2", "kk", "g3", "vv",
                       "g4", "om", "w0row", "dec", "ocat", "obs",
                       "tmo", "x2", "x3", "gg", "gv", "out"):
                out.append((p + nm, 1, HD))
            # --- decay 的中间量 [1, DH] / [1, 1] ---
            out.append((p + "gd", 1, self.DH))
            for h in range(NC):
                out.append((p + f"e{h}", 1, 1))
                out.append((p + f"f{h}", 1, D))
            # HCAT: hc{2..NC-1} 宽度 3D..(NC-1)D，最后一个 hdec 宽度 NC*D == HD
            for h in range(2, NC):
                nm = f"hc{h}" if h < NC - 1 else "hdec"
                out.append((p + nm, 1, (h + 1) * D))
            # --- 逐 head 的 [D,D] 与 [1,D] / [D,1] ---
            for h in range(NC):
                out.append((p + f"S{h}", D, D))
                out.append((p + f"r{h}", 1, D))
                out.append((p + f"k{h}", 1, D))
                out.append((p + f"v{h}", 1, D))
                out.append((p + f"o{h}c", D, 1))
                out.append((p + f"oh{h}", 1, D))
            # --- FFN ---
            out.append((p + "gk", 1, F))
            # --- 逐 head 拼接的中间量 t{1..NC-2} 与 ocat（宽度 NC*D == HD）---
            for h in range(1, NC - 1):
                out.append((p + f"t{h}", 1, (h + 1) * D))
        # 去重（后面的覆盖前面的），保持顺序
        seen = {}
        for nm, r, c in out:
            seen[nm] = (r, c)
        return [(nm, rc[0], rc[1]) for nm, rc in seen.items()]

    def build_vml(self):
        c = self.c
        HD, D, NC, V, NL = self.HD, self.D, self.NC, self.V, self.NL
        F, DH = self.F, self.DH
        E = self.emit
        E("; ==========================================================================")
        E("; 由 tools/gen_eserNN.py **自动生成**，请勿手改。")
        E(f"; 配置: vocab={V} hidden={HD} n_heads={NC} head_dim={D} "
          f"inter={F} decay_hidden={DH} layers={NL} eps={self.EPS:g} T={self.T}")
        E("; 源: <外部 ESERNN 源码目录>/core/ese_rnn.py")
        E(";     ESERNN.forward_one / forward_train / ESERNNBlock / TimeMixing / ChannelMixing")
        E(";")
        E("; 约定（P7-C 实测）:")
        E(";   GEMM 默认 NT: C[M,N] = A[M,K]·B[N,K]^T  （B 按 nn.Linear 的 [out,in]）")
        E(";   `nn` 标志实测不可用，一律不用；需要 A·B 时先 TRANSPOSE。")
        E(";   LayerNorm 沿 src 的**行**归一化；dst 必须与 src 同形状。")
        E(";   EWMUL/EWADD 广播: [M,N]op[M,1] 行广播；[M,N]op[1,N] 列广播。")
        E("; ==========================================================================")
        E("")
        E(".section state")
        # 每层一套持久状态：递归态 st_S{l} 与上一 step 的块输出 st_xprev{l}
        for l in range(NL):
            E(f"    ; ---- 第 {l} 层：递归态（{NC} 个 head 的 [{D},{D}] 水平拼接成 [{D},{NC*D}]）----")
            E(f"    SHAPE   {self.sp(l)},    {D}, {NC*D}")
            E(f"    SHAPE   {self.xpp(l)},    1, {HD}")
        E("")
        E(".section forward")
        E(f"    ; 输入槽必须显式声明形状（推不出来 → 0x0 → 越界写坏内存）")
        E(f"    SHAPE       x,           1, {HD}")
        E(f"    SHAPE       logits,      1, {V}")
        E("")
        E("    ; ---------- 把 LayerNorm 的 eps 设成真实 ESERNN 的 cfg.rms_eps ----------")
        E("    ; 引擎 OP_LN 默认 1e-5，真实 ESERNN 是 1e-6；不发这条 mini 下差 2.8e-2。")
        E("    LN_EPS      eps1e6")
        E("")
        E("    ; ---------- x_prev 说明（**不要**在这里锁存 x！）----------")
        E("    ; 源码 forward_one:")
        E("    ;     for i, blk in enumerate(self.layers):")
        E("    ;         prev_x, kv = state[i][0], state[i][1]")
        E("    ;         x, kv = blk(x, prev_x.unsqueeze(0), kv, H, D)")
        E("    ;         state[i][0].copy_(x[0,-1])")
        E("    ; ⇒ 每层的 x_prev 是「**上一 step 该层自己的块输出**」，")
        E("    ;   完全由持久状态提供，**不是**本 step 的输入 x。")
        E("    ;   各层末尾的 `COPY st_xprev{l}, L{l}_out` 负责推进它。")
        E("    ;")
        E("    ; ⚠ P7-C 手写的单层版在开头写了 `COPY st_xprev, x`，那在 mini 下")
        E("    ;   是**死代码**（mini 的 tm_xr/xk/xv/xw 与 cm_xk 全为 0，")
        E("    ;   于是 xx = x_prev - ln1(x) 对 r/k/v/decay/FFN 全部无影响）。")
        E("    ;   放大后权重非零，它立刻变成真 bug（实测差 6.5e-2）。")
        E("    ;   本生成器因此**不做任何开头的 x_prev 锁存**。")
        E("")
        E("    ; ---------- 入口: x = ln0(embed(token)) ----------")
        E("    LN      u, x, ln0_g, ln0_b")
        E("")

        # 每层输入：第 0 层是 u，之后是上一层的输出 x_out
        cur = "u"
        for l in range(NL):
            cur = self.emit_layer(l, cur)

        # ---------- 输出头（层循环外，只算一次）----------
        E("    ; ---------- 输出头: logits = ln_out(x) @ E^T（tied embedding）----------")
        E(f"    LN      xn, {cur}, lnout_g, lnout_b")
        E(f"    GEMM    logits, xn, E")
        E("")
        E("    HALT")

    def emit_layer(self, l, xin):
        """展开第 l 层。`xin` 是本层输入槽名；返回本层输出槽名。"""
        HD, D, NC = self.HD, self.D, self.NC
        F, DH = self.F, self.DH
        # 单层时槽位/权重名不带层前缀 —— 这样生成的 mini 指令流与 P7-C 手写的
        # `step.vml` **逐字一致**，回归才有意义（见 tools/_diff_vml.py）。
        p = f"L{l}_" if self.NL > 1 else ""
        w = f"l{l}_" if self.NL > 1 else ""
        ss, xp = self.sp(l), self.xpp(l)
        E = self.emit
        E(f"    ; ================= 第 {l} 层（ESERNNBlock）=================")
        E(f"    ; 本层独立的 x_prev（源码 states[{l}][0]），跨 step 由块输出推进")
        E(f"    LN      {p}x1, {xin}, {w}ln1_g, {w}ln1_b")
        E(f"    EWSUB   {p}xx, {xp}, {p}x1")
        E("")
        E(f"    ; ---- TimeMixing: r / k / v ----")
        # 发射顺序与 P7-C 手写版**逐条一致**（数学上等价的顺序也要对齐，
        # 否则 tools/_diff_vml.py 的指令级回归会出现噪声差异）。
        for nm, Wnm, proj, wname in (("g1", "xr", "r", "Wr"),
                                     ("g2", "xk", "kk", "Wk"),
                                     ("g3", "xv", "vv", "Wv")):
            E(f"    EWMUL   {p}{nm}, {p}xx, {w}tm_{Wnm}")
            E(f"    EWADD   {p}{nm}, {p}x1, {p}{nm}")
            E(f"    GEMM    {p}{proj}, {p}{nm}, {w}tm_{wname}")
            if proj == "r":
                E(f"    ACT     {p}r, {p}r, sigmoid")
        E("")
        E(f"    ; ---- decay[h] = exp(-exp(w0[h] + tanh((x1 + xx⊙x_w)@w1)@w2)) ----")
        E(f"    EWMUL   {p}g4, {p}xx, {w}tm_xw")
        E(f"    EWADD   {p}g4, {p}x1, {p}g4")
        E(f"    ; w1/w2 是 nn.Parameter（布局 [H,DH]/[DH,H]，与 nn.Linear 相反）→ 用转置副本")
        E(f"    GEMM    {p}gd, {p}g4, {w}tm_w1pT")
        E(f"    ACT     {p}gd, {p}gd, tanh")
        E(f"    GEMM    {p}om, {p}gd, {w}tm_w2pT")
        E(f"    TRANSPOSE {p}w0row, {w}tm_w0col")
        E(f"    EWADD   {p}om, {p}om, {p}w0row")
        E(f"    EXP     {p}dec, {p}om")
        E(f"    NEG     {p}dec, {p}dec")
        E(f"    EXP     {p}dec, {p}dec")
        E("")
        E(f"    ; ---- per-head 标量 decay：取每 head 第 0 通道，复制 D 份后按列缩放 st_S ----")
        # 逐 head：SLICE 宽度 = 1（偏移 = h*D），这是**必须脚本生成**的那部分。
        for h in range(NC):
            E(f"    SLICE   {p}e{h}, {p}dec, {h*D}, 1")
        # GEMM 把标量 [1,1] 复制成 [1,D]（矩阵乘天然做广播复制）。
        # ⚠ M=1 的 GEMM 链条易断，所以 e*/f*/hc* 全部显式 SHAPE 声明。
        for h in range(NC):
            E(f"    SHAPE   {p}f{h},          1, {D}")
        for h in range(NC):
            E(f"    SHAPE   {p}e{h},          1, 1")
        # hc{h} 是拼接中间量。注意列数是 **(h+1)*D** 而不是 h*D：
        # CONCAT hc{h} = hc{h-1}(宽度 h*D) ++ f{h}(宽度 D)；hdec 是最后一个，宽度 NC*D。
        # 写成 h*D 会让 hc* 的声明宽度比实际窄 D，EWMUL 时直接报 "EW shape mismatch"。
        for h in range(2, NC):
            nm = f"{p}hc{h}" if h < NC - 1 else f"{p}hdec"
            E(f"    SHAPE   {nm},{' ' * max(1, 11 - len(nm))}1, {(h + 1) * D}")
        for h in range(NC):
            E(f"    GEMM    {p}f{h}, {p}e{h}, one_r{D}")
        # 逐 head CONCAT 拼回 [1, HD]
        prev = f"{p}f0"
        for h in range(1, NC):
            nm = f"{p}hc{h}" if h < NC - 1 else f"{p}hdec"
            E(f"    CONCAT  {nm}, {prev}, {p}f{h}")
            prev = nm
        E(f"    ; hdec[1,HD] 每 head 的 D 个通道都是该 head 的 decay 标量")
        E(f"    ; st_S{l}[D, NC*D] 按列缩放：列 j 的 head 是 j//D，命中「[M,N]op[1,N]」列广播")
        E(f"    EWMUL   {ss}, {ss}, {prev}")
        E("")
        E(f"    ; ---- 逐 head 外积写入: S[h] += v[h] ⊗ k[h] ----")
        E(f"    ; 切片 → 独立算 → WRITECOL 写回（OUTER 是原地累加，不能直接在 st_S 上做）")
        for h in range(NC):
            E(f"    SLICE   {p}S{h}, {ss}, {h*D}, {D}")
        for h in range(NC):
            E(f"    SLICE   {p}r{h}, {p}r, {h*D}, {D}")
            E(f"    SLICE   {p}k{h}, {p}kk, {h*D}, {D}")
            E(f"    SLICE   {p}v{h}, {p}vv, {h*D}, {D}")
        for h in range(NC):
            E(f"    OUTER   {p}S{h}, {p}v{h}, {p}k{h}")
        for h in range(NC):
            E(f"    WRITECOL {ss}, {p}S{h}, {h*D}")
        E("")
        E(f"    ; ---- o[h] = S[h] @ r[h] : 默认 NT 取 B=r_h[1,D] ⇒ C[D,1]=S_h·r_h^T ----")
        for h in range(NC):
            E(f"    GEMM    {p}o{h}c, {p}S{h}, {p}r{h}")
        for h in range(NC):
            E(f"    TRANSPOSE {p}oh{h}, {p}o{h}c")
        prev = f"{p}oh0"
        for h in range(1, NC):
            nm = f"{p}t{h}" if h < NC - 1 else f"{p}ocat"
            E(f"    CONCAT  {nm}, {prev}, {p}oh{h}")
            prev = nm
        E("")
        E(f"    ; ---- out = ln(outs); x2 = x + o_w(out) ----")
        E(f"    LN      {p}obs, {prev}, {w}tmln_g, {w}tmln_b")
        E(f"    GEMM    {p}tmo, {p}obs, {w}tm_Wo")
        E(f"    EWADD   {p}x2, {xin}, {p}tmo")
        E("")
        E(f"    ; ---- ChannelMixing(FFN) ----")
        E(f"    LN      {p}x3, {p}x2, {w}ln2_g, {w}ln2_b")
        E(f"    ; g = x3 + (x_prev - x3) ⊙ x_k   ← 用**未 LN 的块级 x_prev**")
        E(f"    EWSUB   {p}gg, {xp}, {p}x3")
        E(f"    EWMUL   {p}gg, {p}gg, {w}cm_xk")
        E(f"    EWADD   {p}gg, {p}x3, {p}gg")
        E(f"    GEMM    {p}gk, {p}gg, {w}cm_Wkey")
        E(f"    ACT     {p}gk, {p}gk, relu2")
        E(f"    GEMM    {p}gv, {p}gk, {w}cm_Wval")
        E(f"    ; ffn_norm_mode='scale'：逐通道 gamma（恒等初始）")
        E(f"    EWMUL   {p}gv, {p}gv, {w}cmln_g")
        E(f"    EWADD   {p}out, {p}x2, {p}gv")
        E("")
        E(f"    ; ---- 推进本层 x_prev ← 块输出（源码 states[{l}][0].copy_(x[0,-1])）----")
        E(f"    COPY    {xp}, {p}out")
        E("")
        return f"{p}out"

    # ---------------- numpy 参考 ----------------
    @staticmethod
    def _ln(x, g, b, eps):
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + eps) * g + b

    def reference(self, eps=None):
        """逐步复算 ESERNN.forward_one 的数学，返回参考量。

        多层语义严格照抄源码（`ese_rnn.py` 的 `forward_one`）：
            x = ln0(embed(token))
            for blk in layers:  x, kv = blk(x, x_prev[l], kv[l]);  x_prev[l] = x
            logits = ln_out(x) @ E^T
        """
        eps = self.EPS if eps is None else eps
        HD, D, NC, NL, V = self.HD, self.D, self.NC, self.NL, self.V
        W = self.W
        S = [np.zeros((NC, D, D), np.float32) for _ in range(NL)]
        xp = [np.zeros(HD, np.float32) for _ in range(NL)]
        lg_all, x_all_all, xstep_all = [], [], []

        for t in range(self.T):
            xt = self.X[t]
            x = self._ln(xt[None, :], W["ln0_g"], W["ln0_b"], eps)
            for l in range(NL):
                p = self.wp(l, "")
                xin = self._ln(x, W[p + "ln1_g"], W[p + "ln1_b"], eps)
                xprev = xp[l][None, :]
                xx = xprev - xin
                r = 1.0 / (1.0 + np.exp(-(xin + xx * W[p + "tm_xr"]) @ W[p + "tm_Wr"].T))
                k = (xin + xx * W[p + "tm_xk"]) @ W[p + "tm_Wk"].T
                v = (xin + xx * W[p + "tm_xv"]) @ W[p + "tm_Wv"].T
                wv = (W[p + "tm_w0col"].T + np.tanh(
                    (xin + xx * W[p + "tm_xw"]) @ W[p + "tm_w1p"]) @ W[p + "tm_w2p"])
                decay = np.exp(-np.exp(wv))
                r_h = r.reshape(NC, D); k_h = k.reshape(NC, D)
                v_h = v.reshape(NC, D); d_h = decay.reshape(NC, D)
                # state = state*decay + v⊗k  (decay 广播在 v 方向 → S[h] *= d_h[h,0])
                S[l] = S[l] * d_h[:, 0][:, None, None] + v_h[:, :, None] * k_h[:, None, :]
                outs = np.einsum("hij,hj->hi", S[l], r_h).reshape(1, NC * D)
                tm_out = self._ln(outs, W[p + "tmln_g"], W[p + "tmln_b"], eps) @ W[p + "tm_Wo"].T
                x = x + tm_out
                xin2 = self._ln(x, W[p + "ln2_g"], W[p + "ln2_b"], eps)
                gk = xin2 + (xprev - xin2) * W[p + "cm_xk"]
                gk = np.maximum(gk @ W[p + "cm_Wkey"].T, 0.0) ** 2
                x = x + (gk @ W[p + "cm_Wval"].T) * W[p + "cmln_g"]
                xp[l] = x[0].copy()
            xstep_all.append(x[0].copy())
            x_all_all.append(x[0].copy())
            lg_all.append((self._ln(x, W["lnout_g"], W["lnout_b"], eps) @ W["E"].T)[0].copy())
        return (np.array(lg_all, np.float32), np.array(S, np.float32),
                np.array(xp, np.float32), x[0].copy(), np.array(xstep_all, np.float32))

    # ---------------- 交叉验证：真实 ESERNN 类 ----------------
    def crosscheck(self, eser_dir=r"<外部 ESERNN 源码目录>"):
        # ⚠ 必须**先**把 ESERNN 目录插进 sys.path，再 import core.ese_rnn。
        if eser_dir not in sys.path:
            sys.path.insert(0, eser_dir)
        try:
            import torch
            from core.ese_rnn import ESERNN, ESERNNConfig
        except Exception as e:                                # noqa: BLE001
            return None, f"无 torch/ESERNN: {e}"
        (logits, S, xp, xf, xs) = self.reference()
        torch.set_num_threads(1)
        cfg = ESERNNConfig(vocab_size=self.V, hidden=self.HD, n_layers=self.NL,
                           n_heads=self.NC, head_dim=self.D, intermediate=self.F,
                           decay_hidden=self.DH, rms_eps=self.EPS, ffn_norm=True,
                           ffn_norm_mode="scale", ffn_act="relu2")
        m = ESERNN(cfg)
        sd = m.state_dict()
        put = lambda n, a: sd.__setitem__(n, torch.from_numpy(
            np.ascontiguousarray(a, np.float32)).reshape(sd[n].shape))
        put("embed_tokens.weight", self.W["E"])
        put("ln0.weight", self.W["ln0_g"].ravel()); put("ln0.bias", self.W["ln0_b"].ravel())
        put("ln_out.weight", self.W["lnout_g"].ravel())
        put("ln_out.bias", self.W["lnout_b"].ravel())
        for l in range(self.NL):
            p, s = self.wp(l, ""), f"layers.{l}."
            put(s + "ln1.weight", self.W[p + "ln1_g"].ravel())
            put(s + "ln1.bias", self.W[p + "ln1_b"].ravel())
            put(s + "ln2.weight", self.W[p + "ln2_g"].ravel())
            put(s + "ln2.bias", self.W[p + "ln2_b"].ravel())
            q = s + "time_mixing."
            for short, full in (("x_r", "xr"), ("x_k", "xk"),
                                ("x_v", "xv"), ("x_w", "xw")):
                put(q + short, self.W[p + "tm_" + full].ravel())
            put(q + "w0", self.W[p + "tm_w0col"].ravel())
            put(q + "w1", self.W[p + "tm_w1p"]); put(q + "w2", self.W[p + "tm_w2p"])
            for nm in ("r_w", "k_w", "v_w", "o_w"):
                put(q + nm + ".weight", self.W[p + "tm_W" + nm[0].lower()])
            put(q + "ln.weight", self.W[p + "tmln_g"].ravel())
            put(q + "ln.bias", self.W[p + "tmln_b"].ravel())
            c = s + "channel_mixing."
            put(c + "x_k", self.W[p + "cm_xk"].ravel())
            put(c + "key_w.weight", self.W[p + "cm_Wkey"])
            put(c + "value_w.weight", self.W[p + "cm_Wval"])
            if c + "ln" in sd:
                put(c + "ln", self.W[p + "cmln_g"].ravel())
        m.load_state_dict(sd, strict=True)
        m.eval()
        st = m.new_state(1)
        out = []
        with torch.no_grad():
            for t in range(self.T):
                lg, st = m.forward_one(int(self.TOKENS[t]), st)
                out.append(lg.numpy().copy())
        # ⚠ 形状要对齐：
        #   state[i][1] (kv) 是 [B, n_heads, D, D]，B=1 ⇒ squeeze 掉 batch 维
        #   我的 numpy 参考 `S` 是 [NL, n_heads, D, D]
        #   state[i][0] (x_prev) 是 [B, hidden] ⇒ squeeze 成 [hidden]
        # 不对齐会让 crosscheck 报出 1.7e-1 的**假**误差（一度让我误判成
        # 多层 bug，实际 logits 只差 9.5e-7，说明前向数学是对的）。
        Ss = np.array([st[l][1].numpy().reshape(self.NC, self.D, self.D)
                       for l in range(self.NL)], np.float32)
        Xp = np.array([st[l][0].numpy().reshape(self.HD)
                       for l in range(self.NL)], np.float32)
        return dict(logits=np.array(out, np.float32), S=Ss, xprev=Xp,
                    ref_logits=logits, ref_S=S, ref_xprev=xp), None

    # ---------------- 落盘 ----------------
    def dump(self, out_dir, prefix="step"):
        os.makedirs(out_dir, exist_ok=True)
        self.build_vml()
        vml = os.path.join(out_dir, f"{prefix}.vml")
        # ⚠ 绝不能用 PowerShell Out-File -Encoding utf8（BOM → 汇编器「未知指令」）
        with open(vml, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(self.lines) + "\n")

        (logits, S, xp, xf, xs) = self.reference()
        for name, arr in self.W.items():
            np.ascontiguousarray(arr, np.float32).tofile(
                os.path.join(out_dir, f"{name}.bin"))
        np.savetxt(os.path.join(out_dir, "ref_logits.txt"), logits, fmt="%.8f")
        S.tofile(os.path.join(out_dir, "ref_S.bin"))
        xp.tofile(os.path.join(out_dir, "ref_xprev.bin"))
        np.ascontiguousarray(xs, np.float32).tofile(
            os.path.join(out_dir, "ref_x_steps.bin"))
        lg5, S5, xp5, _, xs5 = self.reference(eps=EPS_ENGINE)
        np.savetxt(os.path.join(out_dir, "ref_eps5_logits.txt"), lg5, fmt="%.8f")
        S5.tofile(os.path.join(out_dir, "ref_eps5_S.bin"))
        xp5.tofile(os.path.join(out_dir, "ref_eps5_xprev.bin"))
        np.ascontiguousarray(xs5, np.float32).tofile(
            os.path.join(out_dir, "ref_eps5_x_steps.bin"))

        with open(os.path.join(out_dir, "weights.args"), "w",
                  encoding="utf-8", newline="\n") as f:
            rows = []
            for name, arr in self.W.items():
                a = np.asarray(arr)
                r = a.shape[0] if a.ndim >= 1 else 1
                c = a.shape[1] if a.ndim == 2 else 1
                rows.append(f"{name}:{name}.bin:{r}:{c}")
            f.write("\n".join(rows) + "\n")

        with open(os.path.join(out_dir, "cfg.json"), "w", encoding="utf-8") as f:
            json.dump({**self.c, "tokens": self.TOKENS.tolist(), "seed": SEED,
                       "n_instructions": len(self.lines),
                       "n_tensors": len(self.W)}, f, indent=2)
        return vml


def main():
    ap = argparse.ArgumentParser(description="ESERNN .vml 生成器（参数化放大）")
    ap.add_argument("--cfg", choices=["mini", "real"], default=None)
    ap.add_argument("--vocab", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--heads", type=int, default=None)
    ap.add_argument("--head-dim", type=int, default=None)
    ap.add_argument("--inter", type=int, default=None)
    ap.add_argument("--decay-hidden", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="step")
    ap.add_argument("--ckpt", default=None,
                    help="真实 checkpoint 路径（.pt）；给了就用真实权重而非随机初始化")
    ap.add_argument("--ckpt-key", default="intuition")
    ap.add_argument("--crosscheck", action="store_true",
                    help="与真实 ESERNN 类做源码级交叉验证")
    a = ap.parse_args()

    cfg = dict(MINI if (a.cfg or "mini").startswith("mini") else REAL)
    if a.cfg == "real":
        cfg = dict(REAL)
    for k, v in (("vocab", a.vocab), ("hidden", a.hidden), ("heads", a.heads),
                 ("head_dim", a.head_dim), ("inter", a.inter),
                 ("decay_hidden", a.decay_hidden), ("layers", a.layers),
                 ("eps", a.eps), ("T", a.T)):
        if v is not None:
            cfg[k] = v
    # 若只给 hidden 没给 heads/head_dim，按真实比例（8 头）自动配
    if a.hidden is not None and a.heads is None and a.head_dim is None:
        cfg["heads"] = 8
        cfg["head_dim"] = cfg["hidden"] // 8
        if a.inter is None:
            cfg["inter"] = cfg["hidden"] * 3
        if a.decay_hidden is None:
            cfg["decay_hidden"] = 32

    ck = None
    if a.ckpt:
        import torch
        d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        ck = d[a.ckpt_key] if a.ckpt_key in d else d
        if a.hidden is None:
            cfg["hidden"] = int(ck["ln0.weight"].numel())
        if a.layers is None:
            cfg["layers"] = len(sorted({int(k.split(".")[1])
                                        for k in ck if k.startswith("layers.")}))
        if a.heads is None:
            cfg["heads"] = 8
        if a.head_dim is None:
            cfg["head_dim"] = cfg["hidden"] // cfg["heads"]
        if a.inter is None:
            cfg["inter"] = int(ck["layers.0.channel_mixing.key_w.weight"].shape[0])
        if a.decay_hidden is None:
            cfg["decay_hidden"] = int(ck["layers.0.time_mixing.w1"].shape[1])
        if a.vocab is None:
            cfg["vocab"] = int(ck["embed_tokens.weight"].shape[0])

    g = Gen(cfg, ckpt=ck, seed=a.seed)
    g.build_weights()
    vml = g.dump(a.out_dir, a.prefix)

    n_ins = len(g.lines)
    print(f"=== 生成 {a.prefix}.vml ===")
    print(f"  vocab={g.V} hidden={g.HD} n_heads={g.NC} head_dim={g.D} "
          f"inter={g.F} decay_hidden={g.DH} layers={g.NL} eps={g.EPS:g} T={g.T}")
    print(f"  指令行数（含注释/空行） = {n_ins}")
    print(f"  权重张量 {len(g.W)} 个")
    print(f"  输出目录: {a.out_dir}")

    if a.crosscheck:
        r, err = g.crosscheck()
        if r is None:
            print(f"[交叉验证] 跳过（{err}）")
        else:
            for k in ("logits", "S", "xprev"):
                d = float(np.abs(r[k] - r["ref_" + k]).max())
                print(f"  numpy 参考 vs 真实 ESERNN  {k:7s} 最大绝对差 = {d:.3e}")


if __name__ == "__main__":
    main()
