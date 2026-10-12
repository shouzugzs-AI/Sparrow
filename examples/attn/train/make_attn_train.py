#!/usr/bin/env python3
"""
P8 成员 F：Transformer 训练闭环 —— 权重 / 参考梯度 / .vml 生成器

最小配置（先证明训练闭环正确，再谈放大）：
  T=4, D=8, NH=1, HD=8, F=16, V=8

结构照 examples/attn/block.vml 的前向数学（多头注意力 + 因果掩码 + ReLU FFN）：
    x = tok_emb + pos
    xd = LN1(x)
    qkv = xd·Wqkv^T + bqkv            ; 一次大 GEMM [T,3D]
    Q,K,V = split(qkv)
    att = softmax(Q·K^T / sqrt(hd), causal)    ; 逐头
    y = att·V
    x = x + (y·Wproj^T + bproj)                ; 残差
    xh = LN2(x)
    h = relu(xh·W1^T + b1)
    x = x + (h·W2^T + b2)                      ; 残差
    logits = LN_f(x)·E^T

本脚本产出：
  1. 权重 .bin（VMLM 常量池直接读的裸 f32 行主序）
  2. ref_logits.txt / ref_att.txt（Python 前向参考，回归用）
  3. 单批次固定输入 tok.bin / target.bin / pos.bin
  4. torch_grads.txt（**PyTorch autograd 解析梯度**，torch.set_num_threads(1)）
  5. torch_exact_grads.txt（**f64 精确 exp 的 softmax 前向 + torch autograd**）
     —— 用来区分「实现错」与「kernels::exp_f 6.51e-4 近似污染」（P7-A 方法论）
  6. train_step.vml —— 前向 + 完整反向 + AdamW 的训练步指令流

用法：
  python examples/attn/train/make_attn_train.py
"""
import os
import sys
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ---------------- 维度（最小配置） ----------------
T, D, NH, HD, F, V = 4, 8, 1, 8, 16, 8
assert NH * HD == D

# 单批次固定输入（过拟合测试用）
TOK = [1, 2, 3, 0]
TGT = [2, 3, 0, 1]

SCALE = 0.08
LR, B1, B2, EPS, WD = 3e-2, 0.9, 0.999, 1e-8, 0.0


def main():
    rng = np.random.default_rng(2025)

    def rnd(*s, scale=SCALE):
        return (rng.standard_normal(s) * scale).astype(np.float32)

    W = {
        "Wqkv": rnd(3 * D, D),
        "bqkv": rnd(1, 3 * D),
        "Wproj": rnd(D, D),
        "bproj": rnd(1, D),
        "W1": rnd(F, D),
        "b1": rnd(1, F),
        "W2": rnd(D, F),
        "b2": rnd(1, D),
        "ln1_g": np.ones((1, D), dtype=np.float32),
        "ln1_b": np.zeros((1, D), dtype=np.float32),
        "ln2_g": np.ones((1, D), dtype=np.float32),
        "ln2_b": np.zeros((1, D), dtype=np.float32),
        "lnf_g": np.ones((1, D), dtype=np.float32),
        "lnf_b": np.zeros((1, D), dtype=np.float32),
        "E": rnd(V, D),
        "pos": rnd(T, D),
        "tok": np.array(TOK, dtype=np.float32),        # [T] 行索引，EMB 用
        "target": np.array(TGT, dtype=np.float32),     # [T] 逐位置标签，CE_BWD 用
        "inv_sqrt_hd": np.array([1.0 / np.sqrt(HD)], dtype=np.float32),
        "inv_t": np.array([1.0 / T], dtype=np.float32),
        "hparams": np.array([LR, B1, B2, EPS, WD], dtype=np.float32),
    }
    for k, v in W.items():
        v.astype(np.float32).tofile(os.path.join(HERE, f"{k}.bin"))
    # 逐位置的 token 索引标量槽（每个位置一个 [1,1] 槽，供 EMB/EMB_BWD 用）
    # 以及逐位置的标签标量槽（CE_BWD 每位置单样本，需要 [1,1] 目标）
    for r in range(T):
        np.array([TOK[r]], dtype=np.float32).tofile(os.path.join(HERE, f"tokr{r}.bin"))
        np.array([float(r)], dtype=np.float32).tofile(os.path.join(HERE, f"posr{r}.bin"))
        np.array([float(TGT[r])], dtype=np.float32).tofile(os.path.join(HERE, f"tgt{r}.bin"))

    # ---------------- Python 前向参考（与 block.vml 逐条对应） ----------------
    def ln(x, g, b, eps=1e-5):
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + eps) * g + b

    def softmax_causal(s):
        n = s.shape[-1]
        mask = np.tril(np.ones((n, n), dtype=bool))
        s = np.where(mask, s, -1e30)
        m = s.max(-1, keepdims=True)
        e = np.exp(s - m)
        return e / e.sum(-1, keepdims=True)

    tok = W["tok"].astype(int)
    x = W["E"][tok] + W["pos"]
    xd = ln(x, W["ln1_g"], W["ln1_b"])
    qkv = xd @ W["Wqkv"].T + W["bqkv"]
    q, k, v = np.split(qkv, 3, axis=-1)
    q = q.reshape(T, NH, HD).transpose(1, 0, 2)
    k = k.reshape(T, NH, HD).transpose(1, 0, 2)
    v = v.reshape(T, NH, HD).transpose(1, 0, 2)
    att = softmax_causal(q @ k.transpose(0, 2, 1) / np.sqrt(HD))
    y = att @ v
    y = y.transpose(1, 0, 2).reshape(T, D)
    x = x + (y @ W["Wproj"].T + W["bproj"])
    xh = ln(x, W["ln2_g"], W["ln2_b"])
    h = np.maximum(xh @ W["W1"].T + W["b1"], 0)
    x = x + (h @ W["W2"].T + W["b2"])
    xf = ln(x, W["lnf_g"], W["lnf_b"])
    logits = xf @ W["E"].T

    np.savetxt(os.path.join(HERE, "ref_logits.txt"), logits, fmt="%.8f")
    np.savetxt(os.path.join(HERE, "ref_att.txt"), att.reshape(-1, T), fmt="%.8f")

    # 初值 loss（交叉熵，逐位置平均）
    def ce_loss(lg, tgt):
        m = lg.max(-1, keepdims=True)
        lse = m + np.log(np.exp(lg - m).sum(-1, keepdims=True))
        return float((lse[:, 0] - lg[np.arange(len(tgt)), tgt]).mean())

    loss0 = ce_loss(logits, TGT)
    print(f"[gen] T={T} D={D} NH={NH} HD={HD} F={F} V={V}")
    print(f"[gen] Python 参考初值 loss = {loss0:.6f}  (期望 ~2.0 = ln(8)={np.log(8):.4f})")
    print(f"[gen] 参考 logits[0] = {np.round(logits[0], 5)}")
    print(f"[gen] att[0] 因果性（上三角应为 0）:\n{np.round(att[0], 5)}")

    if torch is None:
        print("[gen] 未安装 torch，跳过 PyTorch 梯度参考")
        return

    torch.set_num_threads(1)
    write_torch_grads(W)
    write_vml(W)


# ---------------------------------------------------------------------------
# PyTorch autograd 参考梯度（两份：近似 exp 前向 / f64 精确 exp 前向）
# ---------------------------------------------------------------------------
NAME_ORDER = [
    "Wqkv", "bqkv", "Wproj", "bproj",
    "W1", "b1", "W2", "b2",
    "ln1_g", "ln1_b", "ln2_g", "ln2_b", "lnf_g", "lnf_b",
    "E", "pos",
]
# 注意：pos 参与但 tok_emb 用的是 E，二者都训练；tok 是整数索引不训练。


def _build_graph(W, exact_exp):
    """返回 (params_dict_of_tensors, loss, 前向中间量 dict)"""
    t = {}
    for n in ["Wqkv", "bqkv", "Wproj", "bproj", "W1", "b1", "W2", "b2",
              "ln1_g", "ln1_b", "ln2_g", "ln2_b", "lnf_g", "lnf_b", "E", "pos"]:
        t[n] = torch.tensor(W[n].copy(), requires_grad=True, dtype=torch.float32)

    tok = torch.tensor(W["tok"].astype(np.int64))
    tgt = torch.tensor(W["target"].astype(np.int64))

    def ln(x, g, b, eps=1e-5):
        mu = x.mean(-1, keepdim=True)
        var = ((x - mu) ** 2).mean(-1, keepdim=True)
        return (x - mu) / torch.sqrt(var + eps) * g + b

    if exact_exp:
        # f64 精确 exp 的 softmax（排除 kernels::exp_f 的 6.51e-4 近似污染）
        def softmax_c(s):
            s64 = s.double()
            n = s64.shape[-1]
            mask = torch.tril(torch.ones(n, n, dtype=torch.bool))
            s64 = s64.masked_fill(~mask, float("-inf"))
            m = s64.max(-1, keepdim=True).values
            e = torch.exp(s64 - m)
            e = torch.nan_to_num(e, nan=0.0)
            return (e / e.sum(-1, keepdim=True)).float()
    else:
        def softmax_c(s):
            n = s.shape[-1]
            mask = torch.tril(torch.ones(n, n, dtype=torch.bool))
            s = s.masked_fill(~mask, float("-inf"))
            return torch.softmax(s, dim=-1)

    x = t["E"][tok] + t["pos"]
    xd = ln(x, t["ln1_g"], t["ln1_b"])
    qkv = xd @ t["Wqkv"].T + t["bqkv"]
    q, k, v = torch.split(qkv, D, dim=-1)
    # [T,D] -> [NH,T,HD]
    q = q.reshape(T, NH, HD).permute(1, 0, 2)
    k = k.reshape(T, NH, HD).permute(1, 0, 2)
    v = v.reshape(T, NH, HD).permute(1, 0, 2)
    S = q @ k.transpose(1, 2) * (1.0 / np.sqrt(HD))
    att = softmax_c(S)
    y = att @ v
    y = y.permute(1, 0, 2).reshape(T, D)
    x = x + (y @ t["Wproj"].T + t["bproj"])
    xh = ln(x, t["ln2_g"], t["ln2_b"])
    hpre = xh @ t["W1"].T + t["b1"]
    h = torch.relu(hpre)
    x = x + (h @ t["W2"].T + t["b2"])
    xf = ln(x, t["lnf_g"], t["lnf_b"])
    logits = xf @ t["E"].T

    loss = torch.nn.functional.cross_entropy(logits, tgt)
    return t, loss, {"att": att, "logits": logits}


def _dump(t, loss, path, loss_path):
    loss.backward()
    vals = []
    names = []
    for n in NAME_ORDER:
        g = t[n].grad
        vals.append(g.detach().numpy().astype(np.float32).ravel())
        names.append(n)
    allg = np.concatenate(vals)
    allg.tofile(path)
    with open(loss_path, "w") as f:
        f.write(f"{loss.item():.9f}\n")
        for n, g in zip(names, vals):
            f.write(f"{n} {g.size}\n")
    return loss.item(), allg


def write_torch_grads(W):
    # (1) 与前向 kernel 同族的近似 exp 前向
    t, loss, mid = _build_graph(W, exact_exp=False)
    l1, g1 = _dump(t, loss, os.path.join(HERE, "torch_grads.bin"),
                   os.path.join(HERE, "torch_grads.meta"))
    print(f"[gen] torch(近似exp) loss = {l1:.6f}  梯度元素数 = {g1.size}")

    # (2) f64 精确 exp 的 softmax 前向（用于区分实现错 / exp 近似污染）
    t2, loss2, mid2 = _build_graph(W, exact_exp=True)
    l2, g2 = _dump(t2, loss2, os.path.join(HERE, "torch_exact_grads.bin"),
                   os.path.join(HERE, "torch_exact_grads.meta"))
    print(f"[gen] torch(精确exp) loss = {l2:.6f}  梯度元素数 = {g2.size}")

    # 落一份 att（供 Rust 侧因果性回归对比）
    np.savetxt(os.path.join(HERE, "ref_att.txt"),
               mid2["att"].detach().numpy().reshape(-1, T), fmt="%.8f")


# ---------------------------------------------------------------------------
# .vml 生成
# ---------------------------------------------------------------------------
from gen_vml import gen_vml  # noqa: E402  （同目录，脚本直接跑时可用）


def write_vml(W):
    src = gen_vml()
    with open(os.path.join(HERE, "train_step.vml"), "w", newline="\n") as f:
        f.write(src)
    print(f"[gen] 写出 train_step.vml（{src.count(chr(10))} 行）")

    # 汇编参数清单：vmlm-asm <vml> <vmb> name:file:rows:cols ...
    # 写成每行一个 const 的 .args 文件，供 build.ps1 / Rust 驱动读取，
    # 避免在命令行里手抄 28 个常量（抄错就静默错位）。
    consts = [
        ("Wqkv", 3 * D, D), ("bqkv", 1, 3 * D), ("Wproj", D, D), ("bproj", 1, D),
        ("W1", F, D), ("b1", 1, F), ("W2", D, F), ("b2", 1, D),
        ("ln1_g", 1, D), ("ln1_b", 1, D), ("ln2_g", 1, D), ("ln2_b", 1, D),
        ("lnf_g", 1, D), ("lnf_b", 1, D),
        ("E", V, D), ("pos", T, D),
        ("inv_sqrt_hd", 1, 1), ("inv_t", 1, 1), ("hparams", 1, 5),
    ]
    for r in range(T):
        consts.append((f"tokr{r}", 1, 1))
    for r in range(T):
        consts.append((f"posr{r}", 1, 1))
    for r in range(T):
        consts.append((f"tgt{r}", 1, 1))

    with open(os.path.join(HERE, "asm.args"), "w", newline="\n") as f:
        for name, rows, cols in consts:
            f.write(f"{name}:{name}.bin:{rows}:{cols}\n")
    print(f"[gen] 写出 asm.args（{len(consts)} 个常量）")


if __name__ == "__main__":
    main()
