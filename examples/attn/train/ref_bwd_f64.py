"""第三份独立实现：纯 numpy f64 的 Transformer 手写反向。

存在的理由（P7-A 方法论的延伸）：
  * VMLM 的 `.vml` 是一份实现
  * PyTorch autograd 是第二份（但它可能和图结构不一致，且我看不见它的图）
  * 本文件是第三份：**不依赖 VMLM 引擎、不依赖 torch、不依赖我的 .vml 生成器**，
    完全用 numpy f64 按同样的数学重写一遍

三份互相对拍时，任何两份一致、第三份不同 ⇒ 能直接判定是「实现错」还是「参考错」。
这个文件在 P8-F 里立了大功：它首先暴露出「FFN 10 个参数一致、注意力 6 个不一致」，
把 bug 范围从「整个 Transformer 反向」缩到「注意力 dV 一行」。

用法（repo 根目录）：
    python examples/attn/train/ref_bwd_f64.py

注意：脚本里会 `print` 与 torch 参考的对拍表，所以先跑 make_attn_train.py 生成参考。
"""
import numpy as np

T, D, HD, V, F = 4, 8, 8, 8, 16
DIR = 'examples/attn/train/'
rd = lambda n: np.fromfile(DIR + n + '.bin', dtype=np.float32)


def ln(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


E = rd('E').reshape(V, D)
pos = rd('pos').reshape(T, D)
Wqkv = rd('Wqkv').reshape(3 * D, D)
bqkv = rd('bqkv').reshape(1, 3 * D)
Wproj = rd('Wproj').reshape(D, D)
bproj = rd('bproj').reshape(1, D)
W1 = rd('W1').reshape(F, D)
b1 = rd('b1').reshape(1, F)
W2 = rd('W2').reshape(D, F)
b2 = rd('b2').reshape(1, D)
ln1_g, ln1_b = rd('ln1_g'), rd('ln1_b')
ln2_g, ln2_b = rd('ln2_g'), rd('ln2_b')
lnf_g, lnf_b = rd('lnf_g'), rd('lnf_b')
tok = [1, 2, 3, 0]
tgt = [2, 3, 0, 1]

# ---- 前向 ----
x0 = E[tok] + pos
xd = ln(x0, ln1_g, ln1_b)
qkv = xd @ Wqkv.T + bqkv
q, k, v = np.split(qkv, 3, axis=-1)
S = q @ k.T / np.sqrt(HD)
mask = np.tril(np.ones((T, T), bool))
Sm = np.where(mask, S, -np.inf)
m = Sm.max(-1, keepdims=True)
e = np.exp(Sm - m)
att = e / e.sum(-1, keepdims=True)
y = att @ v
x1 = x0.copy()
x = x0 + (y @ Wproj.T + bproj)
xh = ln(x, ln2_g, ln2_b)
hpre = xh @ W1.T + b1
h = np.maximum(hpre, 0)
x2 = x.copy()
x = x + (h @ W2.T + b2)
xf = ln(x, lnf_g, lnf_b)
logits = xf @ E.T

# ---- 反向（手写 f64，与 .vml 同结构） ----
x0_ = xf
dlogits = np.zeros_like(logits)
for i in range(T):
    p = np.exp(logits[i] - logits[i].max())
    p = p / p.sum()
    dlogits[i] = p
    dlogits[i, tgt[i]] -= 1.0
# torch 用 mean，这里也除 T 便于和 torch 对齐
dlogits_t = dlogits / T

g_E = dlogits_t.T @ xf
dxf = dlogits_t @ E
# LN_f
mu = xf.mean(-1, keepdims=True)  # 占位
def ln_bwd(dy, xin, g):
    mu = xin.mean(-1, keepdims=True)
    var = xin.var(-1, keepdims=True)
    inv = 1 / np.sqrt(var + 1e-5)
    xs = (xin - mu) * inv
    dg = (dy * xs).sum(0)
    db = dy.sum(0)
    dx = inv * (dy * g - (dy * g).mean(-1, keepdims=True) - xs * (dy * g * xs).mean(-1, keepdims=True))
    return dx, dg, db


dx, dlnf_g, dlnf_b = ln_bwd(dxf, x, lnf_g)

g_W2 = dx.T @ h
g_b2 = dx.sum(0)
dh = dx @ W2
dhpre = dh * (hpre > 0)
g_W1 = dhpre.T @ xh
g_b1 = dhpre.sum(0)
dxh = dhpre @ W1
dx2, dln2_g, dln2_b = ln_bwd(dxh, x2, ln2_g)
dx = dx + dx2

g_Wproj = dx.T @ y
g_bproj = dx.sum(0)
dy = dx @ Wproj

datt = dy @ v.T
dv = att.T @ dy
dS = np.zeros_like(S)
for i in range(T):
    lim = i + 1
    dot = (datt[i, :lim] * att[i, :lim]).sum()
    dS[i, :lim] = att[i, :lim] * (datt[i, :lim] - dot)
dS = dS / np.sqrt(HD)

dq = dS @ k
dk = dS.T @ q
dqkv = np.concatenate([dq, dk, dv], axis=-1)

g_Wqkv = dqkv.T @ xd
g_bqkv = dqkv.sum(0)
dxd = dqkv @ Wqkv
dx1, dln1_g, dln1_b = ln_bwd(dxd, x0, ln1_g)
dx = dx + dx1

demb = np.zeros((V, D))
for i in range(T):
    demb[tok[i]] += dx[i]
dpos = dx.copy()
g_E = g_E + demb

print('=== Python f64 解析梯度（除 T） ===')
for nm, g in [('Wqkv', g_Wqkv), ('bqkv', g_bqkv), ('Wproj', g_Wproj), ('bproj', g_bproj),
              ('W1', g_W1), ('b1', g_b1), ('W2', g_W2), ('b2', g_b2),
              ('ln1_g', dln1_g), ('ln1_b', dln1_b), ('ln2_g', dln2_g), ('ln2_b', dln2_b),
              ('lnf_g', dlnf_g), ('lnf_b', dlnf_b), ('E', g_E), ('pos', dpos)]:
    print('  %-8s' % nm, np.round(np.asarray(g).ravel()[:6], 6))

# 与 torch 参考比
tg = np.fromfile(DIR + 'torch_grads.bin', dtype=np.float32)
OFF = {}
o = 0
for nm, sz in [('Wqkv', 3 * D * D), ('bqkv', 3 * D), ('Wproj', D * D), ('bproj', D),
               ('W1', F * D), ('b1', F), ('W2', D * F), ('b2', D),
               ('ln1_g', D), ('ln1_b', D), ('ln2_g', D), ('ln2_b', D),
               ('lnf_g', D), ('lnf_b', D), ('E', V * D), ('pos', T * D)]:
    OFF[nm] = (o, sz)
    o += sz
print('\n=== Python f64 vs torch（应该完全一致，验证我的参考结构没写错）===')
mine = {'Wqkv': g_Wqkv, 'bqkv': g_bqkv, 'Wproj': g_Wproj, 'bproj': g_bproj,
        'W1': g_W1, 'b1': g_b1, 'W2': g_W2, 'b2': g_b2,
        'ln1_g': dln1_g, 'ln1_b': dln1_b, 'ln2_g': dln2_g, 'ln2_b': dln2_b,
        'lnf_g': dlnf_g, 'lnf_b': dlnf_b, 'E': g_E, 'pos': dpos}
for nm in [k for k in OFF]:
    a = np.asarray(mine[nm]).ravel().astype(np.float64)
    b = tg[OFF[nm][0]:OFF[nm][0] + OFF[nm][1]].astype(np.float64)
    d = np.abs(a - b).sum()
    s = np.abs(a).sum() + np.abs(b).sum()
    print('  %-8s Σ|Δ|/(‖a‖+‖b‖) = %.3e' % (nm, d / s if s else 0))
