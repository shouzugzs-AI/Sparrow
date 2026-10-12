# -*- coding: utf-8 -*-
"""第三步(1)：参数预算 — 用与驱动**完全相同**的记账公式先算清楚，再决定配置。

驱动的记账（b1_full_train.rs）：
  main     = layers * dim * k0
  embed    = V * k0_embed
  head     = V * k0_head              (+ head-skip 时再来一份 => 2 份)
  attn     = n_attn * 4 * dim * k0_attn        (稀疏, --attn-sparse 1)
             n_attn = layers / attn_every       (--attn-every 4 => layers/4)
  total    = main + embed + head(+skip) + attn
V = 4054（字符表；n_vocab = 1 UNK + 4053 字符）
"""
V = 4054
rows = []


def calc(name, dim, k0, k0e, k0h, k0a, layers, every=4, skip=True, seq=128, batch=4):
    n_attn = layers // every
    main = layers * dim * k0
    emb = V * k0e
    head = V * k0h * (2 if skip else 1)
    attn = n_attn * 4 * dim * k0a
    tot = main + emb + head + attn
    return dict(name=name, dim=dim, k0=k0, k0e=k0e, k0h=k0h, k0a=k0a, layers=layers,
                n_attn=n_attn, main=main, emb=emb, head=head, attn=attn,
                total=tot, param_mb=tot * 4 / 1e6, opt_mb=tot * 12 / 1e6, seq=seq, batch=batch)


# 当前字符表最优（基数）
rows.append(calc("CHAR 基准 (现在)", 1536, 8, 32, 32, 8, 24))
# 候选
rows.append(calc("C1 ~1.8x", 2048, 12, 48, 48, 12, 24))
rows.append(calc("C2 ~4.0x", 3072, 20, 80, 80, 20, 24))
rows.append(calc("C3 ~10x", 4096, 32, 128, 128, 32, 32))
rows.append(calc("C4 ~12.7x(你原目标)", 4096, 40, 160, 160, 40, 32))
rows.append(calc("C5 ~10x 浅attn(every8)", 4096, 32, 128, 128, 32, 32, every=8))
rows.append(calc("C6 全稠密attn(参考,禁)", 2048, 12, 48, 48, 0, 24))
# C6 的稠密注意力另算
c6 = rows[-1]
c6["attn"] = 24 // 4 * 4 * 2048 * 2048
c6["total"] = c6["main"] + c6["emb"] + c6["head"] + c6["attn"]
c6["param_mb"] = c6["total"] * 4 / 1e6
c6["opt_mb"] = c6["total"] * 12 / 1e6

base = rows[0]["total"]
print("%-22s %6s %5s %5s %5s %5s %5s | %10s %10s %10s %10s | %10s %8s" %
      ("config", "dim", "k0", "k0e", "k0h", "k0a", "lyr", "main", "embed", "head(+skip)", "attn", "TOTAL", "x_base"))
for r in rows:
    print("%-22s %6d %5d %5d %5d %5d %5d | %10d %10d %10d %10d | %10d %7.2fx" %
          (r["name"], r["dim"], r["k0"], r["k0e"], r["k0h"], r["k0a"], r["layers"],
           r["main"], r["emb"], r["head"], r["attn"], r["total"], r["total"] / base))
print()
print("%-22s %10s %10s %10s" % ("config", "params(MB)", "Adam m+v(MB)", "权重+梯度+Adam(MB)"))
for r in rows:
    print("%-22s %10.1f %10.1f %10.1f" % (r["name"], r["param_mb"], r["opt_mb"], r["param_mb"] * 4))
print()
print("激活/中间缓冲（每步, f32, batch=4）: h_ins = layers*T*dim*4B")
print("%-22s %8s %10s %12s %12s" % ("config", "seq", "T=4*seq", "h_ins(MB)", "+其它~20%(MB)"))
for r in rows:
    for sq in (64, 128, 256):
        T = 4 * sq
        h = r["layers"] * T * r["dim"] * 4 / 1e6
        print("%-22s %8d %10d %12.1f %12.1f" % (r["name"], sq, T, h, h * 1.2))
    print()
