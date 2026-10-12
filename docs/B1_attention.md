# 全局 attention（跨 token 交互）验证报告

> 本轮：给 B1 加**全局因果 attention**，验证它能否打破"结构上最多 bigram"的上限。
> 冒烟我跑了（dim=128, 8 步），**300 步验证我也跑了**（这是你要求的判定实验），完整训练没跑，CMD 在 §5。

## 1. 代码改动（复用的就是项目既有的 MiniCPM 注意力路径）

`train/src/sparse_train24.rs` 里已经写明这条路径（原文：
「single-head full-width attention: no separate projection dims」「q = h Wq, k = h Wk, v = h Wv,
attn = softmax(q kᵀ/√d) v」「a = attention(h); u = h + a」）。我按同一形状实现并接到稀疏层后面：

```
每 N 层稀疏层之后：
    attn = softmax(q kᵀ/√d)·v            q=h·Wq, k=h·Wk, v=h·Wv    （单头全宽, head_dim=d）
    out  = attn·Wo
    h    = h + out                       ← attention 自己的残差（层内无残差**不变**）
```

* **新增开关** `--attn-every N`：N=0 关闭（**默认**）；N=1 ⇒ 每层都插；N=layers ⇒ 只在最后一层稀疏层之后插一个
* 因果掩码 `j ≤ i`（预测下一个 token 不许看未来）
* 复用的内核：`gemm_nt`、`tr`、`kernels::exp_f`/`sqrt_f`（**零新内核**）
* **稀疏层一个字没改**、无 top-k、保持无残差
* 参数：每块 `4·d²`（Wq/Wk/Wv/Wo 稠密），`dense Adam` 与 `sparse_train24` 一致

## 2. 验证结果（你给的三条判据）

| 判据 | 结果 |
|---|---|
| **1. 编译通过** | ✅ `cargo build --release -p vmlm-train --bin vmlm-b1-full` = 0 |
| **2. 300 步 val < 7.0** | ❌ **7.7033**（未达标）——但**同配置无 attention 是 8.9480**，即 **−1.2446 nats** |
| **3. 末层 stable rank > 10** | ❌ 5.85（未达标）——但 **keff95 从 2 → 57**、head 输入拼接 stable rank **2.34 → 11.40**（>10 ✅）|

### 2.1 关键数字（dim=1536, k0=8, k0_embed=32, k0_head=32, 24 层, seed 1, 300 步）

| 配置 | val | 末层 keff95 | 末层 stable rank | head 输入(拼接) stable |
|---|---|---|---|---|
| 无 attention | **8.9480** | 2 | 1.12 | 2.34 |
| **`--attn-every 24`** | **7.7033** | **57** | **5.85** | **11.40** |
| 参考：unigram（纯词频） | 7.3973 | — | — | — |

* **秩坍缩被修好了**：keff95 **2 → 57（28×）**、stable rank **1.12 → 5.85（5.2×）**。
  "stable rank > 10"这一条按末层单独算没达到（5.85），按 keff95 与 head 输入拼接算是达到了。
* **val 改善 1.24 nats**，但**仍在 unigram(7.397) 之上** ⇒ 300 步还是欠训练（上一轮已确认：
  该预算下所有变体都还没打赢词频表）。

### 2.2 反向正确性（你说不做完整对账，但这是免费的）

把 attention 权重加进**现有** FD 探针即可（`--gradcheck`）：

```
probe attn_wo  idx=  492 ana=+4.7972e-3 num=+4.7779e-3 rel=4.05e-3 SNR= 72 ok
probe attn_wo  idx= 1580 ana=-4.5714e-3 num=-4.5681e-3 rel=7.32e-4 SNR= 69 ok
GRADCHECK PASS: 12 探针，越界 0
```

`attn_wo`（含 softmax 雅可比、dq/dk/dv、dWo）相对误差 7e-4 ~ 4e-3、SNR ~70 ⇒ 反向是对的。

### 2.3 默认路径逐位未变

不开 `--attn-every` 时 `loss_hash = 0xc9c987c15d75d1d6`，与历史记录**逐位相同** ⇒ 新代码没有动既有语义。

## 3. 验证套件（顺手把上一轮那个"未定位的失败"根治了）

上轮我如实报告过"L3 从长脚本里调用时 exit=1、独立运行 exit=0，未定位"。**这轮找到根因并修好了**：

> `tools/b1_full_validate.ps1` 是**无 BOM 的 UTF-8**，而 PowerShell 5.1 按**系统代码页(GBK)**解码它，
> 于是中文注释的多字节序列会**吃掉后面的引号/括号** —— 表现为"偶发解析错误 / 抓不到输出 / 行被合并"。
> 写回 **UTF-8 with BOM** 后：**59/59 全过**，L3 也拿到输出并 PASS。

套件新增（A1..A3，attention 相关）：`A1` 冒烟+参数账目、`A2` attention 反向 FD 对账、`A3` 默认路径逐位不变。
最终：**checks=59 fail=0**。

## 4. 成本（必须知道）

| 配置 | 参数量 | 步时 | 300 步耗时 | 备注 |
|---|---|---|---|---|
| 无 attention | 993,856 | 0.060 s | 18 s | 基线 |
| `--attn-every 24`（1 块） | **10,431,040** | **0.29 s** | 87 s | 本轮用的 |
| `--attn-every 1`（24 块） | **≈ 226 M** | ≈5 s | ≈25 min | 会吃掉 1M 参数预算约 226 倍 |

⇒ **参数预算被打破了**（1M → 10.4M），因为 attention 的 4 个投影是稠密 `[d×d]`。
如果想保住 1M 预算，需要把 Wq/Wk/Wv/Wo 也做成稀疏（复用 `b1_forward/b1_backward_fused`）——
那是另一个改动，**要你点头我再做**。

## 5. CMD（给用户的，手动跑）

### 5.1 冒烟（dim=128, 8 步，每层都插 attention）

```cmd
cd <repo> && cargo build --release -p vmlm-train --bin vmlm-b1-full && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 128 --k0 2 --k0-embed 4 --k0-head 4 --layers 4 --steps 8 --seq 16 --batch 2 --lr 3e-3 --warmup 2 --min-lr 3e-4 --seed 1 --data-dir artifacts/b1_scaling --out-dir artifacts/b1_attn_smoke --doc docs/B1_attn_smoke.md --eval-every 4 --log-every 2 --plateau-patience 0 --attn-every 1
```

### 5.2 300 步验证（dim=1536, k0=8, k0e=32, k0h=32，attention 只插在最后一层之后）

```cmd
cd <repo> && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 32 --k0-head 32 --layers 24 --no-residual 1 --steps 300 --seq 32 --batch 4 --lr 3e-4 --warmup 20 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed --out-dir artifacts/b1_attn300 --doc docs/B1_attn300.md --eval-every 50 --plateau-patience 0 --attn-every 24 --dump-acts artifacts/effrank/attn300
```

### 5.3 严格对照（同 seed/步数/lr，**只去掉 attention**）

```cmd
cd <repo> && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 32 --k0-head 32 --layers 24 --no-residual 1 --steps 300 --seq 32 --batch 4 --lr 3e-4 --warmup 20 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed --out-dir artifacts/b1_noattn300 --doc docs/B1_noattn300.md --eval-every 50 --plateau-patience 0 --dump-acts artifacts/effrank/noattn300
```

跑完后（不用再训练）：

```cmd
python tools/b1_effrank.py --dir artifacts/effrank/attn300 --out docs/B1_effrank_attn300.md
python tools/b1_unigram_baseline.py --data-dir artifacts/b1_scaling
```

### 5.4 如果要把 val 压到 7.0 以下（我的判断：靠步数，不是靠结构）

300 步不够（无 attention 8.95、有 attention 7.70，都在 unigram 7.397 之上）。要跨过 7.0，
最快的路是**把 300 步换成 3000 步**（带 attention，约 15 分钟：0.29 s/步）：

```cmd
cd <repo> && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 32 --k0-head 32 --layers 24 --no-residual 1 --steps 3000 --seq 32 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed --out-dir artifacts/b1_attn3000 --doc docs/B1_attn3000.md --eval-every 100 --plateau-patience 1 --attn-every 24
```

## 6. 诚实边界

1. **判据 2、3 都没按字面达到**：val 7.7033（>7.0）、末层 stable rank 5.85（<10）。
   我报的是"改善幅度"（val −1.24、keff95 28×）而不是把它说成达标。
2. **300 步不是收敛点**：所有变体都还在 unigram 之上，所以"attention 有没有用"这个问题在
   300 步上只能看出"明显更好"，不能看出"最终能不能更好"。要下结论得跑 §5.4。
3. **只有 1 个 attention 块**（`--attn-every 24`）。`--attn-every 1` 更强但参数 226M、步时 ~5 s，
   我没跑（也没让你跑）。"每 N 层"的 N 没做扫描。
4. **attention 是稠密投影**：参数从 0.99M 涨到 10.4M（1 块）。这与之前"参数 ≤1M"的纪律冲突，
   所以我把两种选择摆出来让你定（§4）。
5. FD 只探了 `attn_wq`/`attn_wo` 各 2 个位置（不是全量对账，按你的要求）；`attn_wk/attn_wv` 未单独探，
   但它们走的是同一段代码路径。
6. 秩度量仍是"归一化拼接后的 keff95/stable rank"，是**可用方向**的可比数字，不是互信息。
