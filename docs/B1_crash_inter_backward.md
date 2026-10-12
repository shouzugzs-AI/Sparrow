# 崩溃定位与修复：inter + pool + window 1（exit 0xC0000005）

## 结论：根因是 **`dwp`（反向的每线程 dw 暂存）在 inter 下只给了 1 份，而反向内核按全局 `nth` 分段**

### 定位过程（按你给的三步思路走，但用了更快的二分）

| 步骤 | 做法 | 结果 |
|---|---|---|
| 1 | 形状扫描：1M 配置（dim1536/L24/k0 8/k0e 32）inter+window+pool | **全部 exit 0**（连跑 50 步也稳）⇒ 与用户环境不一致 |
| 2 | 换成 **C3/10M 几何**（dim4096/L32/k0 32/k0e 128/k0a 32）| **稳定 0xC0000005** ⇒ 复现成功 |
| 3 | 按 inter / pool / window 三向二分 | **只有 inter 崩**；`--persistent-threads 0/1` 都崩、`--attn-window 0/1` 都崩 ⇒ **与线程池、窗口都无关** |
| 4 | 几何再二分（threads 1/2/8、batch 16、dim 2048、k0a 8）| **都崩** ⇒ 不是 dim/线程数/attn 引起；但 1M 几何（k0 8/k0e 32）不崩 ⇒ **与 k0 / k0_embed 规模相关** |
| 5 | 阶段标记 `--attn-debug 1`（新加，1=阶段 2=逐窗口细节）| 4 个 region **前向全做完**（`fwd done`），然后崩 ⇒ **崩在反向** |
| 6 | 关组件：`--attn-every 0` / `--head-skip 0` / `--layers 4` | **全都崩** ⇒ 是**稀疏层反向**（`par_b1_backward`）本身 |

### 根因

`dwp` 是"每线程私有 dw 分段"的暂存，大小 `nth * max_wlen`（`max_wlen` = 所有需要累加的权重里最大的那个）。
inter-op 里我把每个 region 的暂存切成 `slot * max_wlen`，**只给了 `max_wlen` 一份**。
而反向内核（`par_b1_backward` / `attn_backward_sparse_par`）内部按 `tid * m` 分段，**它按全局线程数 `nth` 用**：
即使外层传 `nt_in = 1`，仍会写到 `nth * m` 的范围 ⇒ 越界。

规模依赖正好对上：
* 1M：`nth*m = 4*1536*8 = 49,152 ≤ max_wlen = 4054*32 = 129,728` ⇒ 不越界（所以之前一直没崩）
* C3：`nth*m = 4*4096*32 = 524,288 > max_wlen = 4054*128 = 518,912` ⇒ **越界 5,376 f32 (21 KB)** ⇒ 偶发访问冲突 ✓

### 修法（1 行）

```rust
// 分配
let dwp = alloc(nslot.max(1) * nth.max(1) * max_wlen);   // 每个 region 给足 nth*max_wlen
// 重绑定
let dwp = dwp.add(slot * nth.max(1) * max_wlen);
```

### 验收（全部实测）

| 用例 | 修前 | 修后 |
|---|---|---|
| C3 + inter + window0 + threads4（×3 次）| 0xC0000005 | **exit 0 ×3，`final_train=8.307563` 完全相同** ✓ |
| C3 + inter + **window1 + pool + threads4**（你报的组合）| 崩 | **exit 0** ✓ |
| C3 + inter + window1 + pool + **threads8** | 崩 | **exit 0** ✓ |
| C3 + `--attn-every 0` / `--head-skip 0` / `--layers 4` | 全崩 | **全 exit 0** ✓ |
| 1M：intra+pool vs intra+spawn | — | **逐位相同** `0x3dff4da1857fc065` ✓ |
| 1M：inter+pool vs inter+spawn | — | **逐位相同** `0xb5957ca1fc7b11f6` ✓ |
| 1M：inter vs intra（window1，10 步）| — | `8.305943` vs `8.305963`，**Δ=2.0e-5** ✓ |
| intra 路径逐位不变 | — | `0xc9c987c15d75d1d6` ✓ |
| 套件 / build_all / tests / b1-verify / guard | — | **65/65** / OK（引擎 74752 B）/ 0 / 20-0 / CLEAN ✓ |

新增的常驻诊断开关：`--attn-debug 1`（阶段标记：`[step] enter/fwd done/exit`）、`--attn-debug 2`（逐窗口 RMS + 指针）。

## 对照 CMD（修好后可以直接跑长训练）

```cmd
cd <repo> && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 4096 --k0 32 --k0-embed 128 --k0-head 128 --k0-attn 32 --layers 32 --no-residual 1 --attn-every 4 --attn-sparse 1 --attn-src-shared 1 --attn-calib 1 --attn-calib-target qk --head-skip 1 --seq 64 --batch 32 --seed 1 --steps 10000 --lr 3e-4 --warmup 200 --min-lr 3e-5 --clip 1 --data-dir artifacts/b1_char --vocab-mode full --chars-per-token 1.0 --attn-window 1 --threads 8 --pin 1 --parallel-mode inter --persistent-threads 1 --out-dir artifacts/iop_c3 --eval-every 200 --log-every 50 --plateau-patience 1 --save-model artifacts/iop_c3.cpk
:: 诊断：加 --attn-debug 1 看阶段、2 看逐窗口 RMS（会明显变慢，仅排查用）
:: 若怀疑越界类问题：--parallel-mode intra 作对照（这条路径一直稳定）
```

## 教训 / 纪律

1. **只在"小几何"上验证是不够的**：这个越界在 1M 配置下被 `max_wlen` 恰好盖住，只有 C3 几何才暴露（`nth*m > max_wlen`）。
   以后 per-slot 暂存一律按"最坏使用量"给（`nth*m` 而不是 `m`）。
2. 本次没有改任何稀疏层内核、也没改 intra 路径（哈希逐位不变）、没有跑长训练（最长 1 步）。
3. 线程池与本次崩溃**无关**（修前 pool/spawn 都崩），所以池子保留默认开启。
