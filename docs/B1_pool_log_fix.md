# 修：「常驻线程池」日志被刷屏

## 根因

我上一轮把这段代码插错了位置：脚本用的锚点是 `    if profile {`，而它在文件里**第一次出现的位置在 `step_fn` 体内**
（稀疏 lm_head 那一段的耗时统计）⇒ 于是
```rust
if persistent_threads { init_pool(nth, pin); println!("常驻线程池: …"); }
```
被放进了**热路径**，每次 `step_fn` 调用都会执行一遍 ⇒ 训练循环里刷屏（而且 `init_pool` 还会在 pool worker 线程里被反复调用）。

## 修法

1. **移到顶层一次性位置**：放在 `let t_tokens = batch * seq;` 之前（任何 `par_for` 使用之前、模型分配之前），
   即 `run_main` 的顶层流程里，不在任何循环/闭包里。
2. **加 `AtomicBool` 兜底**：`static POOL_MSG: AtomicBool`，用 `swap(true, Relaxed)` 保证 **只打印一次**（即使将来被误放回可重入路径也不会刷屏）。
3. **审计热路径里其它"只该打印一次"的消息**：
   逐行扫描 `step_fn` 体内（779 行）的 `println!`/`print!`，共 5 处，**全部带门控**：
   | 行 | 门控 |
   |---|---|
   | 2474 `[step] enter` | `if attn_debug >= 1` |
   | 2594 / 2598 `[fw] …`（两行）| `if attn_debug >= 2 && …` |
   | 2739 `[step] fwd done` | `if attn_debug >= 1` |
   | 2916 `[step] exit` | `if attn_debug >= 1` |
   其余一次性日志（`parallel-mode`、`字符表可编码`、`[probe]`、逐块尺度诊断、体检、eval-batch health）都在**顶层顺序流程**里 ✓ 不重复。
   `--attn-debug` 默认 0 ⇒ 默认零热路径输出。

## 验收（实测）

| 检查 | 结果 |
|---|---|
| 5 步训练（`--log-every 1`，inter+window1+pool，batch 32/4 线程） | 「常驻线程池」出现 **1 次** ✓，之后 step/pval/loss 行正常可见 ✓ |
| 同配置 `--persistent-threads 1` vs `0` | `loss_hash=0xab282c40d233676e` **两边逐位相同** ✓（池子不影响数值）|
| C3/10M 几何崩溃回归（inter+window1+pool） | **exit 0** ✓ 池子消息 1 次 ✓ |
| 套件 | **65/65** ✓ |

日志（修后）：
```
parallel-mode: inter (nslot=4, 每 slot 8 样本 / 512 token, 体内并行度=1)
常驻线程池: 4 个 worker（--persistent-threads 0 回到每次 spawn）   ← 只此一次
--- ATTENTION 逐块尺度诊断（view=64 token=单窗口，共 32 个窗口, dim=1536, 6 块; parallel-mode=inter）---
eval-batch health: 输入 UNK 10/8192 (0.122%) | 目标 UNK 10/8064 (0.124%)
[step     5/5] lr=3.000e-5 train=8.3097 val=8.3079 ppl=4039.99 wall=10.3s
final_train=8.309685 final_val=8.307888 best_val=8.307888 @step 5
loss_hash=0x73bedc2bcbee1997
```

## 教训

用**文本锚点**（`if profile {` 这种在文件里出现多次的串）做插入时，必须带**唯一上下文**或先校验唯一性；
这次两次踩同一类坑（之前 `t_tokens` 的批量替换也波及了不该替换的地方）。已在本轮加了注释提醒。
