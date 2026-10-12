# 任务一/二完成：inter+window 根因修复 + 常驻线程池

## 任务一：inter + window 1 的 slot0 全零 —— 根因与修复

### 根因（探针直接抓到，不是猜测）

```
[fw] l=3 b=0 w=0/8 off=0 tt=512 slot=0 base=0    RMS dest=2.574e-2 aq=1.080e-3 ak=1.072e-3   ← 正常
[fw] l=3 b=0 w=1/8 off=0 tt=512 slot=0 base=98304 RMS dest=2.612e-2 aq=1.093e-3 ak=1.090e-3   ← 正常
[fw] l=3 b=0 w=7/8 off=0 tt=512 slot=0 base=688128 RMS dest=0.000e0 aq=0.000e0 ak=0.000e0     ← 全零！
```

**`dest`（attention 的输入）在最后一个窗口就是 0** ⇒ 不是"读错位置"，是**前向只算了一半的样本**。

body 里"按 region 的样本循环"我写成了 `for b in 0..sb`。而 `sb = batch / nslot`（region 的样本数）。
训练区域的调用 `tt = sb*seq` ⇒ `tt/seq == sb` ✓ 没问题；**但标定/体检/评估用的是
`step_fn(0, t_tokens, 0, …)`（整批 8 个样本）**，`sb` 只有 4 ⇒ **只有前 4 个样本被嵌入/算 CE，
后 4 个样本的 `h_in` 一直是 0** ⇒ 传播到最后一层 ⇒ 最后一个窗口的 q/k 全零 ⇒ 标定读到 0
⇒ `FATAL: 块0 的 q·k 恒为 0`。

**修法**（1 行语义）：
```rust
let n_samp = tt / seq;                 // 样本数由 token 数推出，不用 sb
for b in 0..n_samp { … }               // 4 处循环
```
（`n_win` 之前已同样修过：`tt / seq`。）

### 验收（任务一）

| 验收项 | 结果 |
|---|---|
| `inter --attn-window 1 --batch 8 --threads 2 --steps 10` | **exit 0** ✓；`[probe] slot0 aq/ak RMS: 前64行=9.929e-4/9.979e-4` **非零** ✓ |
| `inter --attn-window 1 --batch 32 --threads 4` | **exit 0** ✓；集中度 1.064、非零占比 1.0000 ✓ |
| 与 `intra + window 1` 数值一致 | `final_train` 8.305950 vs 8.305963 ⇒ **Δ=1.3e-5** ✓（≤ε）|
| 确定性 | 同配置跑两次 `loss_hash=0xbab3387aa8428fe9` **相同** ✓ |
| intra 逐位不变 | `0xc9c987c15d75d1d6` ✓（`--parallel-mode intra --log-every 10`）|

## 任务二：常驻线程池（`--persistent-threads 1`，默认开）

实现（最小改动，`std` 即可）：
* `init_pool(nth, pin)`：训练开始前创建 `nth` 个常驻 worker（带名字、可选绑核），`OnceLock` 全局持有。
* `Pool::run(n, f)`：主线程把**类型擦除后的任务指针**（fat pointer 拆成 `(data, vtable)`，避开 `'static` 约束）
  放进 `Mutex<PoolState>` + `Condvar`，`notify_all` 唤醒；worker 按 **各自 idx 分片**执行后 `served += 1`；
  主线程阻塞在 `cv_done` 直到全部完成（屏障）。
* **分片规则与 spawn 路径逐字相同**（`chunk = ceil(n/nt)`，worker i 取 `[i*chunk, min((i+1)*chunk, n))`）
  ⇒ 数值应当逐位相同。
* `IN_WORKER` thread-local 防**嵌套**（worker 内部再 par_for 会死锁）⇒ 这种情形自动回落 spawn 路径。
* `shutdown_pool()` 在建结果前调用：worker 优雅退出。
* `--persistent-threads 0` 回到旧的"每次 spawn"路径（fallback 保留）。

### 验收（任务二）

| 配置 | pool | spawn | 是否逐位相同 |
|---|---|---|---|
| batch 8, threads 4, inter+window | `0xb5957ca1fc7b11f6` | `0xb5957ca1fc7b11f6` | ✅ |
| batch 8, threads 4, intra+window | `0x3dff4da1857fc065` | `0x3dff4da1857fc065` | ✅ |
| batch 32, threads 4, inter+window | `0xdc3bd56a868b92ef` | `0xdc3bd56a868b92ef` | ✅ |
| batch 32, threads 4, intra+window | `0x1309e0cda1d92978` | `0x1309e0cda1d92978` | ✅ |

**速度（实测，5~10 步）**

| 配置 | pool | spawn | 提升 |
|---|---|---|---|
| batch 8, threads 4, inter | 4.0 s | 4.2 s | ~5% |
| batch 32, threads 4, inter | **9.8 s** | 10.3 s | ~5% |
| batch 32, threads 4, intra | 9.2 s | 10.2 s | ~10% |

## 任务三：组合验证 + 一个重要更正

| 配置（batch 32, 4 线程, window=1, 5 步） | 墙钟 | 每步 | loss |
|---|---|---|---|
| intra + spawn | 10.2 s | 2.04 s | 8.309734 |
| intra + pool | **9.2 s** | 1.84 s | 8.309734 |
| inter + spawn | 10.3 s | 2.06 s | 8.309722 |
| inter + pool | 9.8 s | 1.96 s | 8.309722 |

**更正（必须说）**：我上一轮报的 "inter@4 batch32 = 16.0s/10步 = 1.9× 加速" 是在**有 bug 的代码**上测的——
那时只算了一半样本（超快），不是真加速。**修好之后**正确的数字是 **~1.96 s/step**，
而 `intra+pool` 是 **1.84 s/step** ⇒ **在这台 4 核机器上 intra 比 inter 还快 ~6%**。

**线程池只拿到 ~5%，不是 1.5-2×**。原因不是线程创建开销（那是每步 ~0.1 s，池子已全部省掉），
而是**内存带宽/共享工作集争用**：4 个**独立进程**（各自一套工作集）实测 2.88×，
而 4 个线程共享同一套权重与激活缓冲只有 ~1.8-1.9×。**要再往上就得减少每步的内存流量**
（例如更大 batch 摊薄、或按 region 复制权重以换带宽——后者会吃内存），不是线程管理问题。

## 对照 CMD（给你手动跑长训练）

```cmd
:: ① inter + window 1 + 常驻线程池（batch 32, 4 线程）
cd <repo> && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 32 --k0-head 32 --k0-attn 8 --layers 24 --no-residual 1 --attn-every 4 --attn-sparse 1 --attn-src-shared 1 --attn-calib 1 --attn-calib-target qk --head-skip 1 --seq 64 --batch 32 --seed 1 --steps 10000 --lr 3e-4 --warmup 200 --min-lr 3e-5 --data-dir artifacts/b1_char --vocab-mode full --chars-per-token 1.0 --attn-window 1 --threads 4 --parallel-mode inter --persistent-threads 1 --out-dir artifacts/iop_b32 --eval-every 200 --log-every 50 --plateau-patience 1 --save-model artifacts/iop_b32.cpk
:: ② 同配置的 baseline：把 --parallel-mode inter 换成 intra、--out-dir 换 iop_b32_intra
:: ③ 关掉线程池对照：追加 --persistent-threads 0
:: 诊断：加 --attn-debug 1 会打印每层/每窗口的 attention RMS（排查用，默认关）
```
**注意**：比较 `loss_hash` 必须固定 `--log-every`（它哈希的是被记录的步）。

## 纪律核对

| 纪律 | 结果 |
|---|---|
| 不改稀疏层内核 | ✅ 引擎 core size **74752 B** 不变 |
| intra 路径逐位不变 | ✅ `0xc9c987c15d75d1d6` |
| 保留 spawn 作 fallback | ✅ `--persistent-threads 0` |
| 每步 bit-exact 或报告差异 | ✅ pool vs spawn 逐位相同；inter vs intra 差 1.2e-5（已报告）|
| 套件 | ✅ **65/65**、`build_all` OK、`tests 0`、`b1-verify 20/0`、`ci_path_guard CLEAN` |
| 不跑长训练 | ✅ 最长 10 步 |

## 诚实边界

1. **期望收益未达成**：池子只给 ~5%（不是 1.5-2×）；inter 在这台机器上**没有**比 intra 快。
2. 常驻池的**分片固定**（worker i 永远拿第 i 片）⇒ 负载不均时会有长尾；本配置下各 region 等长，问题不显。
3. `--attn-debug` 探针默认关（开了会明显拖慢，仅诊断用）。
4. 未做 8 线程/更大 batch 的扫描；未做长时间训练。
