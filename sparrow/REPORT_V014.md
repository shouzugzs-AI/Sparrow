# Sparrow V0.1.4 —— 路径 2 后端（`.spar → Rust AOT 源`）交付报告

> 交付物：`D:\czsai\sparrow`
> 测试：**81 项全绿**（26 单元 + 15 前端验收 + 4 fork + 6 lib + 6 注意力 + 8 循环化 + 7 汇入 + 9 后端回归）
> 端到端驱动（**8 个**，全部带 can-fail）：
> `e2e_tiny` · `loop_ab` · `residual_ab` · `slot_reuse` · `bf16_ab` · `attn_ab` · **`lib_ab`** · `perf_frontend`
> 已知陷阱：`docs/TRAPS.md` —— **47 条**（本轮新增 G 组 5 条）

| 目标里的五项 | 状态 |
|---|---|
| ① 汇入 + 重复组内部（撤 `E106`） | ✅ 见 `REPORT_V013.md` §1 |
| ② 槽位复用（激活区 −42%，输出逐字符相同） | ✅ 见 `REPORT_V013.md` §2 |
| ③ 性能（前端 18×、bf16 减半、激活区 −42%） | ✅ 见 `REPORT_V013.md` §3 |
| ④ 注意力块 + 序列进 IR | ✅ 见 `REPORT_V013.md` §4 |
| ⑤ **`--backend lib`（.spar → Rust AOT 源）** | ✅ **本报告** |

---

## 1. 怎么用

```powershell
sparrow model.sparrow --backend lib --lang rust `
    --vmlm-root D:\czsai\vmlm `
    --consts-dir examples\e2e_bins `
    -o out\lib_e2e
# 生成 out/lib_e2e/{Cargo.toml, src/main.rs}
cargo run --release --manifest-path out\lib_e2e\Cargo.toml
```

生成的 `src/main.rs`：**一个层一个 `struct` + `impl`**，权重用 `include_bytes!`
内嵌（绝对路径），`Model::forward(&self, input: &[f32]) -> Vec<f32>` 按拓扑序
把各层串起来，`main` 跑一遍并用**引擎那个打印函数**输出。

---

## 2. 判据：与 `.vmb` 后端**逐字符一致** —— 以及为什么必须"同源"

朴素做法（生成 `for` 循环算 GEMM）**做不到**逐字符一致：

* 引擎的 `gemm_nt` 是 **AVX2 + FMA + 8×8 分块**；bf16 又是另一套**双累加器**内核；
* 我在 V0.1.2/V0.1.3 里实测过：纯 numpy 参考（四舍五入**和** longdouble-FMA 模拟）
  都差 1 ULP —— 而这 1 ULP 会翻掉打印的第 6 位小数。

所以生成的 Rust **直接 `use vmlm_ops::kernels::*`，调同一个函数**：

| Sparrow 层 | 生成的 Rust 调什么 | 为什么可逐位一致 |
|---|---|---|
| 线性 | `gemm_nt` / `gemm_nt_bf16` + `BIAS` 标量循环 | 同一个内核函数；每元素只加一次 |
| 归一化 | `layernorm`（eps = 引擎默认 1e-5） | 同一个函数 |
| 激活 | `act_avx2`（引擎用的就是它） | 同一个函数 |
| 稀疏 | 标量 CSR 累加 | `interp.rs` 本来**就是**标量循环（`acc = term` 首项 / `acc += term`） |
| 嵌入 | 行拷贝 / bf16 逐元素加宽 | 纯搬运，无算术 |
| 注意力 | `gemm_nt`（QKV/Wo）+ `gemm`(NN)（QK^T、·V）+ `softmax`(causal) | 同两个内核 + 同一条 causal 语义 |
| 汇入 | 逐元素加 / 列拼接（左折叠） | 与 `.vml` 生成的 `EWADD acc, acc, p` 同序 |
| 打印 | `vmlm_ops::interp::print_f32` | 引擎那个**截断**式打印（陷阱 A7） |

代价要写清楚：**生成的产物不是零依赖的**（它依赖 `vmlm-ops`）。换来的是
"逐字符一致"这个可判定判据 —— 另一条路（接受 ULP 容差）我没有选，
因为容差比较会放过真实的算子错误。

---

## 3. 验收（`tools/lib_ab.ps1`）

| 模型 | 覆盖 | `.vmb` 输出 == Rust 输出 |
|---|---|---|
| `e2e_tiny` | 嵌入 + 稀疏 + 线性 | ✅ `0.975111 1.749076 -0.720557 -0.854350` |
| `residual_add` | 2 前驱汇入（相加） | ✅ `0.143211 1.263532 -1.372653 -0.160191` |
| `residual_three` | 3 前驱汇入（左折叠） | ✅ `4.502551 1.331908 0.514215 0.972272` |
| `residual_dup` | 逐元素尾巴 | ✅ `1.360756 0.000000 0.000000 0.926220` |
| `attn_t1` | 注意力（2 头） | ✅ `-15.786533 9.903620 11.872891 2.825854` |
| `attn_causal` | 注意力 + 序列 4 + 因果掩码 | ✅ 16 个数**逐字符**相同 |

> 这 6 条是**逐字符**比较（不是容差）：同权重、同输入、同内核 ⇒ 不允许有任何差异。
> 其中 `attn_t1`/`attn_causal` 还顺带证明了注意力在两条后端上用的是同一套拆头/转置/softmax 语义。

can-fail 与 hard-fail：

| 判据 | 结果 |
|---|---|
| 把 `LINEAR_W[0]` 加 1.0 ⇒ Rust 输出必须变化 | ✅ `0.975111…` → `1.624559…`（判据非恒真） |
| 循环化的 `重复` 组 ⇒ `E109` 明确拒绝 | ✅ 报清楚理由（Rust 版要把每份实例常量摊平成数组 + 生成 `for`，排在下一轮） |
| 缺 `--vmlm-root` ⇒ `E109` | ✅ 并解释"只有用同一套内核才能逐位一致" |
| `--lang python` ⇒ 用法错误 | ✅ 只有 `rust` |

---

## 4. 工程细节（都踩过，已进 `docs/TRAPS.md` G 组）

1. **生成的 crate 常常落在宿主仓库目录里**（比如 `out/`）⇒ cargo 会以为它属于上层
   workspace 并报错。`Cargo.toml` 里必须有空的 `[workspace]` 声明"我是独立包"。
2. **别把 build warning 混进被比较的字符串**：第一版用 `cargo run --quiet 2>&1` 抓输出，
   于是 `warning: unused doc comment` 全被当成"程序的输出"，6 项全部假失败。
   现在先 `cargo build`（丢弃输出），再**直接跑产物**。
3. `EMBED` 在 vmlgen 里的 `in_dim` 是 **0**（它的输入是词元 id，不是激活向量）⇒
   生成代码里的输入宽度断言要按 1 算，否则 `assert_eq!(x.len(), rows*0)` 立刻 panic。
4. `slots["x"]`（`HashMap` 索引）在表达式位置是**值语义** ⇒ 聚合调用要显式
   `&slots["x"]`，否则 `merge_add(&[slots[a], slots[b]], ..)` 试图移动。
5. 为了让 Rust 后端与文本后端**共享同一张图**，`vmlgen` 导出了 `Plan`
   （层/拓扑序/维度/常量/汇入方式/槽基名）。让 Rust 后端去 parse `.vml` 会立刻
   产生两份真相；让它自己重算一遍图更糟（漂移迟早发生）。

---

## 5. 遗留（明确不假装做完）

1. **Rust 后端不支持循环化的 `重复` 组**（`E109`）：`.vml` 那条路用
   `LOOP_CNT` + `LAYERB` 遍历常量池；Rust 版要把每份实例的常量摊平成数组
   并生成 `for` 循环（`include_bytes!` 每份一个 static）。**去掉 `--loop-repeat`
   就能生成**（展开成 N 份），所以这不是死角。
2. 生成代码不做 `backward()`（任务书里提到 `fn forward()/fn backward()`；
   反向需要 vmlm 的 autograd/train 链路，属于另一条线 —— 与 `loss_hash` 同一个边界）。
3. 目前只支持 f32/bf16 权重；int8 GEMM 的 per-row scale 语义没接。
4. 性能对比（AOT vs 解释器）没测：这轮只保证"逐字符一致"这条**正确性**判据；
   要声称提速得单独做基准（`vmlm` 侧有自己的 bench 设施）。

---

## 6. 复现

```powershell
cd D:\czsai\sparrow
cargo test                                   # 81 项
pwsh -File tools\lib_ab.ps1                  # 生成 → cargo build → 跑产物 → 逐字符比对
# 单看一个模型：
.\target\debug\sparrow.exe examples\e2e_tiny.sparrow --backend lib --lang rust `
    --vmlm-root D:\czsai\vmlm --consts-dir examples\e2e_bins -o out\lib_e2e
cargo run --release --manifest-path out\lib_e2e\Cargo.toml
```
