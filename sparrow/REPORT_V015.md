# Sparrow V0.1.5 —— 逐项收尾报告（进行中）

> **V0.1.6 追加**：§8（发布前加固：行尾 / 可移植性 / 文档）与 §9 起（注意力每头暂存槽复用）
> 是 V0.1.6 这一轮新增的内容；§1–§7 仍是 V0.1.5 的原始记录，数字以各节写作时为准。

> 交付物：`D:\czsai\sparrow`（本版起纳入 git，基线提交 `34e6904`）
>
> 用户本轮的指令：「你一个一个做吧，描述语言可以作为附加仓库或者与 vmlm 合并」。
> 因此本轮**逐项推进、逐项验收**，每项都要求：**独立参考 + can-fail + 复现命令**，
> 不留未验收改动。进度表如下，每完成一项在此追加一节。
>
> **关于本文里的数字**：本文是**逐项追加的版本记录**，各节数字以该节写作时为准
> （例如 §1 的「8 个驱动」是那一项完成时的数量，定稿时是 10 个；§1.5 记的陷阱条数当时是 50，
> 现在 `docs/TRAPS.md` 是 67 条）。**活文档**（`docs/TRAPS.md` / `docs/REPO.md` /
> `tools/verify_all.ps1`）里的数字已与本版同步。

| # | 项 | 状态 | 关键判据 |
|---|---|---|---|
| ① | 版本化与仓库边界 | ✅ 完成 | 本条完成时 8 驱动 + 81 测试全绿（定稿时 **10 驱动 + 85 测试**），且**跑完工作区仍干净**；边界探针 4 条判据（含 can-fail） |
| ② | lib 后端 vs 解释器基准 | ✅ 完成 | 进程内同口径（同格式 `avg us/iter`）；AOT 每层快 3.3%~7.6%；可分辨/非退化/can-fail/隔离 4 类判据全 PASS |
| ③ | lib 后端支持循环化 `重复` 组 | ✅ 完成 | 4 个循环模型 `.vmb` vs Rust **逐字符**一致；两循环组串联的「循环 == 展开」；源层 `重复` 从 panic 改成 `E110` |
| ④ | 槽位复用扩展（暂存槽） | ✅ 完成 | 12 层残差激活区 1664→**256 B**（V0.1.3 只到 960 B）；循环组也能省；结构判据直接数槽名 12→1 |
| ⑤ | 序列 `T>1` 泛化（嵌入链首） | ✅ 完成 | 逐行独立性：T=3 的每一行 == 该词元的 T=1 输出（**逐字符**）；numpy 参考最大差 0；can-fail「只改第 2 个词元 ⇒ 只有第 2 行变」；Rust == `.vmb`；**顺带修掉 T>1+循环的静默错答案** |
| ⑥ | int8 权重（per-row scale） | ✅ 完成 | k=1 **逐字符相同**；k>1 差值 0（界内）；can-fail 动一行 scale 差 0.607；朴素量化 0.0098 对照；`--asm`/`--backend vml` + int8 ⇒ `E109` |
| ⑦ | `backward()` / 训练路径 | ⚠️ **勘察完毕，未落地**（按任务书"不能落地就给可判定边界"）：边界现在是**可判定**的 —— `损失`/`反向`/`训练` 一律报 `E111` 并列出引擎已有的 10 个反向/优化算子 |

---

# §1 版本化与仓库边界 ✅

## 1.1 做了什么

| 动作 | 结果 |
|---|---|
| 描述语言纳入 git | `D:\czsai\sparrow\.git`，基线提交 `34e6904`，**296 个文件**入库 |
| 行尾策略 | `.gitattributes`：`* text=auto eol=lf` + 二进制清单（照上游 `vmlmd` 的实测结论，见其注释） |
| 忽略 | `.gitignore`：`/target/`、`/out/`、`*.log` |
| 清掉未验证残留 | `tools/_patch_bf16.py`（一次性补丁脚本，自身文档写明「跑完即删」）+ 根目录 11 个手工运行现场文件（`out_cn_*.vml`、`t__1_*.bin`、`ATTN_scale.bin`、误重定向文件 `-`）→ 移到 `out/_debris/`（**没有删除**，`out/` 不入库） |
| 一条命令验收 | `tools/verify_all.ps1`：`cargo test` + 8 个驱动 + 「跑完工作区干净」不变量 |
| 边界可测化 | `tools/repo_boundary_probe.ps1` + `docs/REPO.md`（决策、代价、迁移命令、未验证项） |

## 1.2 判据与实测

**判据 1（全量验收）**：`pwsh -File tools/verify_all.ps1` → **退出码 0**（定稿前最后一次运行）

```
G0 跑之前 git status 为空  : PASS  entries=0
T1 cargo test 全绿        : PASS  10 个 target，85 passed / 0 failed（定稿时）
D-e2e_tiny / loop_ab / residual_ab / slot_reuse / bf16_ab / attn_ab / lib_ab / perf_frontend
                          : PASS  8/8 退出码 0（lib_ab 19.6 s，其余均 ≤1.1 s）
G1 跑完之后 git status 为空 : PASS  entries=0   ← 见 1.3
全部 PASS（cargo test + 8 驱动 + 工作区干净）
```

> 首次运行时 G1 报 `entries=1`（`?? tools/verify_all.ps1` 自身还没提交，见 1.3），
> 提交后 G0/G1 均 PASS。

**判据 2（仓库边界，含 can-fail）**：`pwsh -File tools/repo_boundary_probe.ps1` → 全绿

| 判据 | 实测 |
|---|---|
| 1a 父仓库包列表不含 sparrow | PASS `packages=stub-sys,…,stub-server`（13 个占位 crate，members 表逐字照抄 vmlm） |
| 1b 父仓库 `cargo build --release` 成功 | PASS `exit=0`（带 vmlm 的 `panic=abort`/`lto` profile） |
| 2 嵌套 `sparrow\cargo test` **与真实仓库完全一致** | PASS：嵌套 85 passed / 0 failed = 真实仓库 85 / 0（判据**不写死数字** —— 写死过一次 81，后来加到 85 就假失败） |
| 3 **can-fail**：去掉拷贝里的 `[workspace]` 行 | PASS 必须失败：`exit=101`，`error: current package believes it's in a workspace when it's not:` |

判据 3 是判据 2 的支点：没有它，「嵌套也能构建」可能只是「怎么写都能过」。

## 1.3 为什么把**生成物**入库（以及由此换来的一条强不变量）

`examples/` 下的 `.spar`（IR）/`.vml`（opcode）/`.vmb`/`.args` 与 `*_bins/` 常量**全部是生成物**
（由 `tools/gen_*.py` 与 `sparrow.exe` 产出），本仓库仍然**入库**（只排除 `target/`、`out/`）。

于是 `tools/verify_all.ps1` 能在跑完 8 个驱动之后断言 **`git status --porcelain` 为空**。
这一条同时证明两件事，且**都可假**：

1. 生成器是**确定性的**：同 seed 同字节（包括 24 层循环组、残差组、注意力组、bf16 组）；
2. 验收过程**没有留下未回收的改动**（这正是本项目「不留未验收改动」纪律的机器化版本）。

第一次跑时 G1 报 `entries=1`，那一条是脚本自身还没提交（`?? tools/verify_all.ps1`）——
**不是**生成物漂移；提交后 G0/G1 均 PASS。

## 1.4 复现

```powershell
cd D:\czsai\sparrow
git log --oneline                       # 34e6904 基线 / c5a598a verify_all
git status --porcelain                  # 应为空
pwsh -NoProfile -ExecutionPolicy Bypass -File tools\verify_all.ps1           # 退出码 0
pwsh -NoProfile -ExecutionPolicy Bypass -File tools\repo_boundary_probe.ps1  # 退出码 0
```

## 1.5 边界（不藏）

1. **没有 remote**：本仓库只有本地 git；是否推到 GitHub（新仓库 / 或作为上游仓库的子目录）由你决定。上游开源树 `D:\czsai\vmlmd` **零改动**。
2. 探针里的父仓库是**忠实镜像**（members 表 + release profile 照抄，成员是占位 crate），**不是**真实 vmlm 树；真并进去以后仍要跑 `docs/REPO.md` §5 的第 3/4 步（引擎字节纪律 + 父仓库不认识语言 crate）。
3. 迁移到 `vmlm\sparrow\` 这条路**已实测可行**（不需要改父仓库 members 表），但**我没有真的迁**；当前选择是独立仓库，理由是隔离 vmlm 的构建纪律（详见 `docs/REPO.md` §3）。
4. 名称冲突已记录：上游**仓库名**就叫 `Sparrow`（`shouzugzs-AI/Sparrow`，即整套 VMLM），本仓库是**描述语言**，两者不是同一个东西。
5. 本次踩到三个新坑并写进 `docs/TRAPS.md`（47 → **50 条**）：**A9** PowerShell 单元素管道索引退化（造成一次假失败）、**H1** `git init`/`git add -A` 没钉住 CWD（在 20 GB 目录上跑，已及时中止并清理）、**H2** 生成物入库换强不变量的代价。

---

# §2 AOT（`--backend lib`）vs 解释器：可测基准 ✅

## 2.1 为什么必须是**进程内**基准（不是"跑一遍 exe 记 wall time"）

两个数量级问题：
* 进程启动（`CreateProcess` + 页错误 + 装载 2~25 MB 常量）在**毫秒**量级，而单次前向在**几百微秒 ~ 3 毫秒**量级 —— 直接比 wall time，测到的很大一块是启动；
* 引擎的**官方**二进制一次运行只前向一遍（`vmlm-runtime.exe` 固定读 `model.vmb`，默认构建连 argv 都不解析）。

所以两条路都改成进程内计时，并且**口径逐项对齐**：

| | 解释器 | AOT |
|---|---|---|
| 计时设施 | `vmlm-runtime-bench.exe <model.vmb> <reps>`（**vmlm 自带**，`runtime/src/bench.rs`） | `sparrow --backend lib --bench-iters K`（本版新增）生成的主程序 |
| 预热 | 1 遍 | 1 遍 |
| 时钟 | `QueryPerformanceCounter` / `QueryPerformanceFrequency` | `std::time::Instant` |
| 输出格式 | `avg us/iter = N`（整数 µs） | **同一行文本**（整数 µs） |
| 装载 | `Vm::load` 在计时之外 | `Model { .. }`（`include_bytes!` → `Vec`）在计时之外 |
| 打印 | **不在**计时区内（见下） | **不在**计时区内 |

打印为什么必须排除：引擎那条路的 `PRINT` 指令在每个 iter 都要输出 256 个数（实测每秒 256 次 `WriteFile`），量级与 GEMM 同阶，会把计算完全盖掉。做法是**把 `.vml` 里的 `PRINT` 行去掉再汇编**（用 `--asm` 写出的 `.args` 常量清单直接调汇编器），计时用这份 `*_noprint.vmb`；正确性仍然用带 `PRINT` 的官方 `.vmb`。

## 2.2 生成端改了什么（`--bench-iters K`）

* `src/libgen.rs` 新增 `LibOptions { bench_iters }` 与 `generate_rust_with(...)`；`generate_rust(...)` 保持原签名（`bench_iters = 0`），8 处既有调用点无需改。
* **默认路径一个字节都不变**：`bench_iters = 0` 时不生成任何计时代码。这条由单测守着 —— 把插入的计时段整块删掉，剩余文本必须与默认版**逐字节相同**（`tests/lib.rs::bench_iters_adds_a_timer_and_leaves_the_default_untouched`）。
* 计时代码只有 8 行：预热 1 遍 → `Instant::now()` → `for _ in 0..K { black_box(forward) }` → 打印同一行文本。

## 2.3 模型与判据

模型 `chain<N>`（`tools/gen_bench.py`）：`嵌入(8, D=256) → l1 … lN`（线性 256→256，**每层不同 seed**）。

| 判据 | 内容 | 实测 |
|---|---|---|
| 1 | plain 引擎输出 == AOT 输出（**逐字符**） | PASS（N=8/32/96） |
| 1b | **输出非退化**：>D/2 个互不相同的值 | PASS（256/256 个互不相同） |
| 2 | 隔离构建的 bench 引擎 `reps=1` 输出 == plain 引擎输出 | PASS（证明计时用的引擎与官方二进制同一个 VM） |
| 3 | 耗时随 N 增长且 > 3× 观测抖动 | PASS（增长 2617~2734 µs，抖动上限 113 µs） |
| 4 | can-fail：改一个权重字节 ⇒ AOT 输出必须变 | PASS（11.958627… → 11.969413…） |
| 5 | 隔离：vmlm 的 `vmlm-runtime.exe` / `Cargo.lock` SHA256 前后一致 | PASS（**本驱动不写 vmlm 任何文件**：官方引擎拷到 `out/` 里跑，bench 引擎构建到 `out/` 下的独立 `CARGO_TARGET_DIR`） |

## 2.4 数字（D=256，reps=400，每点取 5 轮最小值）

| N | 指令数 | 权重 B | `.vmb` B | AOT exe B | 解释器 µs/iter | AOT µs/iter |
|---|---|---|---|---|---|---|
| 8 | 20 | 2 113 540 | 2 114 952 | 2 271 744 | 206 (±5) | 189 (±11) |
| 32 | 68 | 8 429 572 | 8 434 344 | 8 628 224 | 837 (±79) | 772 (±40) |
| 96 | 196 | 25 272 324 | 25 286 056 | 25 579 520 | 2 940 (±113) | 2 806 (±88) |

**分段边际成本（µs/层）**：

| 段 | 权重 | 解释器 | AOT | 比（AOT/解释器） |
|---|---|---|---|---|
| N 8→32 | 8 MB | 26.29 | 24.29 | **0.924**（AOT 快 7.6%） |
| N 32→96 | 24 MB | 32.86 | 31.78 | **0.967**（AOT 快 3.3%） |

整次前向（N=96）：**2806 µs vs 2940 µs ⇒ AOT 快 4.6%**。

**结论（按数据说话，不夸大）**：
1. **AOT 只快 3~8%**，不是数量级。原因清楚：两条路调的是**同一套 AVX2+FMA 内核**，而这条负载是**内存带宽受限**的（每层要流 256 KB 权重，96 层 = 25 MB，越过 L3），执行时间由带宽决定，不由"谁来派发"决定。
2. **每层成本不是常数**：26.3（权重 8 MB，命中缓存）→ 32.9 µs/层（权重 24 MB，越过 L3）。这条也解释了为什么两参数线性拟合的**截距是负的** —— 那是"模型不合适"的信号，不是"固定开销为负"。所以本报告按**分段边际**报数，不报单一斜率。
3. **AOT 侧还有明显的自留成本**：生成代码每次前向、每层都 `Vec` 分配 + `HashMap<&str, Vec<f32>>` 查槽；解释器的槽是**装好就不再分配**的 arena。也就是说 AOT 的"省下的派发开销"被自己的分配开销吃掉了一大半。改成预分配 arena 应能明显拉开差距 —— **这一版没做**（列在 §2.7）。
4. 所以 `--backend lib` 的价值主张应写成：**产物不依赖引擎的二进制/VM**（不需要 `vmlm-runtime.exe`，可当普通 Rust 库/二进制集成）、**源码长度 O(1)**（循环组，见 §3）、**可在标准工具链里链接与调试**；"比解释器快"这条最多说到 **~5%**（本机、本模型）。
   一句要写准的话：生成的 crate **静态链接引擎的源码 crate** —— `Cargo.toml` 里是
   `vmlm-ops = { path = "<vmlm-root>/ops" }`（见 `src/libgen.rs` 的 `cargo_toml`）。
   这既是它与 `.vmb` 逐位一致的**来源**，也是**代价**：单独发一个 AOT 模型要把 `ops/` 一起带上；
   缺 `--vmlm-root` 时 `E109` 硬失败。

## 2.5 过程中被 can-fail 抓住的一个**恒真基准**（本项最有价值的一件事）

第一版 `gen_bench.py` 用未归一化的随机正态权重：随机矩阵每层增益 ≈ 谱半径 2 ⇒ 96 层后 ≈ 2^96 ⇒ 激活是 **NaN**。而引擎的 `print_f32` 是**截断**实现（`interp.rs:2416`：`av as u64`、`frac as u32` 逐位取数），`NaN as u64 = 0` ⇒ **NaN 被打印成 `0.000000`**。

后果：引擎输出与 AOT 输出**都是 `0.000000 …`**，"逐字符相同" PASS —— 但这是**恒真**的假通过。唯一发现它的是 **can-fail 判据**（"改一个权重字节输出必须变" FAIL，因为 0.000000 改不动）。

修复分两层（都要，缺一不可）：
1. **生成器**：每层权重**按行 L2 归一化**（每层近似保范，激活量级恒定），并加数值自检 —— 输出必须全有限、>D/2 个不同值、扰动 `l1_W[0]` 必须让输出变 >1e-3；
2. **驱动**：加"**输出非退化**"守卫（值个数 == D、互不相同者 >D/2），任何一次"相同"都必须同时过这一关。

教训已写进 `docs/TRAPS.md` **A10**：**截断式打印会把 NaN 洗成合法数字**，于是"逐字符相同"与 can-fail 可以同时恒真。

## 2.6 复现

```powershell
cd D:\czsai\sparrow
pwsh -NoProfile -ExecutionPolicy Bypass -File tools\aot_bench.ps1   # 退出码 0；约 3 分钟（含 bench 引擎 + 4 个生成 crate 的构建）
```

## 2.7 边界（不藏）

1. 只测了**矩阵-向量**形态（D=256、单 token）—— 这正是 VMLM 的推理形态；序列 `T>1`（矩阵-矩阵）与训练/反向**没测**。
2. AOT 侧**没做 arena 优化**（每层 `Vec` 分配），因此 §2.4 的 3~8% 是**当前生成代码**的数字，不是"Rust AOT 架构"的上限。
3. 计时分辨率是**整数微秒**（为了与引擎 bench harness 逐字符同格式）；单次前向低于 1 µs 的模型（如 `e2e_tiny`）会得到 `avg us/iter = 0`，因此基准固定用 D=256 的模型，并在判据 3 里要求"增长量 > 3× 抖动"。
4. `--bench-iters` 只对 `--backend lib` 有意义（`--backend vml` 的产物是 `.vmb`，计时由引擎侧负责）。

---

# §3 `--backend lib` 支持循环化的 `重复` 组 ✅

## 3.1 改了什么

| 位置 | 改动 |
|---|---|
| `vmlgen` | `Plan` 新增 `groups: Vec<PlanGroup>`（`insts` / `seed` / `ext` / `merge` / `in_dim` / `out_dim`）—— 文本后端与 Rust 后端**共用同一张图**，Rust 侧不重新推图 |
| `libgen` | **撤掉该路径的 `E109`**（原文写着"排在下一轮"）；抽出 `fields_of` / `forward_fn`（同一份 `forward` 代码要能挂到普通层 `L{i}` 与循环实例 `G{g}Inst`）；新增 `group_struct`（`G{g}Inst` 一份代码 + `G{g} { items }` N 份常量）与 `group_loop` |
| `libgen` | 循环控制常量（`<前缀>_iters` / `_lstride` / `_nstride`）**不内嵌**：它们是 `LOOP_CNT`+`LAYERB` 的实现细节，Rust 版直接遍历实例 |
| `vmlgen` | 顺带修一个 **panic**：`重复` 组落在**源层**（没有前驱）时原先是 `l.preds[0]` 越界崩溃 → 现在是命名诊断 **`E110`**（两条后端都覆盖） |

## 3.2 语义逐条对齐（这是"逐字符一致"能成立的原因）

| `.vmb`（`LOOP_CNT` + `LAYERB`） | 生成的 Rust |
|---|---|
| `SHAPE carry…; COPY carry, <链输入>_out` | `let mut carry: Vec<f32> = slots["<种子>_out"].to_vec();` |
| 每圈 `LAYERB +stride` ⇒ 第 k 圈读**第 k 份**常量 | `for inst in &self.g0.items` ⇒ 第 k 个实例读**第 k 份**常量 |
| 有外部前驱时每圈 `EWADD/CONCAT merge, carry, ext…` | `merge_add(&[&carry, &slots["<ext>_out"]…], rows, in_dim)` —— **同一操作数顺序**（拼接不可交换） |
| `COPY carry, <最后一份>_out` | `carry = inst.forward(&carry, rows)` |
| 循环结束写 `<最后一份>_out` | `slots.insert("<最后一份>_out", carry)` |

## 3.3 判据与实测（`tools/lib_ab.ps1`，全 PASS）

| 判据 | 实测 |
|---|---|
| 10 个模型的 `.vmb` 输出 == Rust 输出（逐字符） | PASS（6 个非循环 + **4 个循环**：`loop24` / `resloop` / `attnloop` / `loop2`） |
| `loop24`：**24 层稀疏**（权重槽号 > 63，TRAPS B9 ⇒ 展开版根本汇编不出来） | 循环版是唯一可行路径，且 Rust 与它逐字符一致：`4249.876464 -2597.539550 -2484.585449 3531.255126` |
| `loop2`：**两个循环组串联**（补 V0.1.2 起的覆盖缺口） | 循环版输出 == **展开版**输出（`17.620042 12.738257 -3.070589 -72.131530`）—— 这条把两个后端钉在**同一个函数语义**上，而不只是"两个实现互相同意" |
| 结构：源码是「一份代码 + N 份常量」 | PASS：`for inst in &self.g0.items` 存在；24/24 份权重都内嵌；24 层模型只有 **3** 份 `fn forward`；`_lstride`/`_nstride` 没进 Rust |
| can-fail：改一个权重字节 ⇒ 输出必须变 | PASS（`0.975111…` → `1.624559…`） |
| 源层 `重复` + `--loop-repeat` | PASS：`E110`（之前是 panic），展开模式不受影响 |

## 3.4 复现

```powershell
cd D:\czsai\sparrow
pwsh -NoProfile -ExecutionPolicy Bypass -File tools\lib_ab.ps1     # 退出码 0（约 10 s：10 个生成 crate 共用 CARGO_TARGET_DIR）
python tools/gen_loop_two.py                                       # loop2 的确定性权重（每份实例权重互不相同）
```

## 3.5 边界（不藏）

1. **源层不能循环化**（`E110`）：循环体需要一条链输入来播种进位槽，而源层没有"第 0 份"可搬。要做也能做（从模型输入常量播种），**本版没做**。
2. 循环组**组内**不参与槽位复用（组内槽名固定为最后一份的名字）；组间依赖走普通 slot，已用 2 组串联实测。
3. 一个层的 `重复` 组与 `--bench-iters` 同时用**没单独验收**（构建上不冲突，但没跑过就不断言）。
4. int8 权重仍未接（见 §6/⑦ 的取舍）。

---

# §4 槽位复用扩展：**暂存槽**也回收 ✅

## 4.1 改了什么

V0.1.3 的槽位复用只回收**层输出槽**（池按输出宽度分桶、存"基名"、真正槽名是 `<基名>_out`）。
本轮新增一个**暂存池**，回收那些"活到本层读完就死"的中间槽：

| 类别 | 槽名 | 形状 | 何时可还 |
|---|---|---|---|
| 汇入中间槽 | `<层名>_in` | 恒 `(1, in_dim)` | 这一层读完它之后（同一层的输出写完后） |
| 循环进位槽 | `<最后一份>__in` | `(1, in_cols)` | 整个循环体发射完之后 |
| 循环汇入槽 | `<最后一份>__m` | `(1, in_dim)` | 同上 |
| 循环计数器/步长 | `__ctr` / `__ls` / `__ns` | `(1, 1)` | 同上（`LAYERB ns` 是最后一条使用者） |

两个实现细节值得写下来：

1. **桶键是 `(行, 列)` 而不是只有列宽**。同一个槽名被再次 `SHAPE` 时形状必须**完全一致**：
   汇编器按最终形状 64 B 对齐紧密排布激活区，形状变了字节数就变，"复用只改写哪个槽、
   不改数值"这条前提就不再显然。例子：循环进位槽是 `(1, D)`，而 T>1 时层输出槽是 `(T, D)`
   —— 只看列宽会把它们当同一种槽，于是把 `(1, D)` 借给 `(T, D)` 用。
2. 汇入的 dst 由调用方给（`emit_merge` → `emit_merge_into`），这样它才能来自池；
   池里存的是**完整槽名**（`_in` / `__m` / `__ctr` 这些名字不符合"基名 + `_out`"的约定，
   所以没法并入既有那个池）。

## 4.2 判据与实测（`tools/slot_reuse.ps1`，全 PASS）

| 模型 | 激活区（不复用 → 复用） | 说明 |
|---|---|---|
| 12 层残差（展开） | 1664 B → **256 B（−84.6%）** | V0.1.3 只有 1664 → 960 B（−42%）；差额全部来自汇入中间槽 |
| 12 层残差（循环） | 512 B → 448 B（−12.5%） | **新增能力**：循环组的进位/汇入/控制槽 |
| 两个循环组（循环） | 832 B → 768 B（−7.7%） | **新增能力**：多组串联时两组互借 |
| 两个循环组（展开） | 512 B → 256 B（−50%） | |
| 最小闭环（链） | 256 B → 192 B（−25%） | |
| 逐元素尾巴 | 320 B → 256 B（−20%） | |
| 24 层稀疏（循环） | 512 B → 512 B（不增） | 组后没有汇入 ⇒ 暂存池没人来借，**不增**即正确 |
| 三分支汇入 / 两分支汇入 | 384 B → 384 B / 320 B → 320 B（不增） | 分支必须同时活着 ⇒ 省不动 |

另外三条判据：
* **输出逐字符相同**（复用开 vs 关，引擎侧比对）—— 复用只改"写哪个槽"，不改数值；
* **确定性**：同一模型复用两次，产物同长；
* **结构判据（本轮新增）**：直接数**汇入中间槽的槽名个数** —— 12 次汇入在关闭复用时有
  **12** 个不同名字，开启后**1** 个（轮着借用）。这条比"激活区变小"更强：数字变小可能是
  别的原因，而 12→1 只可能来自回收。

## 4.3 这些判据为什么不是恒真

* can-fail 本体就是那三个"分支必须同时活着"的模型：生命周期分析一旦错回收，
  汇入会把 B+B 算成别的（覆盖写），输出立刻与不复用版不同 ⇒ 判据 1 红；
* 独立参考：residual 系列的期望值由 **numpy 独立算过**（`tools/residual_ab.ps1`，逐字符）。

## 4.4 复现

```powershell
cd D:\czsai\sparrow
pwsh -NoProfile -ExecutionPolicy Bypass -File tools\slot_reuse.ps1    # 退出码 0
```

## 4.5 边界（不藏）

1. **注意力内部的 scratch 还没回收**：`q/k/v`、每头的 `qh/kh/vh`、`sc/ctx`、转置 `kT`
   等名字在 `emit_attn` 内部生成，要接池得先把"名字"变成可注入参数 —— **本版没做**。
   这也是下一个能显著省激活区的地方（注意力层数越多越明显）。
2. 暂存池按 `(行, 列)` 精确匹配 ⇒ 不同形状的槽**不能**互借（保守但可证）。
3. 激活区按 64 B 对齐：宽度 4 的激活也占 64 B，所以上表的字节数是**对齐后**的。

---

# §4 附：`--bench-iters` 与循环组**同时用**（补 §3.5 边界 3）

§3.5 当时写着"这两者同时用没单独验收"。补上了：

```powershell
sparrow examples\loop_two.sparrow --loop-repeat --backend lib --lang rust \
    --vmlm-root D:\czsai\vmlm --consts-dir examples\loop_two_bins --bench-iters 10 -o out\bench_loop\rs
```

实测：生成物编译通过；输出两行 —— 第一行（数值）与 `.vmb` 循环版**逐字符相同**
（`17.620042 12.738257 -3.070589 -72.131530`），第二行是计时
（`avg us/iter = 1`，这个模型太小，整数微秒分辨率下就是 1）。
即：计时循环包在 `Model::forward` 外面，与 `forward` 内部的循环组互不干扰。

---

# §5 `序列=T>1` 的**嵌入链首** ✅（并修掉一个静默错答案）

## 5.1 改之前是什么样

V0.1.3 起 `序列=` 就能进 IR，但链首只能是"逐行"的 `线性`：**`嵌入` 链首 + `序列>1` 直接报 `E107`**，
理由是 `EMB` 一次只查一个行号（`[1,1]` → `[1,D]`）。这挡住了最自然的用法：
**一段真实词元序列 → 注意力**。

## 5.2 现在怎么做的（逐行查表 + 拼行）

输入常量在 `序列>1` 时是**一行 T 个词元 id**（`[1,T]`，`SLICE` 一次取一个），然后：

```
每个词元 k：SHAPE tok_k,1,1          ; SLICE 取第 k 个 id
            SLICE tok_k, <输入>, k, 1
            SHAPE row_k,1,D          ; EMB 查表（一次一行）
            EMB   row_k, E, tok_k
            SHAPE c_k,D,1            ; 转成列
            TRANSPOSE c_k, row_k
            按列左折叠 CONCAT 成 [D,T]（每一步用**新槽名**：就地写会互相覆盖）
最后        TRANSPOSE <out>, <stack>  ; → [T,D]（形状由 TRANSPOSE 推断）
```

`--backend lib` 同步改成**逐行查表**（`for k in 0..rows { … }`）；`rows == 1` 时与 V0.1.4 逐字节等价。

## 5.3 判据与实测（`tools/seq_ab.ps1`，全 PASS；模型 `examples/seq_embed.sparrow` 序列=3）

| 判据 | 实测 |
|---|---|
| **逐行独立性**：`se3` 第 k 行 == 只喂第 k 个词元的 `se1` 输出（**逐字符**） | PASS（3/3 行；`-1.732369 1.788167 3.412284 2.389850` / `-0.871481 4.611680 0.400362 3.009402` / `-1.786019 -2.200529 -0.686734 -1.084218`） |
| numpy 独立参考（复刻引擎的截断式打印） | PASS，**最大差 0** |
| can-fail：把第 2 个词元 5 → 6 | PASS：第 1/3 行**逐字符不变**，只有第 2 行变 |
| Rust（`--backend lib`）== `.vmb`（逐字符） | PASS（T>1 也走同一条路） |
| 结构：链首展开成 **T 次** `EMB` | PASS（3 次） |
| 序列=3 **+ 循环组**：循环版 == 展开版（逐字符） | PASS（见 5.4） |

## 5.4 过程中抓到一个**静默错答案**（比 ⑤ 本身更重要）

写 ⑤ 之前先探了一下"`序列=3` + `--loop-repeat`"，结果是：**循环版只输出 4 个数**（应当 12 个），
**退出码 0**。根因是循环发射把进位槽的行数**写死成 1**：

```
SHAPE   t__2__in, 1, 4        ; ← 应该是 3, 4
```

于是循环体里的 `GEMM` 按 1 行算（`m` 取自输入槽形状），组的输出槽变成 `[1,4]`，
`PRINT` 自然只打一行。**这正是本项目最不能接受的一类失败**：不报错、不崩、悄悄少算。

修法：把 `t_seq` 传进 `emit_loop_group`，进位槽 / 汇入槽都按 `[T, D]` 声明
（修正后 `.vmb` 里是 `slot4 t__2__in [act] 3x4`），并把这条**做成常驻判据**
（`tools/seq_ab.ps1` 第 6 节 + `examples/seq_loop.sparrow`）：循环版 == 展开版，逐字符。

同一族还有两个必须**硬拒绝**的情形（否则同样静默只算第一行），本轮一并加了诊断 `E107`：
* `稀疏` 链首或出现在 T>1 链路里：`SPARSE_MATVEC` 是**向量×矩阵**（`[D,1]→[D,1]`），一次只吃一行；
* `嵌入` 出现在链路**中间**（输入已经是 `[T,D]` 激活，不是词元 id）。

## 5.5 复现

```powershell
cd D:\czsai\sparrow
python tools\gen_seq.py                                            # 权重 + numpy 参考
pwsh -NoProfile -ExecutionPolicy Bypass -File tools\seq_ab.ps1     # 退出码 0（12 项判据）
```

## 5.6 边界（不藏）

1. **逐行查表的代价**：T 次 `EMB` + 2T 次 `TRANSPOSE` + T−1 次 `CONCAT` —— T=3 时 13 条指令。
   序列很长时指令数线性增长（没有"批量查表"指令，引擎侧也没有）。
2. `稀疏` 在 `T>1` 下**不可用**（`E107`）：要支持得逐行做"取行 → 转列 → SPARSE → 转行"，
   机制已经在那儿（`SLICE`/`TRANSPOSE`），**本版没做**。
3. 与 `--loop-repeat` 同时用时，循环组的模板**不能**是 `嵌入`/`稀疏`（`E107`）——
   逐行算子（`线性`/`归一化`/`激活`/`注意力`）可以，已实测（§5.4）。
4. 输入常量是**一行 T 个 id**：真要做逐 token 喂入（比如从 `st_tok` 读），那是引擎侧的
   `load_ids` 路径（vmlm 的 P13 机制），语言这边没接。

---

# §6 线性层的 int8 量化权重（per-**输出行** scale）✅

## 6.1 内核事实决定判据（先把这件事说清楚）

`vmlm_ops::kernels::gemm_nt_int8(a, q: *const i8, scale: *const f32, c, m, n, k)` 已经存在
（P13 做的，LLM.int8 布局），它算的是：

```
s = Σ_k a[k] · f32(q[k])      ; 8 宽 FMA 累加 + 尾部标量
out[j] = s · scale[j]         ; 整行共用一个 scale，乘在内积**之外**
```

所以 **不能**沿用 bf16 那条"与 f32 逐字符相同"的判据 —— int8 与 f32 是**两个不同的内核**，
累加顺序不同（`gemm_nt` 是 AVX2 8×8 分块），即使权重可被 int8 精确表示，和也可能差 ~1 ULP。
判据于是分三档：

| 档 | 做法 | 为什么它能假 |
|---|---|---|
| **k=1 精确** | 单层、输入宽度 1（内积只有一项），`w = q/64`（精确可表示，且"先乘 q 再乘 2⁻⁶"与"直接乘 w"在实数上相等）⇒ 两条路必须**逐字符相同** | 行列布局、`scale` 的**行**下标、bias 位置错一个都会红 |
| **k>1 有界** | 两层线性，差值必须 ≤ 界（并**打印实测值**，不藏） | 若内核被写坏、或 scale 用错行，差值会超出界 |
| **can-fail** | 把 `a_scale[0]` 动一个 2⁻⁶ ⇒ 差值必须**远超**上面的界（≥100×） | 保证"差值小"不是"根本没用到 scale" |

## 6.2 实现

* `--weight-dtype int8`（新值）：只作用于**线性层权重**；嵌入表回落 f32（`EMB` 的表不吃 int8）。
* 常量是**两个**文件：`<层>_Wq`（i8，1 B/元素）+ `<层>_scale`（f32，每输出行一个）。
  量化本身是**输入**（和 f32/bf16 一样，`.bin` 由使用者提供），编译器不做量化。
* 生成的 Rust：`wq: Vec<i8>` / `scale: Vec<f32>`，`i8s()` 只做字节解释，前向调
  `kernels::gemm_nt_int8(...)`；**反量化在核内**。
* `.vmb`/汇编器那条路**没有**反量化 GEMM 指令 ⇒ `--weight-dtype int8` 配 `--asm` 或
  `--backend vml` 一律 `E109` **说清理由**（而不是产出一份"看起来能汇编、其实把 int8 当 f32 读"的产物）。

## 6.3 判据与实测（`tools/int8_ab.ps1`，全 PASS）

| 判据 | 实测 |
|---|---|
| 1 k=1：f32 vs int8 **逐字符** | PASS `0.881869 -0.907158 -0.010653 2.102990`（两边一字不差） |
| 2 k=4：差值 ≤ 1e-4 | PASS，**实测最大差 = 0** |
| 3 can-fail：动一行 scale | PASS：**0.607249**（是第 2 条的 ∞ 倍：0 → 0.607） |
| 4 朴素量化对照（`scale = max|row|/127`） | PASS：**0.009791** ≫ 精确集的 0 ⇒ 判据测的是量化损失，不是噪声 |
| 5 `--asm` / `--backend vml` + int8 | PASS：`E109` 两条路径都报清楚 |
| 6 体积比 == 公式 `4/(1+4/K)` | PASS：K=4 时 128 B → 64 B（**2.0×**，与公式一致；K=256 时同公式给 3.94×） |

体积那条特意**不拍一个"≥3×"**：int8 的常量除了 q（1 B/元素）还有每**输出行**一个 f32 scale
（4 B/行），玩具尺寸（K=4）下 scale 占了 1/4 ⇒ 比值只有 2×。判据改成"实测 == 公式"，
并在报告里给出 K=256 时的 3.94×。

## 6.4 复现

```powershell
cd D:\czsai\sparrow
python tools\gen_int8.py                                          # 三个素材集（精确/朴素/can-fail）
pwsh -NoProfile -ExecutionPolicy Bypass -File tools\int8_ab.ps1    # 退出码 0
cargo test --test lib                                              # 结构单测（gemm_nt_int8 / i8s / 两个常量）
```

## 6.5 边界（不藏）

1. **只有"GEMM 形态"的权重**吃 int8：线性层（V0.1.5）与**注意力投影**（V0.1.6，见 §11）；
   嵌入表、归一化、稀疏仍是 f32/bf16（`gemm_nt_int8` 的布局是给 GEMM 的；`EMB`/CSR 没有对应内核）。
2. **没有实测"int8 更快"**：vmlm 自己的 P13 记录是"内核级 3.4×（vs f32），因为 GEMM 受权重字节流量限制"，
   但同一份记录也写着真实模型上 `rel = 2.89e-02`、**没达到它自己 <1e-2 的判据**。
   本仓库这轮只验**接入正确性**，不重复那些数字（要复现得用 vmlm 的 `tools/p13_quant_eval.py`）。
3. 量化本身不在编译器里：`.bin` 由使用者提供。所以"量化误差多大"取决于素材，
   本轮用"精确可表示"和"朴素 max|row|/127"两套素材把这个边界摆出来。
4. `.vmb` 后端**永远**不支持 int8（引擎没有反量化指令）。这是产品边界，不是没做完。

---

# §7 `backward()` / 训练路径：勘察结论 + **可判定边界** ⚠️

任务书对这一项的要求是"先勘察 vmlm autograd/train 现状，再决定能否落地；不能落地就给出
可判定的边界与理由"。下面是勘察结果、决定，以及**已经落地的那部分**。

## 7.1 勘察：引擎侧的"反向"是**有的**，缺的是语言的表达层

| 看什么 | 实测 |
|---|---|
| 引擎的反向/优化算子 | `core/src/interp.rs` 里有 **10+ 个**：`OP_MSE_BWD` / `OP_LOSS_ADD` / `OP_CLIP_GRAD` / `OP_OUTER_BWD` / `OP_EMB_BWD` / `OP_LN_BWD` / `OP_SOFTMAX_BWD` / `OP_ZERO` / `OP_ADAM` / `OP_SGD` / `OP_ADAMW` |
| 谁在驱动训练 | vmlm 的 `train/` crate 里是**手写的 Rust 驱动**（`b1_full_train.rs`、`attn_train.rs`、`rn n_train`…），产出 `vmlm-train.exe` / `vmlm-b1-full.exe` / `vmlm-attn-train.exe` 等 |
| 语言侧有什么 | `.sparrow` 只有 `模型` / 层实例化 / `连接` / `前向` —— **没有**损失声明、没有反向遍历、没有参数更新 |

所以缺的**不是算子**，而是：
1. 语言要有"损失"这个概念（对什么求导、与谁比较）；
2. 要按**拓扑反序**发射反向算子（引擎约定：梯度走 `g_` 前缀槽、参数更新走 `OP_ADAM/SGD`）；
3. 要有"训练循环"的语义（步数、学习率、数据从哪来）。

## 7.2 决定：本轮**不落地**，但把边界做成**可判定**的

理由（可检验，不是"没时间"）：
* **没有独立参考**。vmlm 的训练指令流是手写 Rust 驱动的产物，不是某个 IR 编译器的输出；
  因此"我生成的 backward 与谁逐位一致"没有现成对照物 —— 这与 ①–⑥ 那几项**性质不同**
  （那些都有独立参考或可判定的等价关系）。硬上一版只能是"看起来像反传"。
* 语言层要先做**设计决策**（损失怎么声明、是否支持多损失、优化器怎么选），
  这些不是"补 codegen"能绕过的。

**已经落地的边界**（可判定、可复现）：
1. `损失` / `反向` / `训练`（以及 `loss`/`backward`/`grad`/`train`/`optimizer` 等）出现时，
   一律报 **`E111`**，并说明"本语言还没有损失与反向陈述"，提示里**列出引擎已有的那些算子**。
   之前的行为是 `反向` → `E002 未定义的层类型`、`损失 均方` → `E006 引用了未定义的层`
   —— 两条都在暗示"你打错了"，属于**误导性诊断**。
2. 不误伤：真打错的层类型（`线形`）仍是 `E002` + 建议；把这些词当**实例名**（`训练 = 线性 …`）
   照常工作。三条都由 `tests/acceptance.rs::training_constructs_are_a_named_boundary` 守着。

## 7.3 若将来要落地：最小可判定切片（具体到判据）

1. 语言加 `损失 均方`（或 `损失 交叉熵`）+ `反向`；
2. 生成端按拓扑反序发射 `OP_MSE_BWD` → 各层 `*_BWD` → `OP_ADAM`（梯度槽 `g_`，与引擎约定一致）；
3. **判据不能是"与另一份实现逐位一致"**（没有那份实现），而应是
   **有限差分梯度检查**：`∂L/∂w` 与中心差分的差必须在 FD 的误差界内
   —— vmlm 自己就带 `vmlm-gradcheck.exe`，这条判据在那边的工具链里有先例；
4. can-fail：扰动一个权重 ⇒ 对应梯度必须变（且只影响它该影响的那几步）。

## 7.4 复现（当前边界）

```powershell
cd D:\czsai\sparrow
cargo test acceptance::training_constructs_are_a_named_boundary   # E111 + 两条 can-fail
.\target\debug\sparrow.exe <任意 .sparrow 里写 `反向`> --backend vml   # 看 E111 的提示
```

---

# §8 发布前加固：行尾 / 可移植性 / 文档同步 ✅（A 项）

> 起因：仓库外评审提出三类「不影响功能、但会在发布时咬人」的问题。三条都在**不改任何业务代码**
> 的前提下修掉，每条都给可假判据。

## 8.1 行尾归一化（6 个文件，内容零改动）

| 项 | 实测 |
|---|---|
| 症状 | `git status` / `git diff` 全干净，但编辑 `src/main.rs` 再 `add` 会得到 **1,318 行** diff（`src/vmlgen.rs` 3,922 行） |
| 根因 | `.gitattributes` 的 `* text=auto eol=lf` **不回溯改写已入库的 blob**；这 6 个文件是它生效前入库的 |
| 修法 | 工作区这 6 个文件的 CRLF → LF，一次提交（`d752b9a`） |
| 判据 1（内容中性） | 去掉 CR 后的 SHA-256：**6/6 SAME**；`git diff --cached --ignore-cr-at-eol` **输出为空** |
| 判据 2（一次性） | `git ls-files --eol` 里 `i/crlf`/`i/mixed`/`w/crlf`/`w/mixed`：修前 6 条 → 修后 **0 条** |

已写进 `docs/TRAPS.md` **H3**。

## 8.2 `tools/*.ps1` 去机器相关硬编码

| 改前 | 改后 |
|---|---|
| `$root = "D:\czsai\sparrow"` | `$root = Split-Path -Parent $PSScriptRoot`（本脚本就在 `<root>\tools\` 下） |
| `$vmlm = "D:\czsai\vmlm"` | `$env:SPARROW_VMLM_ROOT`，否则**同级目录**推导；找不到 `Cargo.toml` 即**硬失败** |
| `$git = "E:\Git\cmd\git.exe"` | 从 PATH 取，可用 `$env:SPARROW_GIT` 覆盖 |
| 注释里的「8 个驱动」 | 改成不写死数量（清单就是脚本里的 `$drivers`） |

判据（全部实测）：

* 13 个脚本精确替换 `root=1 / vmlm≤1` 次，**BOM 与行尾原样保留**（脚本逐文件核对后写回）；
* 把 CWD 换成 **`C:\`** 跑 `perf_frontend.ps1` → `exit=0`（不再依赖 CWD）；
* **can-fail**：`$env:SPARROW_VMLM_ROOT=D:\czsai\_nope_vmlm` → 报错并 `exit=1`；清掉后 `exit=0`；
  期间还抓到一版提示语 bug（双引号里 `$env:...` 被插值成**值**而不是变量名），已改成单引号模板 `-f $vmlm`；
* 剩余盘符硬编码：`tools/*.ps1` **0 条**，`src/*.rs` 本来就 0 条（编译器自身一直是可移植的）。

## 8.3 文档同步与措辞修正

| 位置 | 改了什么 |
|---|---|
| `docs/REPO.md` §1/§4 | 296 → **385** 个文件；81 → **85** passed（并注明数字**不写死**） |
| `docs/REPO.md` §2 | 上游开源树内容里的 `CAA` 已删除，改为「**暂不合并外部 PR**」 |
| `docs/REPO.md` §6 | 新增两条边界：**提交作者是占位身份**、**还没有 README/LICENSE/CI 且上游同名且为 MPL-2.0** |
| `docs/TRAPS.md` | H2 的「8 个驱动」改成不写死；新增 **H3**（行尾）⇒ 合计 **67 → 68 条** |
| `tools/repo_boundary_probe.ps1` | 注释里的「必须 81 passed」改成「与真实仓库逐项一致」 |
| 本文 §2.4 | 「产物**不依赖引擎**」改成准确说法：**不依赖引擎的二进制/VM，但静态链接引擎的源码 crate `vmlm-ops`**（`Cargo.toml` 里是 path 依赖）—— 这既是「与 `.vmb` 逐位一致」的来源，也是「单独发一个 AOT 模型要带上 `ops/`」的代价 |
| 本文文首 | 加说明：本文是逐项追加的版本记录，各节数字以该节写作时为准；**活文档**已与本版同步 |

## 8.4 复现（A 项判据）

```powershell
cd D:\czsai\sparrow
git ls-files --eol | Select-String 'i/(mixed|crlf)'                   # 0 条
pwsh -NoProfile -ExecutionPolicy Bypass -File tools\verify_all.ps1    # 退出码 0（G0/T1/10 驱动/G1）
$env:SPARROW_VMLM_ROOT='D:\czsai\_nope_vmlm'
pwsh -NoProfile -ExecutionPolicy Bypass -File tools\slot_reuse.ps1    # can-fail：exit=1 并说清原因
```

## 8.5 边界（不藏）

1. **没有重写历史**：12 个既有提交的作者仍是 `czsai <czsai@localhost>`；只把本仓库的 `user.*`
   对齐成 `shouzugzs`（**只影响将来的提交**）。重写会改掉全部哈希，而本文与 `docs/REPO.md` 引用着它们。
2. **没有做发布证件**：README / LICENSE / CHANGELOG / CI 仍未做；`Cargo.toml` 声明的
   `license = "MIT OR Apache-2.0"` 目前**没有正文**，且它与上游同名仓库（MPL-2.0）的关系待定。
3. **只去掉了硬编码路径**，没有加跨平台支持：脚本仍是 PowerShell + Windows 语义（`pwsh` 必需）。

---

# §9 注意力块内部的**每头暂存槽**接入复用 ✅（C 项之一）

§4.5 边界 1 当时写着：「注意力内部的 scratch 还没回收：`q/k/v`、每头的 `qh/kh/vh/sc`、转置 `kT`
等名字在 `emit_attn` 内部生成，要接池得先把"名字"变成可注入参数 —— **本版没做**」。本节做掉了，
但**没有**去接 §4 那个 `(行,列)` 暂存池，而是用了更省事、也更可证的一招：**按角色命名**。

## 9.1 做法（一行开关；默认路径一个字节都不变）

| 槽 | 复用关（V0.1.5 命名） | 复用开（V0.1.6） | 为什么可以 |
|---|---|---|---|
| 每头 `q/k/kt/v/s` | `ATTN_q0` / `ATTN_q1` … | `ATTN_hq`（**一份**，H 头轮流借） | 写完立刻读、读完不再有人读 |
| 每头输出 `c` | `ATTN_c0` / `ATTN_c1` | **不变**（仍按头分开） | H 份要**一起活到拼头那一刻** |
| 拼头目标 | `ATTN_ctx` | 借 `ATTN_q` | 最后一次读 `_q` 就是最后一个头的 GEMM，形状同为 `[T,D]` |

* 只有 `--slot-reuse` 打开才改名；关闭时与 V0.1.5 **逐字节相同**（默认路径不变，老判据不受影响）。
* 没有引入新的池、没有新的跨层耦合：汇编器**按槽名**分配激活区，名字少了就是字节少了。

## 9.2 判据与实测（`tools/slot_reuse.ps1`，全 PASS）

| 判据 | 实测 |
|---|---|
| 输出逐字符相同（复用开 vs 关，4 个注意力模型） | PASS：T=1 / T=4 因果 / T=4 非因果 / `重复2次` 循环 |
| 激活区变小 | 30→24 槽，**1152 B → 768 B（−33.3%）**；循环版 37→31 槽，**1408 → 1024 B（−27.3%）** |
| 结构判据 **3c**：数注意力块**目的槽名** | **17 → 11**，差额 **== `5×(头−1)+1` = 6** ✓ |
| 3c-3 / 3c-4 / 3c-5 | 五个角色槽各只有一份；`c` 仍按头分开（2 个）；`_ctx` 消失、`_q` 兼任 |
| 既有 9 个模型的数字 | **与 V0.1.5 完全一致**（1664→256 B、512→256 B …）⇒ 无回归 |
| 快速版单测 | `tests/attention.rs::slot_reuse_shares_per_head_scratch_but_not_head_outputs` |

## 9.3 can-fail（本项最有价值的一条）

把 `c` 也改成共用（**一行**改动），四个模型的 off/on 输出**全部不同**：

```
attn_causal  off: -15.786533 9.903620 11.872891 2.825854 …
             on : -1.170246  8.703536 2.947346  4.447749 …
```

⇒「输出逐字符相同」这条判据**会红**，不是恒真的。实验已写进 `docs/TRAPS.md` **G10**。

## 9.4 复现

```powershell
cd D:\czsai\sparrow
pwsh -NoProfile -ExecutionPolicy Bypass -File tools\slot_reuse.ps1   # 判据 1 / 2 / 3 / 3b / 3c / 4
cargo test --test attention                                          # 快速版
```

## 9.5 边界（不藏）

1. **`_q/_k/_v` 三个投影槽本身没有还回 §4 的暂存池**：它们被 H 个头逐个 slice，活到最后一个头；
   本节只做了"块内同形状的暂存槽轮流借"。要把它们也还回池，得把池（或 `reuse` 之外的池引用）
   传进 `emit_attn` —— 收益在 H 大时才明显，本版模型 H=2，**没做**。
2. **H=1 时几乎没有收益**（只有 `_ctx` 借 `_q` 省 1 个槽）；收益随 H 线性增长（省 `5(H−1)+1` 个）。
3. 判据 3c 里的 `头=2` 与 `17/11` 两个数是**对 `attn_causal.sparrow` 写死的**；换模型要一起改。
4. `--backend lib`（AOT）**不受影响**：Rust 侧有一套自己的槽名（`slots: HashMap<&str, Vec<f32>>`），
   不读 `.vml` 的槽名；两边数值一致仍由 `lib_ab` 守着。

---

# §10 `损失` / `反向`：训练路径的**最小可判定切片** ✅（C 项之二）

§7 当时把训练路径列为「⚠️ 勘察完毕，未落地」，理由是"**没有独立参考**"：
vmlm 的训练指令流是手写 Rust 驱动的产物，不是某个 IR 编译器的输出，所以
"我生成的 backward 与谁逐位一致"没有现成对照物。本节把这条**换掉**了 ——
参考不是另一个编译器，而是**中心差分**（纯数值、只依赖前向）。

## 10.1 语言层：`损失` / `反向` 进入 AST（不再是 `E111` 边界）

| 位置 | 改动 |
|---|---|
| `src/ast.rs` | 新增 `Stmt::Loss { kind }` / `Stmt::Backward`（`span()`/`kind()`/`count_stmt` 同步） |
| `src/parser.rs` | `损失 <函数名>` / `反向` 解析成真陈述；`训练`/`优化` 一类**仍是** `E111` |
| `src/keywords.rs` | `TrainingWord` 三分类 + `loss_kind_canonical`（`均方`/`均方误差`/`mse`/`l2` → `mse`） |
| `src/sema.rs` | `CItem::Loss` / `CItem::Backward`；损失函数 canonical 化（`E112` 带候选）；`反向` 无 `损失` ⇒ `E113` |
| `src/codegen.rs` | `.spar` 里输出 `LOSS mse` / `BACKWARD` 两行（无时间戳/路径 ⇒ 确定性不变） |
| `src/json.rs` | `--emit-ast-json` 里输出 `{"stmt":"Loss"}` / `{"stmt":"Backward"}` |

## 10.2 代码生成：反向遍历（`src/vmlgen.rs::emit_backward`）

生成的指令（实测 `examples/train_mse.sparrow`，25 条指令 / 16 槽 / 1,308 B）：

```
SHAPE   tmse_dloss, 1, 2
MSE_BWD l2_out, tmse_tgt, tmse_dloss          ; ← 梯度落在**第 3 个**操作数（与 CE_BWD 不同）
ZERO    g_l2_W ; ZERO g_l2_b                  ; ← GEMM_BWD_W / BIAS_BWD 是**累加**的
GEMM_BWD_W g_l2_W, tmse_dloss, act_out
BIAS_BWD   g_l2_b, tmse_dloss
SHAPE   l2_din, 1, 4 ; GEMM_BWD_X l2_din, tmse_dloss, l2_W
SHAPE   act_dact, 1, 4 ; ACT_BWD act_dact, l2_din, l1_out, tanh
ZERO    g_l1_W ; ZERO g_l1_b
GEMM_BWD_W g_l1_W, act_dact, tmse_in          ; ← 链首：不再产出输入梯度
BIAS_BWD   g_l1_b, act_dact
```

* 目标常量 `<模型名>_tgt` 与权重一样是**输入**（编译器不造数据）；
* 每个参数的梯度在 `PRINT` 里打出来 —— 这是训练步**唯一的可观测输出**，
  没有它，"反向算对了"就只能在驱动里读引擎内部状态自证。

## 10.3 判据与实测（`tools/train_ab.ps1`，全 PASS）

| 判据 | 内容 | 实测 |
|---|---|---|
| **0** | 在 `out/` 下**隔离构建** `--features train` 的运行时（独立 `CARGO_TARGET_DIR` + 显式 `--target`） | PASS（96.5 KB；官方推理版 74 KB） |
| **A** | 引擎打印的解析梯度 == **numpy 独立**算的解析梯度 | PASS，最大相对误差 **9.8e-06**（`l1_W[5]`） |
| **B** | numpy 解析梯度 == numpy **中心差分**（`gen_train.py` 里断言，只依赖前向） | PASS，最大相对误差 **3.0e-07** |
| **C** | can-fail：扰动 `l2_W[0]` ⇒ 对应梯度必变 | PASS：`g_l2_W[0]` 0.958233 → 0.980622（Δ=0.0224）；出口 Δ=0.0335 |
| **D** | 边界可判定：`--slot-reuse`⇒`E114`、`--loop-repeat`⇒`E114`、无损失⇒`E113`、损失名不认识⇒`E112`（带候选）、`训练`⇒`E111`、`--backend lib`⇒`E115` | PASS 6/6 |
| **E** | 隔离：官方 `vmlm-runtime.exe` 的 SHA256 前后一致 | PASS（`040FF947…`） |
| — | 快速版单测（指令顺序 + 操作数约定 + 边界） | `tests/train.rs` 3 项；`cargo test` 合计 **89 passed / 0 failed** |

**A 与 B 缺一不可**：只有 A 会被"两边抄同一份错代码"骗过；
只有 B 会在引擎侧写错时仍然通过。两条一起才等于"引擎算出来的确实是梯度"。

## 10.4 过程中抓到的两条硬事实（都写进 `docs/TRAPS.md`）

1. **`MSE_BWD` 的梯度在第 3 个操作数**，而 `CE_BWD` 在第 2 个
   （`train.rs::mse_bwd(y, t, dy, n)` 的形参顺序）。照 `CE_BWD` 写：`exit=0`、**梯度全 0** ——
   不报错、不崩溃的静默错。见 **G11**。
2. **官方 `vmlm-runtime.exe` 不带 `train` feature**：对同一个 `.vmb` 是 `exit=6` +
   `train feature disabled`；`ADAM/SGD/ADAMW` 在 `cpool_readonly` 下 `exit=9`。
   所以训练验收必须**自己隔离构建**一份训练版运行时（G11、判据 0/E）。

## 10.5 边界（不藏）

1. **只接了 `线性` 与 `激活`**（`GEMM_BWD_W/X` + `BIAS_BWD` + `ACT_BWD`）。
   引擎里还有 `LN_BWD` / `SOFTMAX_BWD` / `EMB_BWD` / `OUTER_BWD`，但它们的槽位约定
   各自不同（gamma/beta 相邻槽、逐头、词表行号），本版**没接**；碰到就硬失败 `E114`
   并列出支持范围。**注意**：连 `归一化` 都还没接 —— 这意味着"一个真实 transformer 块"
   还不能整体训练。
2. **参数更新（`ADAM`/`ADAMW`/`SGD`）不是语言的一部分**：优化器状态、学习率、步数、
   数据从哪来，都属于训练循环的语义，本版没有设计（`训练` 一词仍是 `E111` 边界）。
   所以现在能生成的是**训练步的梯度**，不是"训练循环"。
3. **`反向` 与 `--slot-reuse`/`--loop-repeat` 互斥**（`E114`，TRAPS G12）：反向要读正向激活，
   那两者会把它们回收/覆盖。这是有意的**能力边界**，不是缺陷。
4. **`反向` 与 `--backend lib`（AOT）互斥**（`E115`）：AOT 要接反向得给每个层类型生成
   backward 函数，是另一个量级的工作；本版**硬拒绝**，不产出"看起来能训练"的代码。
5. 只验了**单步**梯度（T=1、单目标、MSE）。多步 BPTT、批量、多损失**没做**；
   `序列=T>1` 下 MSE 的逐行损失**没验**（`SHAPE` 已按 `[T, D]` 声明，但没跑过就不断言）。
6. 梯度只验了**一层非线性**（tanh）的链：`GEMM_BWD_X → ACT_BWD → GEMM_BWD_W` 这条
   传递链被真的走通了，但更深的组合（比如 3 层以上、两个非线性）没单独验。

## 10.6 复现

```powershell
cd D:\czsai\sparrow
python tools\gen_train.py                                          # 素材 + 中心差分参考（自检断言）
pwsh -NoProfile -ExecutionPolicy Bypass -File tools\train_ab.ps1    # 退出码 0（A–E 全绿）
cargo test --test train                                             # 快速版：顺序/约定/边界
```

## 10.7 与 §7 的关系

§7.2 当时的结论是"不落地，给可判定边界"，§7.3 给了最小切片的四步计划。
本节把那个计划的第 1、2、4 步做掉了（损失声明、按拓扑反序发射反向算子、扰动 can-fail），
第 3 步换成了**中心差分梯度检查**（比原计划的"逐位一致"更合适 —— 本来就没有第二份实现）。
§7.4 那段"能复现的当前边界"现在读起来是**历史记录**：`E111` 只剩 `训练` 一类，
`损失`/`反向` 已经是真语法。

---

# §11 int8 铺到**注意力投影** ✅（C 项之三）

§6.5 边界 1 当时写着「只有线性层吃 int8：…注意力投影都还是 f32/bf16」。本节把它推进一格。

## 11.1 为什么注意力投影能吃 int8（以及为什么别的还是不能）

注意力里的 `Wq/Wk/Wv/Wo` **本来就是四组 `gemm_nt`**（NT 布局、权重 `[out, in]`），
所以它们能直接换成 `gemm_nt_int8` —— 反量化在核内，前端只多声明"每输出行一个 scale"。
反向同理不能想当然：`EMB`（词表行号查表）与 CSR 稀疏（邻接表）**没有**对应的 int8 内核，
归一化也不是 GEMM ⇒ 它们仍然只支持 f32/bf16。这条边界是**内核决定的**，不是没做完。

## 11.2 命名的坑（值得单独记）

反量化 scale **不能**叫 `<层>_Wq_scale`：`libgen::lower()` 取常量名**最后一段**
（`a__2_bWq` → `bwq`），于是 `ATTN_Wq_scale` 被压成 `scale`，与 softmax 的 `ATTN_scale`
**撞成同一个字段**（实测 `E0124` + `E0609`）。改成 `ATTN_sWq`（→ 字段 `swq`）后正常。
已写进 `docs/TRAPS.md` **G13**。

## 11.3 判据与实测（`tools/int8_ab.ps1` 第 7 节，全 PASS）

素材 `examples/int8_attn_bins`（f32）与 `examples/int8_attn_i8_bins`（int8），
由 `tools/gen_int8_attn.py` 确定性生成，且**精确可表示**（`w = q/64`，scale 为 2 的幂）。
模型 `examples/int8_attn.sparrow`：`线性(4→4) → 注意力(维度=4 头=2 序列=4 因果=是)`。

| 判据 | 实测 |
|---|---|
| **7a** f32 与 int8 **逐字符相同** | PASS（16 个数一字不差） |
| **7b** can-fail：动 `ATTN_sWq[0]` 一个 2^-6 ⇒ 输出必变 | PASS，最大差 **0.006678**（精确集是 0） |
| **7c** 投影体积 == 公式 `4/(1+4/K)`（K=D=4） | PASS：f32 256 B → int8(q+scale) 128 B，比值 2.000 |
| **7d** 结构：四个投影都调了 `gemm_nt_int8`，字段 `wq: Vec<i8>` + `swq/swk/swv/swo: Vec<f32>` | PASS |

**为什么 7a 是强判据**：四个投影的 `q`、每输出行的 `scale`、偏置位置、拼接顺序，
只要有一处错就不可能逐字符相同。**过程中被这条抓过一次真问题**：fixture 第一版是
`序列=1`，而 `序列=1` 时 softmax 只有一个元素 ⇒ 恒等于 V，**Q/K 投影根本不参与输出**，
于是"动 Q 的 scale 输出不变"（can-fail 恒假）。改成 `序列=4 因果=是` 后 Q/K 才真正进入判据。

## 11.4 复现

```powershell
cd D:\czsai\sparrow
python tools\gen_int8_attn.py                                     # 两套素材（自检：精确可表示）
pwsh -NoProfile -ExecutionPolicy Bypass -File tools\int8_ab.ps1    # 第 1–7 节，退出码 0
```

## 11.5 边界（不藏）

1. **只有线性层与注意力投影**吃 int8；`嵌入`/`稀疏`/`归一化` 没有对应内核（见 §11.1）。
2. **只服务 `--backend lib`**：`.vmb`/汇编器那条路没有反量化 GEMM 指令（`E109`，V0.1.5 就如此）。
3. **没有实测"int8 更快/更准"**：本节只验**接入正确性**（精确可表示时的等价性 + scale 真的被用上 + 体积公式）。
   量化误差、速度、真实模型上的精度，都要用 vmlm 自己的 `tools/p13_quant_eval.py` 那套素材来评（§6.5 边界 2 同）。
4. 只在 `D=4、头=2、序列=4` 的玩具尺寸上验过；大 `D` 下 scale 的 per-row 布局**没单独验**
   （判据 7a 的等价性对任意 D 成立的理由是"内核只吃 q+scale"，但没跑过大 D）。

---

# §12 开源证件：README / LICENSE / CHANGELOG / CI ✅（B 项）

## 12.1 许可：**MPL-2.0，与上游引擎一致**

| 文件 | 内容 | 判据 |
|---|---|---|
| `LICENSE` | MPL-2.0 **官方英文原文**，未改写 | 字节数 **16,726**；SHA-256 `FAB3DD6B…63E85`（与上游 `vmlmd/LICENSE`、`oss-license/LICENSE` 三处一致） |
| `LICENSE-NOTICE` | 集中 **Exhibit A** 通知 + 中文说明 | 未附 Exhibit B ⇒ 可依 §3.3 与 GPL 等组合；显式写了"不合并外部 PR、因此没有贡献者协议" |
| `SOURCE-HEADER.txt` | 源文件头的统一写法 | 与上游同文本 |
| `Cargo.toml` | `license = "MIT OR Apache-2.0"` → **`MPL-2.0`** + `license-file` / `readme`；版本 `0.1.0` → `0.1.6` | `cargo test --locked` 通过 ⇒ `Cargo.lock` 仍只有 1 个包（**零第三方依赖**没被破坏） |

§4 那一节写清了与引擎的许可关系：`.vmb` 那条路是**进程边界**调用引擎二进制；
`--backend lib` 生成的 crate **静态链接 `vmlm-ops`**（那部分仍受 MPL-2.0 约束，
源码可从引擎仓库获得）；而 `examples/**/*_bins` 全部由本仓库的 `tools/gen_*.py` 确定性生成，
**不含第三方代码**。

## 12.2 三份新文档

* **`README.md`**：这是什么 / 与引擎的关系（两种布局都支持）/ 快速开始（含"训练要单独构建
  `--features train` 运行时"这条）/ 语言速览（三段真实例子 + 诊断示例 + 错误码分布）/
  验收怎么跑（11 驱动 + 干净不变量 + 判据示例表）/ **已知边界 7 条**/ 目录 / 许可与贡献。
* **`CHANGELOG.md`**：0.1.0 → 0.1.6 逐版"做了什么 + 关键判据 + 复现命令"，
  早期版本号只存在于报告标题里这点也写明（不逐一改旧报告）。
* **`.github/workflows/ci.yml`**：`ubuntu` + `windows` 两平台跑 `cargo test --locked`（89 项）
  与前端确定性冒烟（`tiny_cn`/`tiny_en`/`tiny_mixed` 三种写法必须产出**逐字节相同**的 `.spar`
  + `level2` 的两条结构 grep）。**诚实边界**：完整验收需要引擎（另一个仓库），
  CI 里**不假装**跑它，而是在最后一步打印怎么在本地跑。

## 12.3 版本与标签

* `Cargo.toml` / `Cargo.lock` 版本 = **0.1.6**；本地打了标签 **`v0.1.6`**（首个 MPL-2.0 版本的
  可证明快照）。仓库**仍然没有 remote** —— 要不要推、推到哪，由维护者决定。

## 12.4 "子库还是合并"的结论

**推荐合并**（作为引擎仓库的子目录 `<引擎仓库>/sparrow/`），理由：
1. 本语言**离开引擎不可用**（`.vmb` 要汇编器/运行时，AOT 要 `vmlm-ops` 源码）⇒ 一次 clone 就该拿到全套；
2. 合并后 `--vmlm-root` 的默认推导（同级目录）**正好等于引擎仓库根**，不用额外参数；
3. 两个仓库许可口径已统一成 MPL-2.0，合并**不产生混许可问题**；
4. 名字冲突随之消失："Sparrow 项目 = 引擎 + 它的描述语言"；
5. `docs/REPO.md` §4 已经把这件事实测过（判据 1a/1b/2/3，含 can-fail：删掉嵌套
   `[workspace]` 行必须失败），代价是零。

**独立性没有丢**：`Cargo.toml` 第一行就是 `[workspace]`，它自己仍是 workspace 根，
父仓库的 `members` 是显式列表，不会把它吸进引擎的构建/链接面。真要拆出来，
拷贝一个目录即可（步骤见 `docs/REPO.md` §5）。

## 12.5 复现

```powershell
cd D:\czsai\sparrow
git log --oneline -1                                  # B 项提交
git tag -l                                            # v0.1.6
cargo test --locked                                   # 89 项，零依赖
pwsh -NoProfile -ExecutionPolicy Bypass -File tools\verify_all.ps1   # 退出码 0
(Get-FileHash LICENSE -Algorithm SHA256).Hash         # FAB3DD6B…63E85
```

## 12.6 边界（不藏）

1. **没有 remote，所以没推**：标签与提交都在本地；`README` 里的仓库 URL 按上游域名写，
   真推之前要确认仓库名与地址。
2. **`CONTRIBUTING.md` / `SECURITY.md` / `TRADEMARK.md` 没做**：它们的口径已经被
   `README §8` 与 `LICENSE-NOTICE §5–7` 覆盖（不合并外部 PR、无贡献者协议、不授予商标权）；
   上游 `oss-license` 工具包里有现成模板，将来要独立成文可直接取用。
3. **CI 只跑不需要引擎的部分**：这是有意的诚实边界（见 12.2），不是"全绿"。
4. **没有做发布证件之外的事**：不带 `NOTICE` 级第三方清单（本仓库无第三方代码）、
   没有 crates.io 发布计划（`repository` 字段尚未指向真实远端）。


---

# §13 合并为引擎仓库的子目录 ✅（B 项后续）

## 13.1 动作

`robocopy D:\czsai\sparrow D:\czsai\vmlm\sparrow /E /XD target out .git` → 432 个文件；
随后**去掉与仓库根重复的证件**（`LICENSE` / `LICENSE-NOTICE` / `SOURCE-HEADER.txt`，
仓库根已有同哈希的那三份），把 CI 挪到仓库根 `.github/workflows/sparrow.yml`
（GitHub 只读根目录的 workflows，且加 `working-directory: sparrow`）⇒ 合并后 428 个文件。
`Cargo.toml` 去掉 `license-file`（正文在仓库根），`README.md` 的许可证件指向改为 `../`。
独立仓库 `D:\czsai\sparrow` **保留为 v0.1.6 的历史快照**（含它的 6 个提交与 `v0.1.6` 标签）。

## 13.2 判据（实测）

| # | 判据 | 实测 |
|---|---|---|
| 1 | 语言自身 `cargo test --locked`（在 `vmlm\sparrow`） | **11 个 target / 89 passed / 0 failed** |
| 2 | 完整验收 `sparrow\tools\verify_all.ps1`（引擎在上一级） | **11 个驱动全部 exit=0，退出码 0** |
| 3 | **引擎字节纪律**：加了 `sparrow/` 之后重建 runtime | `vmlm-runtime.exe` **75,776 B 逐字节相同**（`040FF947…`）；`vmlm-asm.exe` 200,192 B 相同（`13ACA62C…`） |
| 4 | 父仓库不认识语言 crate | `cargo metadata --no-deps \| Select-String sparrow` → **0 行** |

判据 3 是关键：`Cargo.toml` 第一行是 `[workspace]`，父仓库的 `members` 是显式列表，
所以多一个 std crate 不会进引擎的构建/链接面 —— 这次是在**真树**上验的（§4 当时只能用镜像）。

## 13.3 合并把一个假通过和四个隐藏依赖逼了出来

**全新工作区**（`out/`、`target/release` 都空）一跑，11 个驱动**全 exit=1**。根因是五处
"只在老工作区里成立"的假设 —— 它们让独立仓库的三次"全绿"都掺了残留：

1. **引擎路径推导只支持"同级 `vmlm`"**：合并布局里引擎是**上一级** ⇒ 11 个驱动全失败。
   已改成三级推导：`$env:SPARROW_VMLM_ROOT` → 上一级若有 `ops/Cargo.toml`（合并布局）→ 同级 `vmlm`。
2. **`tools/*.py` 有 10 处硬编码 `D:\czsai\sparrow`**（`gen_seq.py` / `gen_residual.py` /
   `gen_bf16.py` / `gen_e2e_tiny.py` / `gen_loop24.py` / `gen_attn.py` / `gen_bench.py` /
   `gen_int8.py` / `gen_loop_two.py`）：素材被写到**旧路径**，于是 `residual_ab` / `seq_ab`
   拿不到刚生成的权重。已全部换成 `__file__` 推导。
   （A 项的"去硬编码"只扫了 `.ps1`，漏了 `.py` —— 这次补上。）
3. **`perf_frontend.ps1`**：`target\release\sparrow.exe` 不存在就直接 `throw`。已改成
   缺了就自己 `cargo build --release`（新克隆/合并布局下必然缺）。
4. **`train_ab.ps1` 的 `.vml` 读错了文件**：它读 `out\train\model.vml`，而汇编器是把 `.vml`
   写在**它的 `.vmb` 旁边**（现在是 `pristine.vml`）。上一轮它读到的是我调试时留下的**残留文件** ——
   **判据 2c 是假通过**。已改成读 `pristine.vml`；修正后重跑仍 PASS（真正的判据 2c 过了）。
5. **`seq_ab.ps1` 依赖 `out\aot_bench\engine_plain`**（那是 `aot_bench` 的产物）——
   新工作区里不存在 ⇒ 必失败。已改成自带 `out\seq\engine` 并拷贝官方运行时。

**另外还修了一处"判据本身会假通过"**：`verify_all.ps1` 的 G0/G1 用
`@(& git status).Count -eq 0`，在本目录不属于任何 git 仓库时（私有工作副本 / 合并布局的
`vmlm` 树）会**空数组恒等于空 ⇒ 自动 PASS**。已改成：先 `git rev-parse --show-toplevel`
判断，不在仓库里就**显式 SKIP**（打印"本目录不在 git 仓库里"），在仓库里则只对**本目录**
`git status --porcelain -- .`。发布树 `vmlmd` 是真 git 仓库，那两条判据在那边是真的。

## 13.4 复现

```powershell
cd D:\czsai\vmlm\sparrow
cargo test --locked                                   # 89 项
pwsh -NoProfile -ExecutionPolicy Bypass -File tools\verify_all.ps1   # 11 驱动，退出码 0
# 引擎字节纪律（在引擎根）
cd D:\czsai\vmlm\runtime ; cargo build --target x86_64-pc-windows-msvc --release
(Get-FileHash D:\czsai\vmlm\target\release\vmlm-runtime.exe -Algorithm SHA256).Hash   # 040FF947…
```

## 13.5 边界（不藏）

1. 独立仓库 `D:\czsai\sparrow` **冻结在 v0.1.6**；13.3 的修复只落在合并树里
   （即"以后从合并树里拷出来"的那份才是最新的）。
2. 合并后**没有**给语言单独建 `Cargo.lock` 之外的版本标识：它的版本号仍在
   `sparrow/Cargo.toml` 里，随引擎仓库一起走。
3. 引擎仓库的 CI 新增了 `sparrow.yml`（只跑不需要引擎的那部分），但**还没有在 GitHub 上
   真跑过一次** —— 推上去之后第一次 push 才会见分晓。

## 13.6 更正一条我自己先说错的判据（引擎"逐字节不变"）

13.2 表格里我最初写的是"重建后 `vmlm-runtime.exe` 与合并前**逐字节相同**"。**那条是空验**：
带 `--target x86_64-pc-windows-msvc` 的构建落在 `target\<triple>\release\`，而我比对的
`target\release\vmlm-runtime.exe` 是 **10-10 的历史文件，根本没被重建**。

做了严格的 A/B 之后，事实是：

* 引擎产物**每次构建都不同**：同一棵树连续两次构建差 **20 字节**（大小恒为 75,776 B）；
* 那 20 个字节的位置在两次实验里**完全一样**：`0xD0`、`0x11EF8`、`0x11F14`、`0x11F30`、
  `0x11F64`–`0x11F6B`；
* 把 `sparrow/` 挪走再构建，与留下时相比差的也是**同一组 20 字节**。

⇒ **在 `sparrow/` 存在与否这件事上，产物没有可检出的差异**；那 20 字节是链接期噪声
（PE / 调试元数据），不是合并带来的。**但"逐字节不变"作为判据本身是立不住的**：
它既无法证明不变（每次构建都不同），也无法在真出问题时报警 —— 这是本轮发现的
**项目级问题**，不是合并的锅。建议按可判性从强到弱换掉它：

1. **行为**（最强）：引擎自带测试 + 维护者脚本里的 20 步训练 `loss_hash=0xea82fbdd26204a70`；
2. **包图**：`cargo metadata --no-deps` 里不含语言 crate（本轮已验：0 行）；
3. **字节**（次强、可自动化）：大小不变（75,776 B）+ **扣掉那 20 个噪声字节后逐字节相同**；
4. 想要真正的字节级可重复构建：给链接器加 `/Brepro`（属于引擎侧改动，**本轮没做**）。

## 13.7 复现（合并后的两棵树）

```powershell
# 私有工作副本（不是 git 仓库：G0/G1 会显式 SKIP，不当成 PASS）
cd D:\czsai\vmlm\sparrow
pwsh -NoProfile -ExecutionPolicy Bypass -File tools\verify_all.ps1     # 11 驱动，退出码 0

# 公开树（真 git 仓库：G0/G1 是真判据 —— 未提交时会 FAIL，这正确）
cd D:\czsai\vmlmd\sparrow
cargo test --locked                                                   # 89 项
pwsh -NoProfile -ExecutionPolicy Bypass -File tools\verify_all.ps1      # 11 驱动 + G0/G1
```