# Sparrow —— 神经网络架构描述语言

> **本目录是引擎仓库的子目录 `sparrow/`**：虚拟机 / 汇编器 / 训练栈就在**上一级**
> （`asm/`、`core/`、`ops/`、`runtime/`、`train/`…）。所以工具默认推导出来的 `--vmlm-root`
> **正好是引擎仓库根**，不用额外参数。许可证件与安全/商标声明也在仓库根。

> **写架构，不写代码。** `模型 → 层 → 连接 → 前向` 这几行就是全部语法；中英文关键字随便混写、
> 标点随便、缩进随便、四套块机制任选。产物是**确定性的**：同一份语义 ⇒ 同一个 `.vmb` 字节流。

| 项 | 状态 |
|---|---|
| 第三方依赖 | **0**（`Cargo.lock` 里只有它自己一个包） |
| 测试 | `cargo test` → **89 passed / 0 failed**（11 个 target） |
| 验收驱动 | **11 个**，一条命令全跑：`pwsh -NoProfile -ExecutionPolicy Bypass -File tools/verify_all.ps1` |
| 跑完之后的不变量 | 全部驱动跑完，`git status` **必须为空**（生成物入库当指纹） |
| 许可 | **MPL-2.0**（见 [LICENSE](../LICENSE) 与 [LICENSE-NOTICE](../LICENSE-NOTICE)） |
| 平台 | 编译器本身零平台依赖；验收脚本是 PowerShell 7（Windows） |

---

## 1. 这是什么

一门**描述神经网络架构的小语言**，外加两个后端：

```
源码 .sparrow ──lexer/parser/sema──▶ .spar（架构 IR）──codegen──▶ .vml（opcode）──汇编器──▶ .vmb（字节码）
                                                          └── lib 后端 ──▶ Rust AOT 源码
```

* **`.vmb` 后端**：交给 [VMLM 引擎](https://github.com/shouzugzs-AI/Sparrow) 的汇编器与运行时执行。
* **lib 后端**（`--backend lib`）：直接把架构生成成一份**普通 Rust 程序**，
  它 `use vmlm_ops::kernels::*` —— 与引擎**同一套算术内核**，所以两条路的输出可以逐字符比。
* **训练路径**（V0.1.6）：`损失 均方` + `反向` ⇒ 前端按拓扑反序发射 `MSE_BWD` /
  `GEMM_BWD_W` / `GEMM_BWD_X` / `BIAS_BWD` / `ACT_BWD`，并把每个参数的梯度打印出来。

## 2. 与 VMLM 引擎的关系（先读这段）

Sparrow 是 VMLM 的**描述语言**，同一位作者、同一个许可口径（MPL-2.0）。但**它是独立的 crate**：

* `.vmb` 那条路要用引擎的 `vmlm-asm.exe` / `vmlm-runtime.exe`；
* lib 后端生成的 crate **静态链接引擎的 `vmlm-ops`**（这是"与 `.vmb` 逐字符一致"的来源，
  也是代价：单独发一个 AOT 模型要把 `ops/` 一起带上）。

因此本仓库可以有两种布局，**工具两种都支持**：

| 布局 | 怎么给引擎路径 |
|---|---|
| **独立仓库** | 引擎放在同级目录（默认推导 `../vmlm`），或用 `--vmlm-root` / `$env:SPARROW_VMLM_ROOT` 指定 |
| **作为引擎仓库的子目录**（`<引擎仓库>/sparrow/`）—— **当前布局** | 默认推导出来就是引擎仓库根 —— **不用额外参数** |

> **名字说明**：上游**仓库名**也叫 `Sparrow`，那是**整套 VMLM**（虚拟机 + 训练栈 + 工具链）；
> 本仓库是其中的**描述语言**（文本前端 + 两个后端）。读文档时别混。

## 3. 快速开始

```bash
# 构建（仓库根就是 workspace 根，第一行是 [workspace]）
cargo build --release          # target/release/sparrow.exe

# 只看前端（不需要引擎）
./target/release/sparrow examples/level2.sparrow                 # → .spar（架构 IR）
./target/release/sparrow examples/level2.sparrow --emit-ast-json # → AST JSON
./target/release/sparrow examples/level2.sparrow --emit-tokens   # → token 流（调试宽容语法用）
./target/release/sparrow examples/level2.sparrow --backend vml   # → opcode .vml
```

要出**字节码**或 **Rust AOT**，需要引擎在旁边：

```powershell
# 1) 一条命令跑全链：.sparrow → .spar → .vml → fork 汇编器 → .vmb
sparrow examples/e2e_tiny.sparrow --asm <引擎>/target/release/vmlm-asm.exe `
        --consts-dir examples/e2e_bins -o out/e2e.vmb

# 2) Rust AOT：生成一份普通 Rust 程序（含 Cargo.toml）
sparrow examples/attn_t1.sparrow --backend lib --lang rust `
        --vmlm-root <引擎仓库根> --consts-dir examples/attn_bins -o out/aot_attn

# 3) 训练步（梯度检查全绿的那条路）
python tools/gen_train.py
pwsh -NoProfile -ExecutionPolicy Bypass -File tools/train_ab.ps1
```

> **训练为什么要单独构建运行时**：官方 `vmlm-runtime.exe` 是**推理**引擎（体积预算 40 KB），
> 反向算子整体在 `#[cfg(feature = "train")]` 后面 —— 它对训练步会 `exit=6` 并打印
> `train feature disabled`。`tools/train_ab.ps1` 会在 `out/` 下**隔离构建**一份
> `--features train` 的运行时（独立 `CARGO_TARGET_DIR`），并用官方二进制 SHA256 不变
> 证明"没有为了训练改引擎"。

## 4. 语言速览

```sparrow
# 全中文、全中文标点、全角括号都能写；也可以混英文
模型 attncausal：
  线性 输入=4 输出=4 序列=4
  注意力 维度=4 头=2 序列=4 因果=是
  连接 线性 注意力
  前向 线性 注意力
```

```sparrow
# 循环、残差、复用、量化、训练：都是几行
模型 resloop:
  嵌入 词表=8 维度=4
  t = 线性 输出=4 重复12次          # `重复` 默认展开；--loop-repeat 走 LOOP_CNT + LAYERB（指令流 O(1)）
  mg = 线性 输出=4
  连接 嵌入 t                      # t 的输入
  连接 t mg                        # 残差：mg 同时读 t 与 嵌入
  连接 嵌入 mg
  前向 嵌入 mg
```

```sparrow
# 训练步（V0.1.6）：损失 + 反向。目标常量由使用者提供，编译器不造数据
模型 tmse:
  l1 = 线性 输入=4 输出=4
  act = 激活 激活=tanh
  l2 = 线性 输入=4 输出=2
  连接 l1 act l2
  前向 l1 l2
  损失 均方
  反向
```

几个刻意做成"能推就不要求写"的地方：层的宽度能从上游推出来就不用写 `维度=`；
`嵌入 512` 这种位置参数会绑定到规范键；`REPEAT`/`重复`/`x 24`/`叠 24 个` 是同一个结构化节点。

**诊断**：错误码是命名的，带位置、插入符（中文按显示宽度对齐）、提示与候选：

```
错误[E003]：层类型 "稀疏块" 需要 2 个参数（维, 连），只给了 1 个
  位置：examples/err_missing_arg.sparrow:7:7
   7 |   t = 稀疏块(1536)
     |       ^^^^^^
  提示：例：`稀疏块(<维>, <连>)`
  候选：维, 连
```

前端 `E0xx`（词法/语法/语义，含 `E010` 注释未闭合、`E100` 汇入宽度不符）、
后端 `E1xx`（`E101` 不支持的自定义算子、`E102` 两个链首、`E107` 序列>1 的非逐行算子、
`E109` 后端/精度组合不支持、`E110` 源层不能循环化、`E111` 训练循环仍是边界、
`E112` 不认识的损失函数、`E113` 反向缺损失、`E114` 反向与复用/循环互斥、
`E115` AOT 不支持反向），另有极少数 `E2xx`（如 `E212` 循环步长不能被 f32 精确表示）。
**"非法输入要硬失败并说清楚"** 是本项目的纪律：宁可报错，不产出一份"看起来能跑"的产物。

## 5. 验收：我不是靠"看起来对"

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File tools/verify_all.ps1   # 退出码 0 = 全绿
```

它跑三件事：`cargo test`（89 项）→ **11 个验收驱动** → 断言**跑完之后工作区仍然干净**。

最后那条不变量是这套东西的支点：`examples/` 下的 `.spar/.vml/.vmb/.args/*_bins` **全是生成物**，
本仓库仍然入库，于是"跑完驱动 `git status` 必须为空"同时证明两件事，且**都可假**：
生成器是确定性的；验收过程没有留下未回收的改动。

每个驱动都给**可假（can-fail）**判据，而不是"跑通了"。几个例子：

| 驱动 | 判据长什么样 |
|---|---|
| `slot_reuse` | 复用开/关输出**逐字符相同**；激活区必须变小；**结构上数槽名**（17 → 11）；把每头输出槽也共用（一行改动）⇒ 输出必变 |
| `attn_ab` | `注意力(T=1)` 恒等于三层线性串联；因果掩码第 0 行 == T=1；不因果必须**不同** |
| `seq_ab` | T=3 的每一行 == 该词元单独喂进去的 T=1 输出；numpy 独立参考；只改第 2 个词元 ⇒ 只有第 2 行变 |
| `int8_ab` | 精确可表示（`w=q/64`）时 f32 与 int8 **逐字符相同**；动一行 scale ⇒ 差值必须**远超**界；体积比 == 公式 |
| `train_ab` | 引擎解析梯度 vs numpy **中心差分**（最大相对误差 9.8e-06）；扰动权重 ⇒ 对应梯度必变；官方运行时哈希前后一致 |
| `lib_ab` | 10 个模型的 `.vmb` 输出 == Rust AOT 输出（逐字符）；循环组的 Rust 是"一份代码 + N 份常量" |
| `aot_bench` | AOT vs 解释器进程内同口径计时；**can-fail** 用"改一个权重字节输出必须变"来证明基准不是恒真 |

## 6. 已知边界（不藏）

1. **训练只接了 `线性` 与 `激活`**（`GEMM_BWD_W/X` + `BIAS_BWD` + `ACT_BWD`）。
   `归一化`/`注意力`/`嵌入`/`稀疏` 还没有反向内核接线 ⇒ 碰到就 `E114` 说清范围。
   所以"一个真实 transformer 块"**还不能整体训练**；**参数更新**（`ADAM`/`ADAMW`/`SGD`）
   也还不是语言的一部分（优化器状态、学习率、步数、数据来源都是训练循环的语义，本版没有设计）。
2. **`反向` 与 `--slot-reuse` / `--loop-repeat` 互斥**（`E114`）：反向要读正向激活，
   而它们会把那些槽回收/覆盖 —— 硬失败，不"警告后继续"。
3. **`--backend lib` 不支持 `反向`**（`E115`）：AOT 侧要接反向，得给每个层类型生成一份
   backward 函数，本版硬拒绝而不产出"看起来能训练"的代码。
   （循环化的 `重复` 组 lib 后端**是支持的**：源码长度 O(1)，Rust 侧"一份代码 + N 份常量"。）
4. **int8 只服务 lib 后端**，且只作用于"GEMM 形态"的权重（线性层、注意力四个投影）；
   `嵌入`/`稀疏`/`归一化` 没有对应内核。量化**不在编译器里**：`.bin` 由使用者提供。
   本仓库只验接入正确性，**没有**实测"int8 更快/更准"。
5. **AOT 产物不依赖引擎的二进制/VM，但静态链接引擎的源码 crate**（`vmlm-ops`）。
6. 验收脚本需要 Windows + PowerShell 7 + 引擎在旁边（CI 里只跑不需要引擎的那部分，见 `.github/workflows/ci.yml`）。
7. 引擎侧对含第三方材料的部分有自己的 `THIRD-PARTY-NOTICES.md`；本仓库的 `examples/**/*_bins`
   全部由本仓库的 `tools/gen_*.py` 确定性生成，不含第三方代码。

## 7. 目录

```
src/            编译器（lexer / parser / sema / codegen / vmlgen / libgen / diag / json / sha256）
tests/          89 项测试（验收 1–12 + 注意力 + 循环 + 合并 + 后端 + 训练）
examples/       47 份 .sparrow 示例 + **生成物指纹**（.spar/.vml/.vmb/.args/*_bins）
tools/          gen_*.py（确定性素材）+ *_ab.ps1（验收驱动）+ verify_all.ps1（一条命令）
docs/TRAPS.md   73 条实测踩过的坑（"现象 → 根因 → 对策"）
docs/REPO.md    仓库边界与迁移手册（独立仓库 vs 并入引擎目录，含实测判据）
REPORT.md       V0.1 前端报告（验收 1–12 的逐条证据）
REPORT_V0*.md   各版报告；REPORT_V015.md 是主报告（§8 起为 V0.1.6 的加固与新功能）
```

## 8. 许可 / 贡献 / 商标

* **MPL-2.0**：[LICENSE](../LICENSE) 是官方英文原文（未改写，SHA-256 `FAB3DD6B…63E85`，16,726 B）；
  集中的 Exhibit A 通知在 [LICENSE-NOTICE](../LICENSE-NOTICE)（**未**附加 Exhibit B ⇒
  可依 §3.3 与 GPL 等 Secondary License 组合）。**未**为每个源文件加许可证头，见 [SOURCE-HEADER.txt](../SOURCE-HEADER.txt)。
* **不合并外部 Pull Request**：与上游同一口径（维护者精力有限）。因此本项目**没有**贡献者协议，
  任何人仍可依 MPL-2.0 使用、修改、分发；想贡献的话以 issue 提报告/复现更有用。详见 [LICENSE-NOTICE](../LICENSE-NOTICE) §5。
* **商标**：`LICENSE` 与 `LICENSE-NOTICE` 都**不**授予商标权；名称尚未注册。
* 商业使用 MPL-2.0 已经允许；需要非 MPL 条款（例如在闭源产品里修改 MPL 覆盖文件）请联系
  [@shouzugzs](https://github.com/shouzugzs)。
