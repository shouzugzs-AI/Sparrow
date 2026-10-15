# Sparrow V0.1.1 —— 端到端跑通 + 两种后端方向

> 交付物：`D:\czsai\sparrow`（45 项测试全绿：21 单元 + 15 前端验收 + 9 后端回归）
> 端到端驱动：`tools/e2e_tiny.ps1`（调真汇编器 + 真引擎，**两者都没改一行**）
> 本文按任务书第七部分的 5 项组织；另附命名约定、CLI、后端路线、边界判断、遗留项。

## 0. 总表

| 项 | 状态 |
|---|---|
| 四层后缀命名（`.sparrow`/`.spar`/`.vml`/`.vmb`） | ✅ 前端输出改名 `.vml` → `.spar` |
| 第一步：opcode 展开规则表 | ✅ 见 §1（10 类语句） |
| 第二步：槽位分配器 | ✅ 见 §2（朴素、不复用、每实例 1 槽） |
| 第三步：e2e_tiny 端到端 | ✅ 见 §3/§4，**引擎输出逐字节相同**（sha256 `3AD018C7…`） |
| 后端路径 1（编译为 VML） | ✅ V0.1.1 完成 |
| 后端路径 2（拼装库 → Rust） | ⛔ V0.1.2（CLI 已预留 `--backend lib`） |
| 不改汇编器 / 不改引擎 | ✅ 只调用：`vmlm-asm.exe`、`vmlm-runtime.exe` |
| 不做槽位复用 / 不做完整优化 | ✅ 明确留给 V0.1.2 |

## 1. opcode 展开规则表

符号：`%x` 激活槽、`@x` 常量槽（由 `.args` 清单绑定到 `.bin`）。
`T` = 序列/批量长度（V0.1.1 固定 1）；`D` 维度、`K` 连接数、`V` 词表、`O` 输出维度。

| Sparrow 语句 | 展开的 opcode 序列 | 常量（`.args` 清单行） |
|---|---|---|
| `模型 名:` | `.section forward` 头（汇编器跳过 `.` 开头行，仅为对齐既有示例） | — |
| `嵌入 词表=V 维度=D` | `SHAPE %n_out,1,D`<br>`EMB %n_out, @n_E, @模型_in` | `n_E:[V,D]` |
| `稀疏 维度=D 连接=K` | `SHAPE %n_col,D,1`<br>`SPARSE_MATVEC %n_col, @n_off, @n_src, @n_W, <上游>`<br>`TRANSPOSE %n_out, %n_col` | `n_off:[D+1,1]`<br>`n_src:[D*K,1]`<br>`n_W:[D*K,1]` |
| `线性 输入=I 输出=O` | `GEMM %n_out, <上游>, @n_W`<br>`BIAS %n_out, @n_b` | `n_W:[O,I]`（NT 布局）<br>`n_b:[1,O]` |
| `归一化 维度=D` | `LN %n_out, <上游>, @n_g, @n_b` | `n_g:[1,D]` `n_b:[1,D]` |
| `激活 act=A` | `ACT %n_out, <上游>, A` | — |
| `连接 a b c` | **不产生 opcode** —— 决定数据流路由：下层读哪个槽 | — |
| `前向 入 出` | 展开顺序 = 主入口序列；末尾 `PRINT %出_out` + `HALT` | — |
| `重复 n` | opcode 层**展开成 n 份**（各自独立槽与常量） | 每份一套 |
| `算子 …` / `@op …`（Level 5/6） | **不展开**，报 `E101` | — |

规则细节与理由（都在源码注释里）：

- **`EMB` 必须显式 `SHAPE`**：汇编器的形状推断表里没有 `OP_EMB`，不写 `SHAPE` 输出槽就停在
  `0x0`，引擎要么报错要么静默不写。这是"按文档写好的程序装载后立刻死掉"那一类坑。
- **`稀疏` 后面必须跟 `TRANSPOSE`**：引擎对 `SPARSE_MATVEC` 有硬契约
  （`interp.rs`：offsets/src 必须是列向量、`dst must be [n_neurons(,1)]`），
  而下游 `GEMM` 要的是行向量。**不能用 `RESHAPE`** —— 汇编器把它列在 `UNIMPLEMENTED_OPS`
  （"形状重解释未实现"），用了会汇编期硬失败；`TRANSPOSE` 是实现过的。
- **`线性` 用两条指令（`GEMM` + `BIAS`）而不是 `bias=` 融合 flag**：实测 `bias=xxx` 里的槽位名
  会被汇编器当 flag 吃掉（汇编器只取 flag 名、丢弃值），偏置槽落成 0 → 读 slot0 垃圾。
  仓库里所有真实示例也都是分开写 `GEMM` + `BIAS`。
- **`连接` 不产生指令**：Sparrow 是声明式，路由 = 决定 `GEMM`/`SPARSE_MATVEC` 的输入槽是谁。
- **`重复 n` 是展开**：`%t__1 … %t__n` 各自独立槽，24 层 → 24 组指令。任务书要求 V0.1.1
  不做优化，`LOOP_CNT`/`LAYERB`（让指令数与层数解耦）排在 V0.1.2。

**稀疏层的 CSR 拓扑**（V0.1.1 固定、确定性规则）：
神经元 `j` 的第 `t` 条连接指向输入 `(j*K + t) mod D`，于是
`off[j] = j*K`、`off[D] = D*K`，行内列号单调递增 —— 与 CSR 的遍历顺序一致，
累加顺序因此与稠密参考完全相同（这是 §4 能逐位比较的前提）。

## 2. 槽位分配算法

朴素版本（任务书明确要求 V0.1.1 不复用）：

```
输入：拓扑序的层实例列表 L0..Ln（`重复` 已展开成多份）
1. 模型输入   = 常量槽 <模型名>_in
                 链首是嵌入 → [1,1]（词元 id）
                 否则       → [1, 链首输入维度]
2. 每个实例 i 独占一个输出激活槽 <实例名>_out
3. 权重一律是**常量槽**（不是激活槽）：<实例名>_E / _W / _b / _off / _src / _g
4. 第 i 层的输入槽 = 第 i-1 层的 <...>_out；链首 = 模型输入槽
5. 稀疏层额外需要一个列向量中转槽 <实例名>_col（因为引擎要求 dst 是 [n,1]）
```

实测（`e2e_tiny`，3 层）：`11 个槽位, 7 个常量`；`g.slots == ["EMBED_out","SPARSE_out","LINEAR_out"]`。

**名字安全**（两条，来自汇编器的既有约定）：

1. 实例名不是"ASCII 字母数字下划线"时换成 `<小写层类型><序号>`（如 `嵌入` → `EMBED`）；
   中文名在符号表/`.args` 命令行/工具链上按字节走，收益为零、风险不为零。
2. 名字**不能以 `st_` 或 `g_` 开头** —— 汇编器用这两个前缀区分"持久状态槽"和"梯度槽"。
   撞上就会把普通激活槽解释成别的种类，属于静默降级那一类；实现里直接改名规避。

**槽位复用（V0.1.2）**：目前 n 层占 n 个槽。可复用的是"生命周期不重叠"的槽
（链式结构里除了被 `连接` 分叉读到的槽，其余用完即可回收），预计能压到 O(1)~O(宽度)。

## 3. e2e_tiny：源码 → .spar → .vml → 汇编 → 引擎

### 3.1 源码 `examples/e2e_tiny.sparrow`

```sparrow
模型 tiny:
  嵌入 词表=8 维度=4
  稀疏 维度=4 连接=2
  线性 输出=4
  连接 嵌入 稀疏 线性
  前向 嵌入 线性
```

### 3.2 `.spar`（架构 IR，244 B）

```spar
; sparrow v0.1 — canonical .spar IR (architecture level)
; model: tiny
.model tiny
  LAYER   EMBED EMBED vocab=8 dim=4
  LAYER   SPARSE SPARSE dim=4 k0=2
  LAYER   LINEAR LINEAR out=4
  CONNECT EMBED SPARSE LINEAR
  FORWARD EMBED LINEAR
.end
```

### 3.3 生成的 opcode `.vml`（975 B，`--backend vml`）

```vml
; sparrow v0.1.1 — opcode .vml (generated from .spar)
; model: tiny
; 常量清单（同 .args）：
;   tiny_in          tiny_in.bin          [1, 1]  模型输入（词元 id）
;   EMBED_E          EMBED_E.bin          [8, 4]  嵌入表 [vocab, dim]
;   SPARSE_off       SPARSE_off.bin       [5, 1]  CSR 行偏移（f32 存整数）
;   SPARSE_src       SPARSE_src.bin       [8, 1]  CSR 列索引（f32 存整数）
;   SPARSE_W         SPARSE_W.bin         [8, 1]  CSR 连接权重
;   LINEAR_W         LINEAR_W.bin         [4, 4]  权重 [out, in]（NT 布局）
;   LINEAR_b         LINEAR_b.bin         [1, 4]  偏置 [1, out]
.section forward
    SHAPE   EMBED_out, 1, 4
    EMB     EMBED_out, EMBED_E, tiny_in
    SHAPE   SPARSE_col, 4, 1
    SPARSE_MATVEC SPARSE_col, SPARSE_off, SPARSE_src, SPARSE_W, EMBED_out
    TRANSPOSE SPARSE_out, SPARSE_col
    GEMM    LINEAR_out, SPARSE_out, LINEAR_W
    BIAS    LINEAR_out, LINEAR_b
    PRINT   LINEAR_out
    HALT
.end
```

配套 `.args`（权重清单，由 `--args` 写出，逐行 `name:file:rows:cols`）：

```
tiny_in:tiny_in.bin:1:1
EMBED_E:EMBED_E.bin:8:4
SPARSE_off:SPARSE_off.bin:5:1
SPARSE_src:SPARSE_src.bin:8:1
SPARSE_W:SPARSE_W.bin:8:1
LINEAR_W:LINEAR_W.bin:4:4
LINEAR_b:LINEAR_b.bin:1:4
```

### 3.4 汇编（既有汇编器，未改一行）

```
generated: OK: 9 条指令, 7 个常量, 11 个槽位, 1104 字节 → e2e_tiny_generated.vmb
reference: OK: 7 条指令, 8 个常量,  8 个槽位, 1100 字节 → e2e_tiny_ref.vmb
```

### 3.5 引擎输出（既有引擎，未改一行）

```
[生成的]  exit=0   0.975111 1.749076 -0.720557 -0.854350
[手写参考] exit=0   0.975111 1.749076 -0.720557 -0.854350
```

numpy 手工参考（f32 顺序累加）：
`0.975112 1.749076 -0.720557 -0.854351` —— 末位差是 BLAS 归约顺序不同，与引擎无关。

## 4. 与手写 `.vml` 的 sha256 对比

手写参考 `examples/e2e_tiny.vml` **故意走另一条指令路径**：稀疏层用
`GEMM` + 稠密等价矩阵 `DENSE_W`（`W[n,k]` = 连接 `n←k` 的权重），
而生成版走 `SPARSE_MATVEC`(CSR) + `TRANSPOSE`。
所以这一比较是"两条不同指令流算同一个函数"，而不是"两段相同文本"。

| 项 | 值 |
|---|---|
| 生成的输出 sha256 | `3AD018C7CF7B4B97E5CCDFA71236D5C5C340B8637E6E1E414B89042847945B3B` |
| 手写参考输出 sha256 | `3AD018C7CF7B4B97E5CCDFA71236D5C5C340B8637E6E1E414B89042847945B3B` |
| 输出字节数 | 39 vs 39 |
| **差异字节数** | **0** |
| `.vmb` 大小 | 1104 B（生成） vs 1100 B（手写，多一个稠密权重常量、少两条指令） |

验收判据（`tools/e2e_tiny.ps1` 里的 A1/A2/B1–B4）：生成版汇编成功、参考版汇编成功、
两者引擎退出码 0、输出长度相同、**逐字节差异 0**、sha256 相同 —— 全部 PASS。

复现：

```powershell
cd D:\czsai\sparrow
cargo build
pwsh -NoProfile -ExecutionPolicy Bypass -File tools\e2e_tiny.ps1
```

## 5. CLI（任务书第四部分）

```text
sparrow model.sparrow                      # 架构级 .spar（默认）
sparrow model.sparrow --backend vml -o out.vml --args out.args
sparrow model.sparrow --emit-ast-json
sparrow model.sparrow --emit-tokens
sparrow model.sparrow --sha256
sparrow model.sparrow --backend lib        # V0.1.2：显式报"未实现"而不是"未知后端"
```

**与任务书原命令的差异（如实说明）**：任务书写的是 `--backend vml -o model.vmb`，
即前端一步产出 `.vmb`。V0.1.1 没有这么做，原因是 `.vmb` 必须由**既有汇编器**产出，
而汇编器还需要用户提供全部权重 `.bin`（Sparrow 是架构描述语言，不产生权重）。
所以前端只出 `.vml` + `.args`，由驱动脚本 `tools/e2e_tiny.ps1` 把清单喂给汇编器。
前端自己 fork 汇编器当然也能做（"不改汇编器"≠"不能调用"），但那属于打包体验，
不影响端到端是否成立；要的话在 V0.1.2 加一个 `--asm <path-to-vmlm-asm>` 即可。

## 6. 两种后端方向

| | 路径 1：编译为 VML | 路径 2：拼装库（Rust/C） |
|---|---|---|
| 状态 | **✅ V0.1.1 完成** | ⛔ V0.1.2 |
| 链路 | `.spar` → `.vml` → 汇编器 → `.vmb` → 引擎 | `.spar` → 源码 → rustc/cc → `.a`/`.dll`/`.exe` |
| 用途 | 研究、服务端、开发期、跨平台 | 部署、嵌入式、零运行时 |
| 代价 | JIT 启动开销（实测 24 层前向 ≈0.14 s/步） | 换模型必须重编、每平台各编一次 |

共享前端不变：`.sparrow → AST → 语义检查 → .spar IR`，两个后端从 `.spar` IR 分叉。
后端 2 的生成结构已定（模型定义 → `struct` + `impl`；权重 → `include_bytes!` 或内嵌数组；
前向 → `fn forward()`；反向 → `fn backward()`），V0.1.2 实现。

架构约束复核：后端 1 没有改汇编器（只调用），没有改引擎（只产出文本）；
两个后端的数值一致性要求也已就绪 —— 路径 1 的判据已经是"与独立指令路径逐字节相同"。

## 7. 边界判断（都写进源码注释）

1. **无 `连接` 时按声明序成链**。Level 0/1 没有连接语句；`连接` 只是显式路由。
   有 `连接` 但没覆盖全部层 → `E102`（两个链首），不猜。
2. **`重复 n` 展开，不生成循环**。24 层 → 24 组指令（可编、可跑，`.vmb` 1104 B 量级的
   例子见测试）。指令数与层数线性相关，超大层数要等 V0.1.2 的 `LOOP_CNT`。
3. **稀疏层拓扑固定为 `(j*K+t) mod D`**，不接 `conn_sparse` 的窗口拓扑：V0.1.1 的目标是
   端到端跑通，接训练侧的连接表排在 V0.1.2（那里还要处理 top-k 选择）。
4. **中文实例名在 opcode 层转 ASCII**；`;` 注释里的中文保留（汇编器会剥掉注释）。
5. **模型输入固定 `T=1`**：序列/批量长度在架构 IR 里没有声明，V0.1.1 用单 token。
6. **`稀疏`/`线性` 的输入维度靠链上前驱传播**；链首是 `线性` 且没写 `in` → `E103`。
7. **注意力/自定义算子/裸指令一律显式拒绝**（`E104`/`E101`），并在提示里给出排期，
   不做"猜一个近似展开"。

## 8. 开发中被抓出来的坑（全部已修并留回归）

| # | 现象 | 根因 |
|---|---|---|
| 1 | 生成版引擎 `exit=9`：`dst must be [n_neurons(,1)]` | `SPARSE_MATVEC` 的 dst 必须是**列向量**；我原以为"按扁平下标写"就能用 `[1,D]`，引擎有显式形状守卫 |
| 2 | 想用 `RESHAPE` 转行向量 → 汇编期硬失败 | 汇编器把 `RESHAPE` 列在 `UNIMPLEMENTED_OPS`；改用实现过的 `TRANSPOSE` |
| 3 | `EMB` 不写 `SHAPE` → 输出槽 0x0 | 汇编器形状推断表里没有 `OP_EMB` |
| 4 | `.ps1` 中文全变乱码、脚本语法报错 | Windows PowerShell 5.1 按 ANSI 读**无 BOM** 的 `.ps1`；加 UTF-8 BOM + 用 pwsh 跑 |
| 5 | 测试里 `模型 m:` 报 E007 | `M` 是 `model` 的关键字别名（第九部分），保留字当模型名 |
| 6 | 重复层常量数对不上（76 vs 75） | 我的期望值漏算了嵌入表 |

## 9. 遗留项

1. **槽位不复用**（任务书要求）→ V0.1.2。
2. **重复是展开不是运行时循环** → V0.1.2 用 `LOOP_CNT` + `LAYERB`。
3. **汇入（一个层多个前驱）不支持**（`E100`）→ V0.1.2 的 `EWADD`/`CONCAT`。
4. **CSR 拓扑是固定规则**，未接 `conn_sparse` 的连接表与 top-k 选择 → V0.1.2。
5. **注意力块未展开**（`E104`）：需要 QKV GEMM / 拆头 / 掩码 / softmax / 合并，几十条 opcode。
6. **`.vmb` 由驱动脚本产出**，前端不出 `.vmb`（见 §5 的差异说明）。
7. **权重必须外部提供**：Sparrow 只描述架构；`.args` 清单 + `tools/gen_e2e_tiny.py`
   是当前提供权重的方式（生成器用固定 seed，可复现）。
8. **模型输入 `T=1`**；序列长度/批量尚未进入架构 IR（没有对应的 Sparrow 语法）。
9. **`--backend lib`（路径 2）只有 CLI 占位**，报"排在 V0.1.2"。
10. V0.1 前端的所有遗留项（中文数字紧贴写法、第七部分省略层类型、`@名字` 歧义…）
    保持不变，见 `REPORT.md` §8。
