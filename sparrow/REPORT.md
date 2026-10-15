# Sparrow V0.1 —— 神经网络架构描述语言 · 文本前端报告

> **V0.1.1 改名说明**：本文是 V0.1 的记录。V0.1.1 起，前端的架构级输出
> 后缀由 `.vml` 改为 `.spar`（`.vml` 让给 opcode 序列）。文中出现的 `.vml`
> 一律读作 `.spar`。V0.1.1 的报告见 `REPORT_V011.md`。

> 交付物：`D:\czsai\sparrow`（零依赖 Rust crate，`cargo test` 36 项全绿）
> 管线：`源码 → lexer → parser → sema → codegen → .vml`
> 本文按任务书第十四部分的 8 项组织；第九部分另附**边界判断**与**遗留项**。

## 0. 交付清单与验收总表

| 文件 | 行数 | 职责 |
|---|---|---|
| `src/ast.rs` | 359 | `Span`/`Ident`/`Ast`/`Item`/`Stmt`/`Expr` + `AstStats` |
| `src/keywords.rs` | 795 | 结构关键字表(21 类) + 层类型表(6) + 属性键表(16) + 编辑距离建议 |
| `src/diag.rs` | 302 | 行列定位、显示宽度对齐、诊断渲染/JSON |
| `src/lexer.rs` | 837 | 标点归一、缩进宽容、注释、中文数字、裸指令保真 |
| `src/parser.rs` | 1360 | 五种模式 + 四套块机制 + 四种书写顺序 |
| `src/sema.rs` | 485 | 层类型解析、参数绑定、引用解析、缺参校验 |
| `src/codegen.rs` | 146 | canonical `.vml` 输出（纯函数、字节可复现） |
| `src/json.rs` | 209 | `--emit-ast-json` |
| `src/sha256.rs` | 125 | 零依赖 SHA-256（验收自检用，已与 `Get-FileHash` 交叉验证） |
| `src/main.rs` | 167 | CLI：`--emit-vml`/`--emit-ast-json`/`--emit-tokens`/`--sha256`/`-o` |
| `tests/acceptance.rs` | 662 | 验收 1–12 全部断言 |
| `examples/*.sparrow` | 23 个 | Level 0–6、A–G、tiny 三份、5 个错误样例 |

**源码 4489 行 / 测试 618 行 / 测试 36 项（21 单元 + 15 验收）全绿。**

| # | 验收项 | 结果 | 证据 |
|---|---|---|---|
| 1 | Level 0–6 全部解析通过 | ✅ | `acceptance_1_levels_0_to_6` |
| 2 | 纯中文 7 示例全部解析通过 | ✅（含 1 处边界，见 §4） | `acceptance_2_*` |
| 3 | tiny_cn/en/mixed 三次 sha256 相同 | ✅ `86DF36FD…` | `acceptance_3` |
| 4 | 缩进宽容（2/4/8/Tab + 宽度不统一） | ✅ 5 种写法同 `.vml` | `acceptance_4` |
| 5 | 标点宽容（中文/英文/无标点） | ✅ 同 `.vml` | `acceptance_5` |
| 6 | 块机制宽容（冒号缩进/开始结束/{} /【】/一行分号） | ✅ 5 套同 `.vml` | `acceptance_6` |
| 7 | 中文标识符（嵌入层/主干/x1/_tmp/入/干/出） | ✅ 未被关键字吞 | `acceptance_7` |
| 8 | 中文标点作分隔符 → 3 标识符列表 | ✅ 5 种分隔符 | `acceptance_8` |
| 9 | 关键字冲突场景报错清晰 | ✅ E007 + 转义提示 | `acceptance_9` |
| 10 | ≥20 组"语义相同写法不同"同 `.vml` | ✅ **20 组 / 56 次比较全同** | `acceptance_10` |
| 11 | 错误报告（未定义层/参数缺失/括号不匹配） | ✅ 行列 + 上下文 + 插入符 + 提示 + 候选 | `acceptance_11` |
| 12 | `--emit-ast-json` 输出可读 AST | ✅ 括号配平、中文不转义 | `acceptance_12` |

---

## 1. AST 结构摘要

严格按第八部分，**只加机制、不改形状**：

```rust
pub struct Span { pub start: u32, pub end: u32 }          // 源码字节区间
pub struct Ident { pub name: String, pub span: Span }     // 原文保留，不做转写

pub enum Ast { Program { items: Vec<Item> } }

pub enum Item {
    ModelDef { name: Ident, body: Vec<Stmt>, span: Span },
    LayerDef { name: Ident, params: Vec<Ident>, kind: Ident, attrs: Vec<(Ident, Expr)>, span: Span },
}

pub enum Stmt {
    Instantiate { name: Ident, kind: Ident, args: Vec<Expr>, span: Span },
    Repeat      { body: Box<Stmt>, count: Expr, span: Span },        // 结构化，不是文本替换
    Connect     { path: Vec<Ident>, span: Span },
    Forward     { input: Ident, output: Ident, span: Span },
    CustomOpDef { name: Ident, opcode: Expr, args: Vec<Ident>, span: Span },
    RawOp       { opcode: Ident, operands: Vec<Expr>, span: Span },
}

pub enum Expr { Int(i64), Float(f64), Ident(String), Named { key: Ident, value: Box<Expr> }, Arrow(Vec<Expr>) }
```

三条设计说明（都写在源码注释里）：

1. **不往 AST 塞 `explicit_name` 标志位**。没有显式名字时 `Instantiate.name` 直接等于 `kind`
   原文（`嵌入 512` → `name=kind=嵌入`）。codegen 用 `canonical(name)==canonical(kind)`
   判定"名字是默认的"，于是中英对照能落到同一个实例名 `EMBED`。AST 保持第八部分原样。
2. **`Repeat` 是结构化节点**：`重复24次` / `x 24` / `* 24` / `24层` / `叠 24 个`
   产出**同一个** `Stmt::Repeat{count:Int(24)}`，`.vml` 里是 `REPEAT 24 LAYER …`。
3. **参数位置/命名在 AST 里就分开**：位置参数是裸 `Expr`，命名参数是 `Expr::Named`。
   归一化（位置 → 规范键）发生在 sema，不在 AST。

---

## 2. 关键字表（中英 + 缩写全集）

结构关键字（第九部分的表 + 示例里实际出现的虚词，共 21 类）：

| 类 | 别名 |
|---|---|
| model | 模型 / 网络 / model / net / network / **M** |
| layer | 层 / 定义 / layer / def / **L** |
| connect | 连接 / 连 / 接 / 串 / 到 / `->` / `→` / connect / conn / link |
| forward | 前向 / 预测 / 推理 / forward / fwd / infer / predict |
| repeat | 重复 / 复用 / 叠 / 堆 / 循环 / repeat / rep / **x** / `*` / 做 |
| to | 到 / 至 / `->` / `→` / to |
| and | 和 / 与 / 以及 / `、` / `,` / `，` / `;` / `；` / and |
| start | 开始 / 起 / `{` / `【` / begin / start |
| end | 结束 / 止 / 完 / `}` / `】` / `。`※ / end / stop |
| colon | `:` / `：` | 
| semicolon | `;` / `；` |
| comma | `,` / `，` / `、` |
| lparen / rparen | `(` `（` `[` ／ `)` `）` `]` |
| op | 算子 / 自定义算子 / 算子定义 / op / operator |
| opcode | opcode / 操作码 / 指令 / 指令号 |
| args | args / 参数 / 形参 / arity |
| as | 用 / 为 / 把 / 以 / as / with / using |
| from | 从 / 自 / from / src |
| then | 再 / 然后 / 接着 / then / next |
| filler | 的 / 个 / 次 / 就 / 了 / 让 / 使 / 将 / 一下 |

※ `。` 的处理见 §9 边界判断 3。

层类型表（**仅精确匹配**，见 §9 边界判断 1）：

| canonical | 别名 | 位置参数绑定 |
|---|---|---|
| EMBED | 嵌入 / 嵌 / 词嵌入 / embed / embedding / emb | 1 个→`dim`；2 个→`vocab,dim` |
| SPARSE | 稀疏 / 稀 / 稀疏层 / sparse / sp | 1 个→`dim`；2 个→`dim,k0` |
| LINEAR | 线性 / 线 / 全连接 / linear / lin / dense / fc | 1 个→`out`；2 个→`in,out` |
| NORM | 归一化 / 层归一化 / 范数 / norm / layernorm / rmsnorm / ln | `dim` |
| ACT | 激活 / 激活函数 / act / activation | `act` |
| ATTN | 注意力 / 自注意力 / 多头注意力 / attn / attention / mha | 1 个→`dim`；2 个→`dim,heads` |

属性键表（16 个 canonical，含中英缩写）：`vocab`(词表/词汇表/vocab/v)、`dim`(维度/维/dim/d/hidden_size)、`k0`(连接/连/连接数/稀疏度/k0/k)、`in`(输入/入/in)、`out`(输出/出/out)、`hidden`、`heads`、`act`、`norm`、`eps`、`layers`(层数/深度/depth)、`seq`、`batch`、`bias`、`tie`、`scale`。

**匹配策略分两档**（源码 `keywords.rs` 模块头有说明）：
- 结构关键字 / 属性键：精确 + **唯一前缀**（`连`=`连接`=`conn`=`connect`；`模`→`模型`）；
  前缀资格有闸：中文 ≥1 字，ASCII ≥3 字符 —— 否则 `e`/`t`/`o`（示例 F/G 的实例名）
  会被 `end`/`then`/`op` 的前缀吞掉。
- 层类型：**只精确**。因为验收 11 要求 `Spars` 报"未定义的层类型"并建议 `Sparse`，
  而 `Sparse`.starts_with(`Spars`) 为真 —— 前缀匹配会把错误吞掉。

---

## 3. Level 0–6 与生成的 .vml

`.vml` 语法（前端产出，确定性纯函数）：

```
; sparrow v0.1 — canonical vml (generated by sparrow front-end)
; model: <名>
.model <名>
  LAYER   <实例名> <KIND> k=v ...
  REPEAT  <n> LAYER <实例名> <KIND> k=v ...
  OPCALL  <实例名> <算子>
  CONNECT <a> <b> <c>
  FORWARD <in> <out>
  COP     <名> opcode=<expr> args(<a>,<b>)
  RAW     <opcode> <操作数> ...
.end
```

| Level | 输入要点 | 生成的 .vml | sha256(前 16) |
|---|---|---|---|
| 0 | `嵌入 512`（无模型头 → 隐式 `main`） | `LAYER EMBED EMBED dim=512` | 2771936D5AD4E808 |
| 1 | `模型 小:` + `嵌入 维度=512` | 同上（模型名 `小`） | 63497A7E644B8B3D |
| 2 | 三层 + `连接 嵌入 稀疏 线性` | +`SPARSE dim=512 k0=8`、`LINEAR out=4054`、`CONNECT` | 1EDED28BC399B4F6 |
| 3 | `嵌 e 512`（**类型在前**）+ `连 e t o` | `LAYER e EMBED`、`t SPARSE`、`o LINEAR`、`CONNECT e t o` | B2F4144229F945C7 |
| 4 | `定义 稀疏块 用 维 连 为:` + `重复24次` | `REPEAT 24 LAYER t SPARSE dim=1536 k0=8` + `FORWARD e o` | 284630BF8FE47EED |
| 5 | `算子 我的激活 = opcode(200) args(2)` + 单独一行 `我的激活` | `COP 我的激活 opcode=200 args(arg0,arg1)`、`OPCALL`、`CONNECT EMBED SPARSE 我的激活 LINEAR` | 7634DB559C24ECD1 |
| 6 | 4 条 `@op` | `RAW load r0 cpool+0` / `RAW gemm r0 x r0 h` / … | C42D0E8913E2E8A8 |

Level 4 三个片段（可对照上面第 4 行）：

```vml
; model: 大
.model 大
  LAYER   e EMBED dim=1536
  REPEAT  24 LAYER t SPARSE dim=1536 k0=8
  LAYER   o LINEAR out=4054
  CONNECT e t o
  FORWARD e o
.end
```

**交叉印证**：`examples/level4.sparrow` 与 `examples/cn_f.sparrow` 是两份独立文件
（描述同一个模型），产出同一个 sha256 `284630BF8FE47EED…`。

---

## 4. 七个纯中文示例的解析结果

| 示例 | 写法特征 | 结果 | sha256(前 16) |
|---|---|---|---|
| A | 英文标点混入 | ✅ | 8D9364034EAE1F9B |
| B | 全中文标点 | ✅ | 8D9364034EAE1F9B（与 A 同） |
| C | 无缩进 + `开始`/`结束` | ✅ | 8D9364034EAE1F9B（与 A 同） |
| D | 一行写完（分号） | ✅ | C4F46C985681362A（**内容不同，见下**） |
| E | 缩写 + 自然语序 `把…连到…再连到…` | ✅ | 0379384E8E20422E |
| F | `定义` + `重复24次` | ✅ | 284630BF8FE47EED |
| G | 极端宽容（`网络 big：e 嵌入 1536` / `x 24`） | ⚠️ **单独不可编译**，见下 | — |

三条必须如实说明的边界（任务书"7 个示例必须解析成同一个 AST"逐字看**不成立**）：

1. **A/B/C 三者确实同 AST（去 span 后逐字段相同）**，也同 `.vml`；测试 `acceptance_2b` 断言了这一点。
2. **D 与 A 的内容不同**：D 写的是 `线性 1536 4054`（两个位置参数 → `in=1536,out=4054`），
   而 A 写的是 `线性 输出=4054`（只有 `out`）。`.vml` 因此多出 `in=1536`。
   若把 D 写成 `线性 4054`，它与 A 逐字节相同（验收 6 的"一行写完"项已用等价内容验证过）。
3. **G 单独不成立**：G 用了 `稀疏块` 但没有 `定义`，它隐含依赖 F 的定义。
   逐字保留的 G 能**正常解析**（AST 正确），sema 会报
   `E002 未定义的层类型 "稀疏块"`；我另存了自洽版本 `examples/cn_g2.sparrow`
   （补上 F 的定义，其余逐字不动），它完整编译通过。

---

## 5. 20 组变体测试通过率

`cargo test --test acceptance acceptance_10_variant_groups -- --nocapture`：

```
变体组 20 组，比较 56 次，全部同 .vml
test result: ok. 1 passed; 0 failed
```

| 组 | 主题 | 写法数 |
|---|---|---|
| 01 | 隐式/显式模型头（含 `开始/结束`） | 3 |
| 02 | 位置/命名参数 | 3 |
| 03 | 两个位置 = 两个命名（顺序任意） | 4 |
| 04 | 层类型缩写/大小写（嵌/Embed/EMBED/emb） | 4 |
| 05 | 稀疏层五种写法 | 5 |
| 06 | 线性层输出维度（线性/输出/Linear/线/dense） | 5 |
| 07 | 连接：关键字/箭头/逗号/分号/`connect(...)` | 5 |
| 08 | 自然语序连接（`把…连到…再连到…`） | 3 |
| 09 | 中文复合连接词 `连到` | 3 |
| 10 | 前向：`前向`/`forward(...)`/`从…前向到`/`前向`/`forward` | 5 |
| 11 | 重复：`重复24次`/`x 24`/`* 24`/`24层`/`重复 二十四 次`/`叠 24 个` | 6 |
| 12 | 层定义展开 = 直写（含 `定义 f(维,连):` 形式） | 3 |
| 13 | 注释与空行（`#`/`//`/`/* */`/`。;`） | 3 |
| 14 | 模型头：`:`/无冒号/`{}`/`开始`/`【】` | 5 |
| 15 | 裸指令：逗号/括号/空白分隔 | 3 |
| 16 | 自定义算子两种声明 | 3 |
| 17 | 标点体系：中文/英文/无 | 3 |
| 18 | 数值写法（`1e-5` = `0.00001` = `1E-5`） | 3 |
| 19 | 属性键语言/缩写（维度/dim/维/D） | 4 |
| 20 | 关键字大小写混写（`MoDeL`/`EmBeD`/`CoNnEcT`） | 3 |

**通过率 56/56 = 100%。**

---

## 6. 五个错误示例的报错输出

### 6.1 未定义的层类型（验收 11 主例）

```
错误[E002]：未定义的层类型 "Spars"
  位置：examples\err_undefined_kind.sparrow:4:7
   4 |   t = Spars(1536, 8)
     |       ^^^^^
  提示：是不是想写 "Sparse"？
  候选：Sparse, Embed, Linear, Norm, Activation, Attention
```

### 6.2 参数缺失

```
错误[E003]：层类型 "稀疏块" 需要 2 个参数（维, 连），只给了 1 个
  位置：examples\err_missing_arg.sparrow:7:7
   7 |   t = 稀疏块(1536)
     |       ^^^^^^
  提示：例：`稀疏块(<维>, <连>)`
  候选：维, 连
```

### 6.3 括号不匹配（指出**开**括号在哪）

```
错误[E004]：括号不匹配：`(` 没有对应的 `)`
  位置：examples\err_paren.sparrow:4:10
   4 |   t = 稀疏块(1536, 8
     |             ^
  提示：在文件末尾补一个 `)`
```

### 6.4 关键字当标识符（验收 9）

```
错误[E007]："m" 是关键字（model），不能作为实例名
  位置：examples\err_keyword.sparrow:3:3
   3 |   m = 嵌入 512
     |   ^
  提示：改用 `@m` 或 `"m"` 转义，或者换个名字
  附注：第四部分：关键字保留，除非用 @ 或引号转义
```

### 6.5 引用了未定义的层

```
错误[E006]：连接/前向引用了未定义的层 "注意力"
  位置：examples\err_undefined_ref.sparrow:5:9
   5 |   连接 嵌入 注意力
     |             ^^^^^^
  候选：EMBED, LINEAR
```

其它**已实现且可达**（源码里逐条核对过引用次数）的错误码：
`E003`(重复次数非正/非整数、层类型零参数)、`E004`(多余右括号)、`E005`(意外 token/空连接/缺 `前向`)、
`E006`(引用未定义层)、`E007`(关键字当标识符)、`E010`(词法：注释/字符串未闭合、非法字符)、
`E013`(文件里没有模型)、`E014`(模型为空)、`E015`(多模型，仅警告)。

注：任务书第十部分示例里的 `ParseError::UnrecognizedPattern` 我合进了 `E005`/`E002` ——
没有单列 `E001`，因为"裸词开头"在当前语法里总能落到某种实例化，最终由 sema 给出更准的
`E002 未定义的层类型`；保留一个永不触发的错误码只会误导使用者。

排版细节：插入符按**显示宽度**对齐（中文占 2 列），列号按**字符**计数 —— 两者都单独测过
（`chinese_column_alignment`）。

---

## 7. AST JSON 示例

`sparrow examples/level2.sparrow --emit-ast-json`：

```json
{
  "kind": "Program",
  "items": [
    {
      "item": "ModelDef",
      "name": "小",
      "span": {"start":41,"end":128},
      "body": [
        {
          "stmt": "Instantiate",
          "name": "嵌入",
          "kind": "嵌入",
          "span": {"start":55,"end":65},
          "args": [
            {"expr": "Int", "value": 512}
          ]
        },
        {
          "stmt": "Instantiate",
          "name": "稀疏",
          "kind": "稀疏",
          "span": {"start":68,"end":83},
          "args": [
            {"expr": "Int", "value": 512},
            {"expr": "Named", "key": "k0", "value": {"expr": "Int", "value": 8}}
          ]
        }
      ]
    }
  ]
}
```

`Stmt::Repeat` 在 JSON 里也保持结构化：

```json
{"stmt": "Repeat", "count": {"expr": "Int", "value": 24},
 "body": {"stmt": "Instantiate", "name": "t", "kind": "稀疏块", "args": [...]}}
```

测试 `acceptance_12` 断言：键名齐全、括号配平、中文不转义（`"name": "大"` 原样可读）。

---

## 8. 遗留项

1. **中文数字只在"数字位置"生效**。词法层不拆词（`二十四` 是字母，拆了 `一致`/`一年`
   就碎了），所以 `重复 二十四 次` ✅、`重复24次` ✅，但 **`重复二十四次`（全无空格）❌**。
   第十三部分本身就写"不做中文数字、更推荐阿拉伯数字"，故按此取舍。
2. **第七部分示例的"省略层类型"写法不支持**。`嵌入层 词表=4054 维度=1536`（层类型位置
   放的是名字）会报 `E002`；需要写 `嵌入层 = 嵌入 词表=…` 或 `嵌入 词表=…`。
   我没有做"按名字猜层类型" —— 那是不可靠的启发式（`主干层`/`输出层` 猜不出来）。
3. **用户定义层类型在"名字在前且不带括号"时认不出来**：`t 稀疏块(1536, 8)` ✅、
   `t = 稀疏块(1536, 8)` ✅、`t 稀疏块 1536 8` ❌（parse 期还不知道 `稀疏块` 是类型）。
4. **`@名字` 只在与 `=`/`:` 相邻时视为转义标识符**，否则视为裸指令（Level 6 的 `@load`）。
   于是 `@call foo=1` 会被当成裸指令而不是"call 的实例化"。
5. **裸指令的操作数里括号只作分隔符**：`@gemm(r0, x, r0)` ≡ `@gemm r0 x r0`，
   原始的括号文本不保留（操作数文本其余部分逐字保留，`cpool+0` 仍是一片）。
6. **多模型文件只输出第一个模型**，其余给 `E015` 警告（`--emit-ast-json` 不受影响）。
7. 未做（任务书第十三部分明确排除）：类型系统、宏展开、泛型、控制流、积木前端、
   以及任何对 `trans`/运行时的改动。`.vml` 是前端自定的 canonical 文本格式，**没有**
   与 `vmlm` 的汇编器对接（需要的话是下一步）。

---

## 9. 边界判断（我做了选择，都记在源码注释里）

1. **容错分两档**：结构关键字/属性键允许唯一前缀；层类型只允许精确。
   理由：前缀匹配会让 `Spars` 变成合法的 `Sparse`，直接违背验收 11。
2. **前缀资格闸**：中文 ≥1 字、ASCII ≥3 字符。否则第九部分的单字母别名
   （`x`=repeat、`M`=model、`L`=layer）会把 `e`/`t`/`o` 这类正常实例名吞掉。
3. **`。` 是分隔符，不是块结束标记**。第九部分把 `。"` 列在 `end` 类，第四部分说中文标点
   "绝不"是标识符、"只作为分隔符"。取第四部分（"绝不"级约束 > 枚举表）。
4. **中文复合词只在"期待关键字的位置"切分**：`连到`=`连`+`到`、`前向到`=`前向`+`到`。
   不能放进 lexer —— `嵌入层` 同样能切成 `嵌入`+`层`，但四/七部分要求它是一个标识符。
5. **裸标识符列表用"整行前瞻"判定**。第二/三部分对 `;` 的定位冲突（元素分隔符 vs 语句
   分隔符）：若整逻辑行只有 ≥2 个词 + 分隔/连接词，则整行是一个 `Connect` 路径。
6. **缩进是建议性的**。缩进块一直延续到"文件结束 / 块结束标记 / 下一个 `模型`·`层` 条目"，
   多余的 Indent/Dedent 一律吞掉。这是"缩进错了不报错"最直接的实现。
7. **`定义` 模板不进 `.vml`**。它是编译期糖：展开后的 `LAYER` 行已含全部信息；若模板也
   进 `.vml`，"定义式写法"与"直写式写法"就不同哈希，直接违背验收 10。模板在 AST JSON 里可见。
8. **隐式模型名固定为 `main`**（语言中立），这样 `嵌入 512` 与 `模型 main: 嵌入 512` 同 `.vml`。
9. **`m = 嵌入 512` 不会退化成新的模型头**：`M` 是 model 的合法别名，块循环与顶层循环都用
   "后面是否跟 `=`/`(`/`:`"区分"条目头"与"恰好撞上关键字的标识符"，于是它报的是
   `E007 关键字不能作标识符`，而不是莫名其妙的"需要模型名"。
10. **`@` 的双重身份**：`@名字` + `=`/`:` → 转义标识符；否则 → 裸指令（Level 6）。

---

## 10. 开发过程中被测试抓出来的真 bug（全部已修并留回归）

| # | 现象 | 根因 |
|---|---|---|
| 1 | `重复24次` 粘成一个标识符 | `is_ident_continue` 含数字；中文后接数字串要切词 |
| 2 | `loc()` panic（字节落在汉字中间） | 未退回字符边界 |
| 3 | 缩进后第一个字符被吃掉（`嵌入`→`入`） | lexer 主循环在缩进处理**之前**绑定了 `c`，`match` 用了过期字符 |
| 4 | `模型 小:` 报"需要参数值，实际是 `:`" | `expect_ident` 只返回不推进游标 |
| 5 | 示例 D 后半段掉进隐式模型 | 同行内联块一遇 `;` 就 break（`;` 在行内也是语句分隔符） |
| 6 | `线性 4054` 变成 `in=线性 out=4054` | 同 #4 的连带效应 |
| 7 | `连接 嵌入;稀疏;线性` 只剩 1 个元素 | 连接路径里 `;` 被当成语句结束（第九部分同时归 `and`/`semicolon`） |
| 8 | `m = 嵌入 512` 报"需要模型名" | 块循环把 `m`（model 别名）当成下一个模型头 |
| 9 | `t 稀疏块(1536,8) x 24` 报未定义层类型 | 用户定义类型在"名字在前"时认不出来 |
| 10 | `forward(a, b)` 报"forward 是关键字" | 我自己的 `!lookahead_is_call()` 守卫把 `forward(...)` 挡在 parse_forward 之外 |
| 11 | `@m` 转义无效 | 转义符被吃掉后信息丢失，`expect_ident` 仍按关键字拒绝 |
| 12 | `算子 f … 参数 arg0 arg1` 把下一行的 `f` 吃进形参 | 形参循环里 `skip_separators()` 跳过了换行 |
| 13 | `{ }` 块内缩进导致 `这里需要层类型，实际是缩进` | 块循环没吞多余 Indent/Dedent |
| 14 | SHA-256 的"中文"期望值不符 | **测试数据错了**，我的实现是对的（用 `Get-FileHash` 独立核对确认） |

---

## 11. 复现方式

```powershell
cd D:\czsai\sparrow
cargo test                                  # 36 项全绿（21 单元 + 15 验收）
cargo run -- examples\level4.sparrow        # 出 .vml
cargo run -- examples\tiny_cn.sparrow --sha256
cargo run -- examples\level2.sparrow --emit-ast-json
cargo run -- examples\err_undefined_kind.sparrow   # 报错到 stderr，退出码 1
```

无第三方依赖（`Cargo.toml` 的 `[dependencies]` 是空的），因此离线可构建、可复现。
