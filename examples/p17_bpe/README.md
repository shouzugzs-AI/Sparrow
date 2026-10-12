# P17-P3: BPE 分词器（mmap 外部词表 + 纯 Rust 手写解码）

本目录是 task-4 的产物与实测证据。**所有数字都是跑出来的**，命令与原始输出在下面。

## ⚠️ 本阶段最重要的一条结论（请先读这条）

> **round-trip 逐字节相同是必要条件，不是充分条件。**

本阶段有一个 bug 是 round-trip **结构上不可能**发现的：

```
word "engine":  fast(错) = [259,445,291] = "en" "gi" "ne"
                correct  = [628,101]      = "engin" "e"
```

两种切分**解码回完全相同的字节**，所以 `text → ids → text` 双双通过。
只有与一个**结构不同的实现**（这里是 O(n²) 参考编码器）对拍才发现。

⇒ 判据必须来自**两个不同来源**。同一个实现自证、或只看 round-trip，
对「切分质量」这一类错误是盲的。round-trip 覆盖的是**存储与还原的完整性**，
覆盖不了「编出来的 id 序列是不是对的」。

这条与 infer-dev 在 P1 发现的「k=8 vs k=k0 逐位比较是恒假判据」是同一枚硬币的两面。

| 文件 | 是什么 |
|---|---|
| `bpe_vocab.vmb` | 词表产物，5352 B，**外部文件**，运行时只读 mmap |
| `bpe_vocab.json` | 词表元数据（含展开尺寸/最长 piece/操作数统计）+ 4 条 golden case 的 id 序列 |
| `corpus.txt` | 训练语料 37800 B（round-trip 判据就是在它上面做的） |
| `bpe_size.json` | 隔离 target-dir 体积实测（**当前只含 arm A/B** —— 见 §1 的现状标注） |
| `bpe_xcheck.json` | Rust×Python 交叉校验 + 15 条 can-fail 对照 |
| `bpe_roundtrip.json` / `.log` | 验收链路 text→id→logits→text 的判据清单 |

复现全部证据：

```
python tools/p17_bpe_vocab.py --check      # 生成词表 + 8 条自控制
python tools/p17_bpe_xcheck.py             # 交叉校验 + 守卫 can-fail 对照
python tools/p17_bpe_roundtrip.py          # 验收链路 + 改一个 id 的对照
python tools/p17_bpe_size.py               # 隔离 target-dir 体积实测
```

---

## 1. 引擎体积增量：**实测 0 B**

硬要求是「BPE 模块必须放 feature 后面，且对默认引擎贡献 0 B（隔离 target-dir 实测）」。
每个 arm 用**独立的临时 target-dir + 独立 CARGO_TARGET_DIR**，并且从 `runtime/` 构建
（no_std 链接参数在 `runtime/.cargo/config.toml`；从仓库根构建会 `LNK1561`）。

| arm | 构建 | 体积 |
|---|---|---|
| A | 默认引擎，`bpe` feature **关闭** | **74752 B** |
| B | 同一份源码，`--features bpe` | **74752 B** |
| C | **正对照**：`--features train` | **96256 B** |

```
[PASS] bpe feature contributes 0 B to the default engine (A=74752 B=74752 delta=+0)
[PASS] positive control MOVES the size (train A=74752 C=96256 delta=+21504)
[PASS] default engine within budget (A=74752 <= 84480, headroom 9728)
```

> ⚠️ **落地证据的现状（如实标注，勿误读）**：`bpe_size.json` 目前**只含 arm A/B**
> （74752 / 74752），因为最后一次跑的是 `--quick`（它刻意跳过正对照，见脚本注释）。
> 上表 arm C 的 **96256 B / +21504 B** 来自**先前的完整三次构建运行**，是真实实测值，
> 但**当前那份 JSON 里没有它** —— 我没有为了补上它而在 lead 的静默窗口内再触发构建。
> 要恢复完整三 arm 证据，跑一次（不构建、仅核对时不要跑）：
> `python tools/p17_bpe_size.py`（**不带** `--quick`）。
> `--quick` 覆盖同一个 JSON 是脚本的一个缺陷，已记在此处而不是掩盖。

**为什么 arm C 不是装饰**：A==B==C 就说明测量装置坏了，那么「0 B」什么也证明不了。
C 必须动，A/B 的 0 才是证据。这里是 +21504 B，装置有效。

**为什么必须隔离 target-dir**：在同一 target-dir 里连构，cargo 会复用缓存产物，
第二次构建可能报出**第一次的体积** —— 一个恰好为 0 的假 delta。这个失败模式是静默的，
而且看起来像成功。`tools/build_all.ps1` 里有这条 METHOD WARNING 的原始记录。

> 注：`runtime/Cargo.toml` **没有**转发 `bpe` feature，所以 `--features bpe` 从 runtime/
> 直接构建是**不被接受**的 —— arm B 在 runtime/ 的一次性副本里加上转发行来构建。
> 被测源码始终是**出厂的** `core/src/bpe.rs`；只有 manifest 在 arm 之间不同，这个差异是
> 显式声明并记录在脚本里的。这也同时证明了 `bpe` feature 是**正交**的：
> `--features bpe` 单独开能编译过（`default-features = false` + 只有 `bpe`）。

---

## 2. round-trip：**逐字节相同**

验收链路：`text → token id → logits → text`。

```
text     : 'VMLM 是一个把 Transformer 权重预先汇编成指令流的推理引擎。 The engine is no_std and links against kernel32 directly.'
bytes    : 136
sha256   : 481c29d62d9ab0614435dfdf924085dc1bd1496abe51c4d550f78e1a25b9416a

[PASS] stage A: Rust ids == Python ids (independent encoders)   rust=52 python=52, equal=True
[PASS] stage C: Rust round trip is byte-identical               n_bytes=136 (input 136) exact=1, bytes_equal=True
[PASS] stage D: Python decoder agrees byte for byte             python=136 B
[PASS] corpus round trip (Rust): 37800 B, byte-identical        n_ids=15080 sha_in=c769f7e9f390aa05 sha_out=c769f7e9f390aa05
[PASS] corpus round trip (Python, independent)                  15080 ids -> 37800 B
```

**判据两端来自不同来源**：Rust 是出厂模块，Python 是独立实现（自己的训练器、自己的
编码器、自己的解码器）。只有一端的话，「错误两次仍然自洽」也会通过。

### 2.1 must-fail 对照：改一个 token id ⇒ 输出必须变

```
[PASS] control: mutating ONE token id changes the Rust output   orig id=415 -> 65; output changed=True, len 136 -> 133
[PASS] control: the mutated output is not merely a re-encode    also differs from the Python decoder's output
```

### 2.2 logits 那一段是**占位**的（诚实标注）

任务说 logits 可以先用随机/占位。本阶段**没有**训练好的模型接到这个词表上，所以：

- **B1 恒等模型**：ids 原样通过。这是**退化**的模型，正是它让 round-trip 成为关于
  **分词器**的陈述，而不是关于模型的陈述。
- **B2 占位 logits → argmax → ids**：种好的 PRNG 造一张伪 logits 表，argmax 出 id，
  再送进解码器。这**真实地**跑通了 `ids → logits → ids` 的接线（生成的 id 全部可解码，
  无守卫触发），但 logits 是**占位**的，所以**不期望**恢复原文本 —— 这条判为
  **UNPROVEN**，不当 PASS 报。

---

## 3. 守卫清单（每条 exit(9) 且打印具体数值）

15 条 can-fail 对照全部实测通过（`bpe_xcheck.json`）：

| 守卫 | 触发条件 | 消息 | 对照 |
|---|---|---|---|
| G0 | 文件比头部还短 | `BPE: available bytes below header` | (j) |
| G1 | 长度 != 头部推出的尺寸（**精确相等**） | `BPE: file length disagrees with header` | (a) |
| G2 | `n_merges == 0` | `BPE: n_merges is zero` | — |
| G3 | `merge_bytes != n_merges * MERGE_ENTRY` | `BPE: merge_bytes != n_merges * MERGE_ENTRY` | (b3) |
| G4 | 尺寸溢出 u32 | `BPE: vocabulary file exceeds 32-bit offsets` | — |
| G5 | `n_pieces != 256 + n_merges` | `BPE: n_pieces != 256 + n_merges` | (b) (b2) (h) |
| G6 | 特殊表 code 重复声明 | `BPE: special byte code declared twice` | — |
| G7 | 特殊 piece 的 id >= 256 | `BPE: special piece is a byte-fallback id` | — |
| G8 | 错 magic | `BPE: bad vocabulary magic` | (c) |
| G9 | 版本不对 | `BPE: unsupported vocabulary version` | (d) |
| G10 | 操作数引用了**尚不存在**的 id | `BPE: merge operands must be ids that already exist` | (g) |
| G11 | `merged_id != 256 + 规则下标` | `BPE: merged_id is not 256 + merge index` | (f) |
| G12 | token id 越界（**解码期**） | `BPE: token id out of vocabulary range` | (j2) |
| G13 | 输出超过上限 / 词超过单词语缓冲 | `BPE: encode output exceeds id limit` / `BPE: word exceeds the per-word id buffer` | — |
| G14 | `reserved0 != 0` | `BPE: reserved0 is not zero` | (e) |
| G15 | `special_bytes != n_special * 5` | `BPE: special_bytes != n_special * 5` | (i) |
| G16 | 空输入 | （返回 0，显式记账，非失败） | — |

外加一条**独立**的守卫：`BPE: arena too small to expand the vocabulary` —— 见 §5。

---

## 4. 格式与设计决定

### 4.1 文件格式 `BPE1`

```
off 字段             含义
  0  magic           0x31504542（"BPE1" 小端）
  4  version         == 1（下一版用 "BPE2"，版本漂移会在 magic 上失败）
  8  n_merges        M
 12  n_pieces        V == 256 + M（**不存词表**，只为长度校验而声明）
 16  merge_bytes     == M * MERGE_ENTRY
 20  special_bytes   == n_special * 5
 24  n_special       特殊 token 数
 28  reserved0       必须为 0
32  merge array       M × 8 B = (u16 left, u16 right, u32 merged_id)
      special array   n_special × 5 B = (u8 code, u32 piece_id)
```

### 4.2 **只存合并表，不存词表副本** —— 这就是「不占引擎字节」的落地方式

piece 的字节是**可推导**的：

```
bytes(id) = [id]                          若 id < 256
bytes(id) = bytes(left) ++ bytes(right)   若 id = 256 + i，规则 i = (left, right)
```

存一份词表副本 = **同一件事声明两遍**，而两份声明可以不一致且没人发现（本项目在 S8
上吃过这个亏）。所以文件里只有合并表；`expand_vocab` 自底向上展开一次到调用方给的
arena，稳态解码是「按 id 取 (off,len) + memcpy」：**不递归、不分配**。

实测：665 条规则、921 个 piece、展开后 38189 B（最长单个 piece 长 15120 B）。

### 4.3 操作数是 **u16**，不是 u8

第一版用 u8，理由是「操作数就是字节」—— **错的**：操作数可以是一条**早先的合并结果**，
其 id >= 256。实测 1330 个操作数里 **848 个 >= 256**，最大 919。
u8 会（Python 侧）抛异常 /（Rust 侧）**静默截断**，而截断后的 id 仍然「看起来像个合法数字」。

### 4.4 解码 = **展开到叶子再拼接**，不是「重放」

这是本阶段最重要的设计决定，它替换了一个**被实测否定**的设计。

第一版解码走「**重放**」：给定 id 256+i，在已建好的输出 buffer 里找规则 (left,right) 的
相邻出现再拼接。决定性证据说明它错了：

```
'VMLM' 编码成 [415]，而 415 的规则是 (86, 411) —— id 411 根本不在 id 序列里。
```

真实 BPE 解码器**从不**依赖「操作数出现在序列里」。重放要求 411 出现在 buffer 里，
那是一个**格式从未要求的额外假设**，而它在「整词 = 单个 id」这种最常见的情形下**必然失败**。
实测：`'VMLM'` / `'Transformer'` / `'kernel32'` / `'是一个把'` / 整句中文**全部硬失败**，
只有 `'the'` 这种纯字节词通过。

**为什么不能靠「扫描输出 buffer」补救（实测数据，留着当历史依据）**：

| 版本 | 规则 | 结果 |
|---|---|---|
| v1 | 找操作数**相邻**的位置替换 | 665 条规则里 **415 条**匹配不到，大面积硬失败 |
| v2 | 按 piece 字节的**最左**出现拼接 (`find`) | **错误且安静**：真实序列上返回**非法 UTF-8**（`b'\x96...'`）；13 个探针词只有 **2 个**恰好正确 |
| v3 | 按**最右**出现拼接 (`rfind`) | 好些，但仍依赖「操作数以连续后缀存在」这个假设；实测 **100 条采样规则里 27 条的左出现 ≠ 尾随出现**，且仍在整词单 id 上失败 |

⇒ 规则：**不要从输出 buffer 推导 piece 的字节。**从合并表推导，然后拼接。

附带收益：这个改动**删掉了一整类错误** ——「不可达的合并 id」不可能发生，于是没有延迟
工作队列、没有第二遍、没有需要维护的状态。**删掉一类错误比维护一套处理它的协议好。**

### 4.5 不做 ASCII 大小写归一化 —— 这是一条**显式设计决定**，不是遗漏

两条理由，第二条是决定性的：

1. `core/src/lib.rs` 只有 `#[link(name = "kernel32")]`，**没有 user32**。用 `CharUpperA`
   需要新增一个绑定，而引擎体积是硬预算。
2. **归一化会弄错**：BPE 合并表本身大小写敏感（`'T'` 和 `'t'` 是不同 id）。把输入折叠
   成统一大小写会**破坏 round-trip 逐字节相同**这条判据。归一化不是「少做一点」，
   而是**会算错**。

同理，CLI 子命令大小写敏感。复用 core 已有的 `arg(n,dst)`（内部即 `GetCommandLineA`），
**不新增任何 kernel32 绑定**。

### 4.6 分隔符模型（写死，且是判据的一部分）

文本按**单个 U+0020** 切词；每个非空词编码成 id 序列；每个空词、以及词与词之间，
各写一个 id 32（`b' '`）。全空输入按**字节数**写 id 32，一一对应。

为什么全空要特例：「每个空词一个 id 32」无法区分 1 个和 2 个空格（都只有 1 个空词），
于是编码在那个输入族上**不是单射**，round-trip 静默失败。这是实测踩出来的，不是想出来的。

---

## 5. 两个「上界 / 消息文本」的实测教训

### 5.1 arena 「上界」写错 ⇒ 访问违例

第一版 `expand_vocab_measure` 返回上界 `256 + 2*merge_bytes`，理由是「每个操作数的叶子数
<= merge_bytes」。**那个证明是错的，实测把它打穿了**：

```
上界 = 10896      实际需要 = 38189      比值 0.29
```

实测最长的一个 piece 有 **15120 字节**，**单个**就超过那个「上界」。错误在于把「字节数」
和「节点数」混为一谈：`merge_bytes` 是**规则条数**的量级，而一个 piece 的字节数可以是它的
**和**级（BPE 的定义就是每次合并把两棵子树**相加**）。

后果：arena 给小了 ⇒ `copy_nonoverlapping` 越界写 ⇒ 进程 **0xC0000005**，
报出来是 `rc=3221225477` 这种毫无归因价值的数字（本轮 8 个 FAIL 里有 5 个是它）。
「上界写错」不是一个保守的浪费，它是一个**崩溃源**。

⇒ 现在返回**精确值**，而且 `expand_vocab` 显式收 `arena_cap` 并逐次检查，报出
「第几条规则、需要多少、还剩多少」而不是崩。

### 5.2 守卫消息里的字面量会与常量分叉

G3 的**判定**一直正确（用的是 `MERGE_ENTRY`），但**消息文本**写死了 `" * 7"`
（当时的 `MERGE_ENTRY`）。常量改成 8 后字符串没跟着改，于是守卫**打印了一个错误的期望公式**
—— 报错消息本身把排查引向歧途，比不报更糟。抓它的是 can-fail 对照 (b3)。

⇒ 消息里改成打印 `MERGE_ENTRY` 的**值**，让它不可能与代码分叉。

### 5.3 `MERGE_ENTRY` 第一版写成 **7**（真实 bug）

布局是「两个 u16 后接一个 u32，直接拼接」：2 + 2 + 4 = **8**，由定义得出。
7 的来源是「按 C 结构体对齐算再减掉填充」—— 但本格式根本没有那个对齐层。

后果：头部声明 665×7 = 4655，而 Python 侧实写 665×8 = 5320，文件 5352 B / 声明 4687。
**抓它的是独立的另一端**（Rust G1 在装载期）：

```
BPE: file length disagrees with header
BPE: avail=5352 required=4687
```

写入方自己**永远不会**发现。这是「声明尺寸与真实尺寸必须在写入点互相校验」的实证，
也是为什么 G1 用**精确相等**而不是 `>=`。

---

## 6. 最容易误读的一个 bug：fast encoder 不是最低 rank 优先

第一版「见一个合一个」（从左到右，能合就合，反复多轮）**不是**最低 rank 优先。实测反例：

```
word "engine":  fast(错) = [259, 445, 291]  = "en" "gi" "ne"
                correct  = [628, 101]       = "engin" "e"
```

它把 `("e","n") -> "en"` 合了，因为那一对在当前扫描位置可用，尽管
`("i","n") -> "in"` 的 rank 更低（2）本应先合。

**关键在于：两个切分解码回同样的字节。** 所以 **round-trip 判据看不出这个 bug** ——
只有与 O(n²) 参考实现对拍才发现（Python 控制 **C6** 当时 FAIL 了）。

这正是本项目「判据两端必须来自不同来源」的价值：round-trip 只覆盖**存储/还原的完整性**，
覆盖不了**切分质量**；后者需要一个独立的参考实现。

---

## 7. PASS / FAIL / UNPROVEN 总表

### PASS（全部实测，含原始输出）

| # | 判据 |
|---|---|
| 1 | `bpe` feature 对默认引擎贡献 **0 B**（隔离 target-dir；A=B=74752，C=96256 正对照） |
| 2 | 默认引擎 74752 B ≤ 84480 B 预算（余 9728 B） |
| 3 | `--features bpe` 在 `default-features = false` 下**单独**可编译（正交性） |
| 4 | Python emitter 与出厂产物**逐字节相同**（sha256 `ecda210459697343…`） |
| 5 | Rust 与 Python 对**每个**头部字段一致（6/6） |
| 6 | Rust arena 精确尺寸 == Python 独立算出的展开尺寸（38189 B） |
| 7 | Rust 解码 == Python 解码，**全部 3881 个语料词**，0 mismatch |
| 8 | round-trip 逐字节相同：136 B 样例 + **37800 B 全文**（`sha_in == sha_out`） |
| 9 | Rust 与 Python 对全文 **id 数一致**（15080） |
| 10 | must-fail：改一个 token id ⇒ 输出必须变（Rust 与 Python 两侧都测） |
| 11 | must-fail：改一个合并操作数 ⇒ 该 id 的字节必须变（并与 Python 独立期望值相符） |
| 12 | 15 条守卫 can-fail 对照：(a)(b)(b2)(b3)(c)(d)(e)(f)(g)(h)(i)(j)(j2)(k)(l) 全部 rc=9 且带数值 |
| 13 | 词表自控制 C1/C2/C3/C6/C7/C8 全 PASS |

### UNPROVEN（**不报 PASS**）

| # | 判据 | 为什么 |
|---|---|---|
| U1 | 特殊表（byte-fallback）路径在**真实产物**上被走过 | 665/665 规则都在语料上触发 ⇒ `n_special = 0` ⇒ 特殊表为空 ⇒ 该路径在出厂产物上**没有**被覆盖。G6/G7 只由畸形文件对照覆盖。**不去造一个假的 special 表来让它变 PASS。** |
| U2 | encoder == 第三方 BPE 实现 | 本仓库没有参考 tokenizer，且不允许 pip；一致性是在**两个本仓库自写实现**之间测的 |
| U3 | 占位 logits 生成文本 == 输入文本 | logits 是种子 PRNG，不是模型；本阶段不声称模型可用 |

### FAIL

无。

---

## 8. 复现与文件清单

```
core/src/bpe.rs                 出厂模块（no_std，无第三方 crate，无新增 kernel32 绑定）
core/Cargo.toml                 bpe = []            （共享文件，由 lead 代改）
core/src/lib.rs                 #[cfg(feature="bpe")] pub mod bpe;  （同上）
tools/p17_bpe_vocab.py          独立 Python 参考：格式、训练器、编解码器、自控制 C1–C8
tools/p17_bpe_xcheck.py         Rust×Python 交叉校验 + 15 条守卫 can-fail 对照
tools/p17_bpe_roundtrip.py      验收链路 text→id→logits→text + 改 id 对照
tools/p17_bpe_size.py           隔离 target-dir 体积实测（3 arm）
examples/p17_bpe/*              产物与证据（见本目录表）
```

复现全部证据的四条命令见文件开头。全部编号数字均可在对应的 `*.json` 中查到原始值。
