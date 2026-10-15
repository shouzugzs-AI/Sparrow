//! Sparrow V0.1.1 —— opcode 后端：架构级 `.spar` → opcode 级 `.vml`
//!
//! 展开规则表（每条规则一个函数，见 V0.1.1 报告 §1）：
//!
//! | Sparrow 语句 | opcode 序列 | 常量（写进 `.args` 清单） |
//! |---|---|---|
//! | `模型 小:` | `.section forward` 头（`.section` 被汇编器跳过，仅为对齐既有示例） | — |
//! | `嵌入 词表=V 维度=D` | `SHAPE n_out,1,D` + `EMB n_out, n_E, 模型输入` | `n_E:[V,D]` |
//! | `稀疏 维度=D 连接=K` | `SHAPE n_col,D,1` + `SPARSE_MATVEC n_col, n_off, n_src, n_W, in` + `TRANSPOSE n_out, n_col` | `n_off:[D+1,1]` `n_src:[D*K,1]` `n_W:[D*K,1]` |
//! | `线性 输出=O`（输入 I） | `GEMM n_out, in, n_W` + `BIAS n_out, n_b` | `n_W:[O,I]` `n_b:[1,O]` |
//! | `归一化 维度=D` | `LN n_out, in, n_g, n_b` | `n_g:[1,D]` `n_b:[1,D]` |
//! | `激活 act=A` | `ACT n_out, in, A` | — |
//! | `连接 a b c` | **不产生 opcode**（决定数据流路由：下层读哪个槽） | — |
//! | `前向 入 出` | 展开顺序 = 主入口序列；末尾 `PRINT <出>_out` + `HALT` | — |
//! | `重复 n` | opcode 层**展开成 n 份**（各自独立槽，朴素不复用） | 每份一套 |
//!
//! 三条硬约束（任务书第六部分）：
//! 1. **不改汇编器** —— 只用既有助记符与操作数形式；
//! 2. **不改引擎** —— 只产出 `.vml` 文本；
//! 3. opcode 序列必须能被 `vmlm-asm` 编译通过。
//!
//! 槽位分配（朴素版，复用排在 V0.1.4）：每个层实例独占一个输出槽 `<inst>_out`，
//! 权重是**常量槽**（由 `.args` 清单绑定到 .bin，不占激活槽）。
//!
//! # V0.1.2 P0-2：`重复` 的循环发射（`--loop-repeat`）
//!
//! 默认仍是**展开**（V0.1.1 行为，字节不变）。打开 `loop_repeat` 后，
//! 一个 `重复 N` 组只发射**一份**层模板，用 `LOOP_CNT` 转 N 圈、
//! `LAYERB` 每圈把常量基址推进一个"层块步长"：
//!
//! ```text
//!     SHAPE   t__24__in, 1, D      ; 进位槽（循环体的进与出都走它，避免别名）
//!     COPY    t__24__in, 嵌入_out
//!     SHAPE   t__24__ctr, 1, 1     ; 计数器/步长槽必须是**激活槽**
//!     COPY    t__24__ctr, @t__1_iters
//!     SHAPE   t__24__ls, 1, 1
//!     COPY    t__24__ls, @t__1_lstride
//!     SHAPE   t__24__ns, 1, 1
//!     COPY    t__24__ns, @t__1_nstride   ; = -N*stride（收尾还原）
//! t__24__l1:
//!     <层模板：常量名一律引用第 1 份 t__1_*，槽位用第 N 份 t__24_*>
//!     COPY    t__24__in, t__24_out       ; 进位
//!     LAYERB  t__24__ls                  ; base += stride
//!     LOOP_CNT t__24__ctr, @t__24__l1    ; 减完再判（引擎语义）
//!     LAYERB  t__24__ns                  ; base 还原，供循环后的常量访问
//! ```
//!
//! 四个必须同时成立的条件（错一个都是**静默错**，所以每条都验）：
//!
//! 1. **池里第 r 份与第 r+1 份的偏移差恒等于步长** —— 由 `vmbmeta` 读
//!    `.vmb` 的常量表**实测**，不是前端复刻 `CONST_ALIGN` 猜的（`E210`）；
//! 2. **循环体里不能出现非 per-layer 的常量访问** —— 否则会被 `layer_base`
//!    一起平移。所以循环的输入一律先进**激活槽**（`COPY` 在循环外做）；
//! 3. **计数器/步长槽必须是激活槽** —— `LAYERB`/`LOOP_CNT` 读槽走 `const_ptr`，
//!    常量槽会被 `layer_base` 平移，读到错的值；
//! 4. **步长必须能被 f32 精确表示** —— `LAYERB` 的增量槽是 f32，落不到整数就
//!    从第 2 层起错（`E212`）。
//!
//! 输出槽沿用**最后一份**实例的名字（`t__24_out`），所以下游接线与展开版
//! 完全一致；`.args` 仍列全 N 份常量（池必须连续）。

use std::collections::HashMap;

use crate::ast::{Expr, Span};
use crate::diag::{Diag, Result};
use crate::sema::{lower, CItem, Lowered};
use crate::vmbmeta::{align_up, f32_exact_int, ConstEnt, CONST_ALIGN};

/// 后端开关。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GenOptions {
    /// `重复 N` 走 `LOOP_CNT` + `LAYERB`（O(1) 指令流），而不是展开 N 份。
    pub loop_repeat: bool,
    /// 激活槽按生命周期复用（V0.1.3）。判据是"与不复用版逐字节相同"，
    /// 收益是 `.vmb` 的**激活区字节数**变小（缓存局部性）。
    pub slot_reuse: bool,
    /// 权重存储精度：`bf16` 让线性层权重与嵌入表的字节数**减半**。
    ///
    /// 引擎的加宽是精确的（bf16 = f32 高 16 位），所以判据是
    /// "与'把 f32 权重预先截断成 bf16 精度'的模型输出相同"。
    /// 只有 `GEMM` 的 b 操作数与 `EMB` 的表能吃 bf16 —— 引擎装载期会硬审计。
    pub weight_dtype: DataType,
}

/// 一个循环化的 `重复` 组 —— 步长对账时需要的信息。
#[derive(Clone, Debug)]
pub struct LoopGroup {
    /// 模板实例的常量名前缀（`t__1`）。
    pub cprefix: String,
    /// 循环次数 N（`重复N次` 的 N）。
    pub iters: u32,
    /// 前端**预测**的每层常量块字节步长（必须与 `.vmb` 实测值相等）。
    pub stride: u64,
    /// `sets[r][p]` = 第 r+1 份的第 p 个常量在 `Generated::consts` 里的下标。
    pub sets: Vec<Vec<usize>>,
}

impl LoopGroup {
    /// 循环结束时的基址（= N × stride）——收尾 `LAYERB` 要减掉它。
    pub fn total_shift(&self) -> u64 {
        self.stride * self.iters as u64
    }
}

/// 常量在 `.vmb` 里的存储精度（V0.1.3 性能项）。
///
/// 引擎的**加宽是精确的**：bf16 = f32 的高 16 位，`(u32)h << 16` 还原成同一个
/// f32 位模式（`interp.rs` / `format.rs` 的 `DT_BF16` 注释）。
/// 所以 bf16 权重算出来的结果与"把这些 f32 权重先截断成 bf16 精度"完全一致 ——
/// 这正好给了一条**可判定**的验收：两条路必须输出相同。
///
/// 只有两类常量能吃 bf16（引擎装载期的硬审计只放行这两个位置）：
/// `GEMM` 的 `b`（线性层权重）与 `EMB` 的 `a`（嵌入表）。
/// `BIAS`/`LN`/`SPARSE_MATVEC` 的常量必须是 f32，否则装载时 exit(9)。
///
/// **Int8（V0.1.5 ⑥）只在 `--backend lib` 里有意义**：它是 `vmlm_ops::kernels::gemm_nt_int8`
/// 的权重布局（`q: i8` + 每**输出行**一个 `scale`，LLM.int8 的标准布局）。
/// `.vmb`/汇编器那条路**没有**反量化 GEMM 指令，所以前端在 `--backend vml` / `--asm`
/// 下会明确拒绝 int8（`E109`），而不是产出一个跑不起来的 `.vmb`。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DataType {
    #[default]
    F32,
    Bf16,
    /// 线性层权重的 int8 量化形式：`q: i8`（1 B/元素）+ `scale: f32[out]`。
    Int8,
}

impl DataType {
    pub fn bytes(self) -> u64 {
        match self {
            DataType::F32 => 4,
            DataType::Bf16 => 2,
            DataType::Int8 => 1,
        }
    }

    pub fn arg_suffix(self) -> &'static str {
        match self {
            DataType::F32 => "", // 省略即 f32（汇编器的默认）
            DataType::Bf16 => ":bf16",
            // `.vmb` 路径不支持 int8（见上），这个后缀只为报错信息可读。
            DataType::Int8 => ":i8",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            DataType::F32 => "f32",
            DataType::Bf16 => "bf16",
            DataType::Int8 => "int8",
        }
    }
}

/// 一条常量声明：`name:file:rows:cols[:dtype]`（汇编器命令行格式）。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstDecl {
    pub name: String,
    pub file: String,
    pub rows: u32,
    pub cols: u32,
    pub what: &'static str,
    pub dtype: DataType,
}

impl ConstDecl {
    pub fn arg(&self) -> String {
        format!(
            "{}:{}:{}:{}{}",
            self.name,
            self.file,
            self.rows,
            self.cols,
            self.dtype.arg_suffix()
        )
    }

    /// 元素个数 × 每元素字节数（**未补齐**）。
    pub fn bytes(&self) -> u64 {
        self.rows as u64 * self.cols as u64 * self.dtype.bytes()
    }
}

#[derive(Clone, Debug)]
pub struct Generated {
    /// opcode 级 `.vml` 正文。
    pub vml: String,
    /// 汇编器权重清单（每行一条 `name:file:rows:cols`）。
    pub args_text: String,
    pub consts: Vec<ConstDecl>,
    /// 分配的激活槽数（朴素 = 每层 1 个 + 嵌入索引等）。
    pub slots: Vec<String>,
    /// 循环化的 `重复` 组（P0-2；`loop_repeat` 关闭时为空）。
    pub loop_groups: Vec<LoopGroup>,
    /// 前端自己产出的极小常量文件 `(文件名, f32 字节)` —— 计数器与步长。
    /// 调用方要写进**汇编器的 CWD**（= `.args` 里相对路径的基准目录）。
    pub extra_bins: Vec<(String, [u8; 4])>,
    /// 结构化图（Rust 后端用）。
    pub plan: Plan,
}

/// 一个待展开/待循环化的层实例（`重复` 在收集阶段展成多份，组内连续）。
#[derive(Clone, Debug)]
struct Layer {
    /// `.spar` 里的原名（`连接`/`前向` 引用它）。
    orig: String,
    /// opcode 层实际使用的名字（槽位/常量名前缀）。中文名会被换成 ASCII 名。
    name: String,
    kind: String,
    attrs: Vec<(String, Expr)>,
    span: Span,
    out_dim: u32,
    in_dim: u32,
    /// 前驱在链上的下标。**空 = 链首**（输入来自模型输入常量）；
    /// **≥2 = 汇入**（V0.1.3：先把多个前驱合起来，再喂给这一层）。
    preds: Vec<usize>,
    /// 汇入方式（只在 `preds.len() >= 2` 时有意义）。
    merge: MergeMode,
    /// 所属 `重复` 组的编号（同一 CItem::Layer 的实例共享）。
    group: usize,
    /// 组内序号（0 起）。
    gi: usize,
    /// 组的大小（1 = 没写 `重复`）。
    gn: usize,
}

/// V0.1.3 汇入（一个层有多个前驱）时的合流方式。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergeMode {
    /// 相加 `EWADD`（要求各前驱等宽）—— 残差连接就是这种。
    Add,
    /// 列拼接 `CONCAT`（宽度相加）。
    Concat,
}

impl MergeMode {
    pub fn name(self) -> &'static str {
        match self {
            MergeMode::Add => "相加(EWADD)",
            MergeMode::Concat => "拼接(CONCAT)",
        }
    }
}

fn attr_int(attrs: &[(String, Expr)], key: &str) -> Option<i64> {
    attrs.iter().find(|(k, _)| k == key).and_then(|(_, v)| match v {
        Expr::Int(n) => Some(*n),
        _ => None,
    })
}

fn attr_ident(attrs: &[(String, Expr)], key: &str) -> Option<String> {
    attrs.iter().find(|(k, _)| k == key).and_then(|(_, v)| match v {
        Expr::Ident(s) => Some(s.clone()),
        _ => None,
    })
}

/// 真假属性（`因果=是` / `causal=1` / `掩码=真`）。
///
/// 缺失 = 假；给了但不认识的真假词 → `E108`（**不**默认成假：
/// "掩码=或许"这种写法如果被静默当成"不掩码"，就是静默算错模型）。
fn attr_flag(attrs: &[(String, Expr)], key: &str, span: Span, layer: &str) -> Result<bool> {
    let Some((_, v)) = attrs.iter().find(|(k, _)| k == key) else {
        return Ok(false);
    };
    let raw = match v {
        Expr::Ident(s) => s.clone(),
        Expr::Int(n) => n.to_string(),
        Expr::Float(f) => f.to_string(),
        _ => {
            return Err(Diag::error(
                "E108",
                format!("层 `{layer}` 的 `{key}` 需要一个真假词（是/否）"),
                span,
            ))
        }
    };
    let w = crate::keywords::norm(&raw);
    if matches!(
        w.as_str(),
        "是" | "真" | "对" | "有" | "开" | "启用" | "1" | "true" | "yes" | "on" | "y"
    ) {
        Ok(true)
    } else if matches!(
        w.as_str(),
        "否" | "不" | "假" | "无" | "关" | "0" | "false" | "no" | "off" | "n"
    ) {
        Ok(false)
    } else {
        Err(
            Diag::error(
                "E108",
                format!("层 `{layer}` 的 `{key}` 值 `{raw}` 不是真假词"),
                span,
            )
            .with_candidates(vec!["是".to_string(), "否".to_string()]),
        )
    }
}

/// 模型输入常量的名字。
fn input_name(model: &str) -> String {
    safe_ident(model, "model", 0, "_in")
}

/// opcode 层的名字必须是 ASCII：符号表、`.args` 命令行、各工具链都按字节处理，
/// 中文名在这条路上没有收益只有风险（第九部分"边界判断"）。
/// 另外要避开汇编器的两个前缀约定：`st_` = 持久状态槽、`g_` = 梯度槽。
fn safe_ascii(name: &str) -> bool {
    !name.is_empty()
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !name.starts_with("st_")
        && !name.starts_with("g_")
}

fn safe_ident(name: &str, kind: &str, idx: usize, suffix: &str) -> String {
    if safe_ascii(name) {
        format!("{name}{suffix}")
    } else {
        format!("{}{}{}", kind.to_ascii_lowercase(), idx, suffix)
    }
}

/// 一个层的**结构化**描述（V0.1.4 的 Rust 后端用它，而不是去解析 `.vml` 文本）。
///
/// 为什么要有它：文本后端与 Rust 后端必须来自**同一张图**。
/// 让 Rust 后端去 parse `.vml` 会立刻产生"两份真相"，而让两边各自重算一遍图
/// 更糟（漂移迟早发生）。`Plan` 是 `generate_with` 已经算好的那份：
/// 层、拓扑序、维度、常量、汇入方式、槽基名 —— 原样导出。
#[derive(Clone, Debug)]
pub struct PlanLayer {
    /// 槽基名（输出槽 = `<base>_out`）。
    pub base: String,
    /// `.spar` 里的原名（报错用）。
    pub orig: String,
    pub kind: String,
    pub attrs: Vec<(String, Expr)>,
    pub in_dim: u32,
    pub out_dim: u32,
    /// 前驱在 `Plan::layers` 里的下标（≥2 = 汇入）。
    pub preds: Vec<usize>,
    pub merge: MergeMode,
    /// 该层是否属于一个 `重复` 组（Rust 后端暂不支持循环组，见 E109）。
    pub group: usize,
    pub gn: usize,
}

/// 一个循环化的 `重复` 组在 `Plan` 里的投影（V0.1.5）。
///
/// 文本后端用 `LOOP_CNT`+`LAYERB` 平移到第 k 份的常量；Rust 后端不能平移指针，
/// 它直接按实例取常量。两边必须来自**同一条图**，否则"两条后端逐字符一致"
/// 就退化成"两份各自推导的图碰巧一致"。这里把循环真正需要的那几个量显式记下来：
/// 实例序列、链输入（播种进位槽）、每圈都参与的外部前驱、汇入方式。
#[derive(Clone, Debug)]
pub struct PlanGroup {
    /// 组内实例的层下标，**按实例顺序**（第 0 个是模板、也是循环触发点）。
    pub insts: Vec<usize>,
    /// 链输入：第 1 份的 `preds[0]`（它的输出槽用来播种进位槽）。
    pub seed: usize,
    /// 每圈都参与的外部前驱（第 1 份的 `preds[1..]`）。
    pub ext: Vec<usize>,
    pub merge: MergeMode,
    pub in_dim: u32,
    pub out_dim: u32,
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub model: String,
    pub layers: Vec<PlanLayer>,
    /// 拓扑序（下标指向 `layers`）。
    pub order: Vec<usize>,
    /// 每个层输出槽的物理基名（槽位复用会改它）。
    pub slot_base: Vec<String>,
    pub consts: Vec<ConstDecl>,
    /// 序列长度（>1 时每层都是 `[T, D]`）。
    pub t_seq: u32,
    /// 出口层在 `layers` 里的下标。
    pub out_idx: usize,
    /// 循环化的 `重复` 组（下标的语义见 [`PlanGroup`]）。
    pub groups: Vec<PlanGroup>,
    /// `损失 <kind>`（V0.1.6）：canonical 名（现在只有 `mse`）。
    pub loss: Option<String>,
    /// `反向`（V0.1.6）：要求发射反向遍历。Rust 后端（`--backend lib`）不支持它。
    pub backward: bool,
}

/// 主入口：`.spar` IR → opcode `.vml`（默认 = 展开 `重复`，V0.1.1 行为）。
pub fn generate(lowered: &Lowered, src_path: &str) -> Result<Generated> {
    generate_with(lowered, src_path, GenOptions::default())
}

/// 主入口（带后端开关）。
pub fn generate_with(
    lowered: &Lowered,
    src_path: &str,
    opts: GenOptions,
) -> Result<Generated> {
    let _ = src_path;
    let model = &lowered.model;
    let mut layers: Vec<Layer> = Vec::new();
    let mut connects: Vec<Vec<String>> = Vec::new();
    let mut forward: Option<(String, String)> = None;
    let mut unsupported: Vec<(&'static str, Span, String)> = Vec::new();
    let mut n_groups = 0usize;
    // V0.1.6：训练陈述（模型级）。`损失` 给反向一个起点，`反向` 要求发射反向遍历。
    let mut loss: Option<String> = None;
    let mut backward: Option<Span> = None;

    // ---- 1. 收集 ----
    for item in &lowered.items {
        match item {
            CItem::Layer { name, kind, attrs, repeat, span } => {
                let n = repeat.unwrap_or(1).max(1) as usize;
                let base = safe_ident(name, kind, layers.len(), "");
                let group = n_groups;
                n_groups += 1;
                for r in 0..n {
                    let lname = if n == 1 { base.clone() } else { format!("{base}__{}", r + 1) };
                    layers.push(Layer {
                        orig: name.clone(),
                        name: lname,
                        kind: kind.clone(),
                        attrs: attrs.clone(),
                        span: *span,
                        out_dim: 0,
                        in_dim: 0,
                        preds: Vec::new(),
                        merge: MergeMode::Add,
                        group,
                        gi: r,
                        gn: n,
                    });
                }
            }
            CItem::Connect { path, .. } => connects.push(path.clone()),
            CItem::Forward { input, output, .. } => {
                if forward.is_none() {
                    forward = Some((input.clone(), output.clone()));
                }
            }
            CItem::CustomOp { name, span, .. } => {
                unsupported.push(("自定义算子", *span, name.clone()))
            }
            CItem::OpCall { name, span, .. } => {
                unsupported.push(("自定义算子实例", *span, name.clone()))
            }
            CItem::Raw { opcode, span, .. } => unsupported.push(("裸指令", *span, opcode.clone())),
            CItem::Loss { kind, .. } => loss = Some(kind.clone()),
            CItem::Backward { span } => backward = Some(*span),
            CItem::Def { .. } => {}
        }
    }

    if let Some((what, span, name)) = unsupported.first() {
        return Err(Diag::error(
            "E101",
            format!(
                "{what} `{name}` 不能编到 opcode：V0.1.1 的后端只支持内置层类型\
                 （嵌入/稀疏/线性/归一化/激活）"
            ),
            *span,
        )
        .with_hint("自定义算子需要 opcode 名 → VMLM 助记符 的映射表，排在 V0.2")
        .with_note("裸指令（`@gemm` 等）本来就是 opcode 级写法，不需要再编译"));
    }

    if layers.is_empty() {
        return Err(Diag::error("E104", format!("模型 `{model}` 里没有可编译的层"), Span::ZERO));
    }

    // ---- 2. 连边：把 `连接 a b c` 翻译成 a→b→c 的前驱关系 ----
    //
    // `重复 n` 展开后，一个名字对应 n 份：**入边接第一份、出边取最后一份**
    // （`t = 稀疏块(...) 重复24次` 的语义就是 24 层串联）。
    let mut first_of: HashMap<String, usize> = HashMap::new();
    let mut last_of: HashMap<String, usize> = HashMap::new();
    for (i, l) in layers.iter().enumerate() {
        first_of.entry(l.orig.clone()).or_insert(i);
        last_of.insert(l.orig.clone(), i);
    }
    for path in &connects {
        for w in path.windows(2) {
            let (Some(&a), Some(&b)) = (last_of.get(&w[0]), first_of.get(&w[1])) else {
                return Err(Diag::error(
                    "E102",
                    format!("连接路径里的 `{}` 或 `{}` 不是本模型的层", w[0], w[1]),
                    Span::ZERO,
                )
                .with_candidates(last_of.keys().cloned().collect::<Vec<_>>()));
            };
            // V0.1.3：同一层被多条 `连接` 指到 = **汇入**（多前驱），
            // 不再报错；合流方式在维度传播时定（等宽→相加，不等宽→拼接，
            // 也可用 `合并=相加|拼接` 显式指定）。
            if !layers[b].preds.contains(&a) {
                layers[b].preds.push(a);
            }
        }
    }
    // 没有任何 `连接` 时，**声明顺序就是数据流顺序**（Level 0/1 没有连接语句）。
    // `连接` 只是显式路由；不写时按声明序成链，这是最宽容且确定的默认。
    if connects.is_empty() {
        for i in 1..layers.len() {
            if layers[i].preds.is_empty() {
                layers[i].preds.push(i - 1);
            }
        }
    }
    // 重复层内部串成链（t__1 → t__2 → … → t__n）
    for i in 1..layers.len() {
        if layers[i].orig == layers[i - 1].orig && layers[i].preds.is_empty() {
            layers[i].preds.push(i - 1);
        }
    }
    // **重复组内每个实例的前驱结构必须一致**（V0.1.3）。
    //
    // 为什么：`连接 x t` 只能指到组内**第一份**（`first_of` 的语义），
    // 于是"残差前驱"只会挂到 t__1 上，t__2/t__3 只剩内部链前驱 ——
    // 展开版与循环版会算出**不同的函数**（循环版每圈都汇入）。所以把第一份的
    // "链前驱之外的前驱"复制给组内每一份：
    //   t__1 = [链前驱] + 残差；t__k = [t__(k-1)] + 残差
    for g in 0..n_groups {
        let members: Vec<usize> = (0..layers.len()).filter(|i| layers[*i].group == g).collect();
        if members.len() < 2 {
            continue;
        }
        let ext: Vec<usize> = layers[members[0]].preds.iter().copied().skip(1).collect();
        if ext.is_empty() {
            continue;
        }
        for k in 1..members.len() {
            let chain = members[k - 1];
            let mut p = vec![chain];
            for e in &ext {
                if !p.contains(e) {
                    p.push(*e);
                }
            }
            layers[members[k]].preds = p;
        }
    }

    // ---- 3. 定序（拓扑序）：**链首仍唯一**（唯一的数据入口），但允许分叉后合流 ----
    let roots: Vec<usize> = (0..layers.len()).filter(|i| layers[*i].preds.is_empty()).collect();
    if roots.len() != 1 {
        let (idx, what) = if roots.is_empty() {
            (0, "图里有环（没有任何链首）".to_string())
        } else {
            (
                roots[1],
                format!(
                    "图里有 {} 个链首（{}）：只支持一个数据入口",
                    roots.len(),
                    roots
                        .iter()
                        .map(|i| layers[*i].name.clone())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            )
        };
        return Err(Diag::error("E102", what, layers.get(idx).map(|l| l.span).unwrap_or(Span::ZERO))
            .with_hint("所有分支都要从同一条链首出发；用 `连接 a b c` 或 `前向 a c` 明示数据流"));
    }
    // Kahn 拓扑序：每一步取"下标最小且前驱都已就位"的那个 ⇒ 确定性。
    let mut placed = vec![false; layers.len()];
    let mut order: Vec<usize> = vec![roots[0]];
    placed[roots[0]] = true;
    while order.len() < layers.len() {
        let next = (0..layers.len()).find(|i| {
            !placed[*i] && !layers[*i].preds.is_empty() && layers[*i].preds.iter().all(|p| placed[*p])
        });
        match next {
            Some(i) => {
                placed[i] = true;
                order.push(i);
            }
            None => break,
        }
    }
    if order.len() != layers.len() {
        return Err(Diag::error(
            "E102",
            format!(
                "有 {} 个层既不在数据流里、也不是环的一部分（{}）：检查 `连接` 是否成环或漏接",
                layers.len() - order.len(),
                layers
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !placed[*i])
                    .map(|(_, l)| l.name.clone())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Span::ZERO,
        )
        .with_hint("汇入（多前驱）是允许的；环不允许"));
    }

    // ---- 4. 维度传播（决定常量形状）；顺带定下每个汇入节点的合流方式 ----
    let mut in_dim_of_chain: Option<u32> = None;
    for &i in order.iter() {
        let preds = layers[i].preds.clone();
        let pdims: Vec<u32> = preds.iter().map(|&p| layers[p].out_dim).collect();
        let (ind, outd) = if preds.len() >= 2 {
            // 汇入：先合，再喂给这一层
            let mode = merge_mode(&layers[i], &pdims)?;
            layers[i].merge = mode;
            let w = match mode {
                MergeMode::Add => pdims[0],
                MergeMode::Concat => pdims.iter().sum(),
            };
            let (declared_in, outd) = dims_of(&layers[i], Some(w))?;
            if declared_in != w {
                return Err(Diag::error(
                    "E100",
                    format!(
                        "层 `{}` 声明输入宽度 {declared_in}，但它的 {} 个前驱汇入后是 {w}（{}）",
                        layers[i].name,
                        preds.len(),
                        layers[i].merge.name()
                    ),
                    layers[i].span,
                )
                .with_hint("汇入的宽度由前驱决定：相加要求等宽，拼接是宽度之和；别在这一层再写 `输入=`"));
            }
            (w, outd)
        } else {
            dims_of(&layers[i], pdims.first().copied())?
        };
        layers[i].in_dim = ind;
        layers[i].out_dim = outd;
        if i == roots[0] {
            in_dim_of_chain = Some(ind);
        }
    }

    // ---- 5. 发射 opcode ----
    let mut consts: Vec<ConstDecl> = Vec::new();
    let mut loop_ctl: Vec<ConstDecl> = Vec::new();
    let mut loop_groups: Vec<LoopGroup> = Vec::new();
    let mut extra_bins: Vec<(String, [u8; 4])> = Vec::new();
    let mut slots: Vec<String> = Vec::new();
    let mut body = String::new();
    let is_embed_root = layers[order[0]].kind == "EMBED";

    // 序列长度进架构 IR（V0.1.3）：任何一层写了 `序列=T` 就抬高**模型输入的行数**
    // （输入激活变成 [T, D]），这样注意力（以及任何逐行算子）能拿到整段序列。
    //
    // V0.1.5 起链首也可以是 `嵌入`：逐行查表（T 次 `EMB`）+ 按列拼接 + 一次
    // `TRANSPOSE` 拼成 [T, D]（见 `emit_layer_body` 的 EMBED 分支）。
    // 仍然明确拒绝的是**逐行算子之外**的类型 —— 它们一次只吃/产一行：
    //   * `稀疏`：`SPARSE_MATVEC` 是向量×矩阵（[D,1] → [D,1]）；
    //   * 出现在链路中间的 `嵌入`（不是链首）。
    // 这两类若放过去，运行时会**只算第一行**并正常退出 —— 静默错答案。
    let t_seq: u32 = layers
        .iter()
        .map(|l| attr_int(&l.attrs, "seq").unwrap_or(1).max(1) as u32)
        .max()
        .unwrap_or(1);
    if t_seq > 1 {
        if let Some(sp) = layers.iter().find(|l| l.kind == "SPARSE") {
            return Err(Diag::error(
                "E107",
                format!(
                    "序列={t_seq} 时 `稀疏` 层 `{}` 不能进 IR：`SPARSE_MATVEC` 是**向量×矩阵**\
                     （[D,1] → [D,1]），一次只吃一行输入，产不出 T 行",
                    sp.name
                ),
                sp.span,
            )
            .with_hint("序列 >1 的链路里请用 `线性`（GEMM 天然按行），或去掉 `序列=`"));
        }
        if let Some(em) = layers
            .iter()
            .find(|l| l.kind == "EMBED" && l.name != layers[order[0]].name)
        {
            return Err(Diag::error(
                "E107",
                format!(
                    "序列={t_seq} 时 `嵌入` 层 `{}` 只能是**链首**：它的输入是词元 id 列表，\
                     而链路中间它的输入已经是 [T, D] 的激活",
                    em.name
                ),
                em.span,
            )
            .with_hint("把 `嵌入` 放在链首（它就是模型输入的查表），或去掉 `序列=`"));
        }
    }

    // 模型输入常量
    if is_embed_root {
        // 序列 >1：输入是**一行 T 个词元 id**（`SLICE` 一次取一个去做查表）
        consts.push(ConstDecl {
            name: input_name(model),
            file: format!("{}_in.bin", model),
            rows: 1,
            cols: t_seq,
            what: if t_seq > 1 {
                "模型输入（T 个词元 id，一行）"
            } else {
                "模型输入（词元 id）"
            },
            dtype: DataType::F32,
        });
    } else {
        consts.push(ConstDecl {
            name: input_name(model),
            file: format!("{}_in.bin", model),
            rows: t_seq,
            cols: in_dim_of_chain.unwrap_or(1),
            what: if t_seq > 1 { "模型输入 [T, D]" } else { "模型输入" },
            dtype: DataType::F32,
        });
    }
    // ---- 5.5 槽位复用（V0.1.3）：把激活区的物理槽按**生命周期**回收 ----
    //
    // 复用只改"写进哪个槽名"，不改任何算子、不改常量、不改指令顺序 ⇒
    // 数值上必须与不复用版**逐字节相同**（engine 侧 A/B 验，见 tools/slot_reuse.ps1）。
    //
    // 生命周期：`last_use[i]` = 最后一个**读** `i` 的层在 `order` 里的位置；
    // 被 `前向` 打印的那个槽 pin 到最后。一个槽在"最后一个读者"处理完之后
    // 就能被回收 —— 注意是**处理完那一层之后**，因为那一层同时还要读它。
    //
    // 池按**输出宽度**分桶：同一个物理槽要承载不同宽度的激活会影响汇编器
    // 为它分配的字节数（每个槽按最终形状 64B 对齐紧密排布），
    // 所以只回收"宽度完全相同"的槽 —— 这是一条**保守但可证**的规则。
    let mut last_use: Vec<usize> = vec![usize::MAX; layers.len()]; // MAX = 一直活到最后
    let mut last_use_set = vec![false; layers.len()];
    let out_idx = match &forward {
        Some((_, out)) => last_of.get(out).copied(),
        None => order.last().copied(),
    };
    // 出口槽**钉死**：它是唯一"被 PRINT 读"的槽，不能被回收（否则后面
    // 某一层复用它的名字，PRINT 打出来的就不是出口层的值了 —— 这是静默错）。
    if let Some(oi) = out_idx {
        last_use_set[oi] = true;
        last_use[oi] = usize::MAX;
    }
    for (pos, &i) in order.iter().enumerate() {
        for &p in &layers[i].preds {
            if !last_use_set[p] {
                last_use[p] = pos;
                last_use_set[p] = true;
            } else if last_use[p] != usize::MAX {
                last_use[p] = last_use[p].max(pos);
            }
        }
    }
    // `slot_base[i]` 是这一层输出槽的**基名**：真正的槽名是 `<base>_out`
    // （`emit_layer_body` 自己会拼 `_out`）。复用只换基名。
    let mut slot_base: Vec<String> = (0..layers.len()).map(|i| layers[i].name.clone()).collect();
    let slot_name = |b: &str| format!("{b}_out");
    let mut pool: HashMap<u32, Vec<String>> = HashMap::new();
    // **暂存池**（V0.1.5）：回收那些"活到本层读完就死"的中间槽 —— 汇入的
    // `<name>_in`、循环组的进位槽/汇入槽/计数器/步长槽。它们与输出槽的区别是
    // **名字形状**不同（`_in` / `__m` / `__ctr` 而不是 `<base>_out`），所以不能
    // 走上面那个"按基名 + `_out`"的池；这里存**完整槽名**。
    //
    // 桶键是 `(行, 列)` 而不是只看列宽：同一个名字被再次 `SHAPE` 时形状必须**完全一致**，
    // 否则汇编器为它分配的字节数会变（槽按最终形状 64B 对齐紧密排布），
    // 而"复用不改数值"这条判据就不再是显然的了。用例：循环组的进位槽是 `(1, D)`，
    // 若拿去当 `(T, D)` 的层输出槽，T>1 时两者字节数不同。
    let mut scratch: HashMap<(u32, u32), Vec<String>> = HashMap::new();
    let take_scratch = |pool: &mut HashMap<(u32, u32), Vec<String>>, key: (u32, u32), fallback: String| -> String {
        if opts.slot_reuse {
            pool.get_mut(&key).and_then(|v| v.pop()).unwrap_or(fallback)
        } else {
            fallback
        }
    };

    let mut done: Vec<bool> = vec![false; layers.len()];
    // V0.1.6：每层的**输入槽**（反向要用它算权重梯度 `dW += dy^T x`）。
    // 正向发射时记下来，反向就不用重新推一遍图。
    let mut layer_input: Vec<String> = vec![String::new(); layers.len()];
    for (pos, &i) in order.iter().enumerate() {
        if done[i] {
            continue;
        }
        let l = &layers[i];
        // 输入槽：0 个前驱 = 模型输入常量；1 个 = 前驱的输出槽；
        // ≥2 个 = **先汇入**到一个中间槽 `<name>_in`，再喂给这一层。
        let pred_slots: Vec<String> =
            l.preds.iter().map(|&p| slot_name(&slot_base[p])).collect();
        let pdims: Vec<u32> = l.preds.iter().map(|&p| layers[p].out_dim).collect();
        let (input_slot, merge_ops) = if pred_slots.len() >= 2 {
            let fallback = format!("{}_in", l.name);
            // 汇入的中间槽形状恒为 `(1, in_dim)`（`emit_merge` 里 rows 写死 1）⇒
            // 它可以安全地复用任何同形状的暂存槽。
            let dst = take_scratch(&mut scratch, (1, l.in_dim), fallback);
            let ops = emit_merge_into(l, &pred_slots, &pdims, l.merge, &dst);
            (dst, ops)
        } else {
            (
                pred_slots.first().cloned().unwrap_or_else(|| input_name(model)),
                String::new(),
            )
        };
        // 这一层的输入是不是"汇入产生的中间槽"（决定之后要不要回收它）
        let input_is_merge_scratch = pred_slots.len() >= 2;
        layer_input[i] = input_slot.clone();

        // --- 循环化的 `重复` 组（只在组内第一份上发射一次）---
        if opts.loop_repeat && l.gn >= 2 && l.gi == 0 {
            // 源层（没有前驱）不能循环化：循环体需要一条**链输入**来播种进位槽，
            // 而展开版里第 1 份读的是模型输入常量 —— 循环版没有"第 0 份"可搬。
            //
            // 这里必须给**明确的诊断**：V0.1.4 及以前直接 `l.preds[0]` 越界
            //（实测 `panicked at src\vmlgen.rs: index out of bounds: the len is 0`），
            // 而本项目纪律是"非法输入要硬失败并说清楚"，不是崩给用户看。
            if l.preds.is_empty() {
                return Err(Diag::error(
                    "E110",
                    format!(
                        "`重复` 组 `{}` 的第 1 份没有前驱（它是链首/源层）：循环化需要一条链输入来播种进位槽",
                        l.orig
                    ),
                    l.span,
                )
                .with_hint("给这一层加一条 `连接` 指向前驱，或去掉这一层的 `重复`（展开成 N 份，行为不变）"));
            }
            // 第 1 个前驱 = 链输入（**播种进位槽**）；其余 = 每圈都参与的残差前驱。
            // 组归一化已经保证组内每一份的前驱结构一致（见上文），
            // 所以"循环版"与"展开版"算的是同一个函数。
            //
            // 注意播种源必须是 preds[0] 的**输出槽**，不能用 `input_slot`
            // —— 后者在有汇入时是 `<name>_in`（展开版的汇入槽），循环版里不存在。
            let seed_slot = slot_name(&slot_base[l.preds[0]]);
            let ext_slots: Vec<String> =
                l.preds.iter().skip(1).map(|&p| slot_name(&slot_base[p])).collect();
            let ext_dims: Vec<u32> = l.preds.iter().skip(1).map(|&p| layers[p].out_dim).collect();
            let members: Vec<usize> =
                (0..layers.len()).filter(|j| layers[*j].group == l.group).collect();
            debug_assert_eq!(members.len(), l.gn);
            for &m in &members {
                done[m] = true;
            }
            let inst_names: Vec<String> =
                members.iter().map(|&m| layers[m].name.clone()).collect();
            let g = emit_loop_group(
                l,
                &seed_slot,
                &ext_slots,
                &ext_dims,
                &inst_names,
                t_seq,
                opts.weight_dtype,
                opts.slot_reuse,
                &mut extra_bins,
                &mut consts,
            )?;
            body.push_str(&g.body);
            slots.extend(g.slots);
            loop_ctl.extend(g.ctl);
            extra_bins.extend(g.bins);
            // 循环期间活着、循环后没人读的暂存槽 → 暂存池
            if opts.slot_reuse {
                for (r, c, n) in g.pool_in {
                    scratch.entry((r, c)).or_default().push(n);
                }
            }
            loop_groups.push(g.group);
            continue;
        }

        // 槽位复用：这一层要写哪个物理槽？（池里同宽度的空闲槽优先）
        if opts.slot_reuse {
            if let Some(name) = pool.get_mut(&l.out_dim).and_then(|v| v.pop()) {
                slot_base[i] = name;
            }
        }

        body.push_str(&merge_ops);
        let mut declared_here: Vec<ConstDecl> = Vec::new();
        let ops = emit_layer_body(
            l,
            &input_slot,
            &slot_base[i],
            &l.name,
            t_seq,
            opts.weight_dtype,
            opts.slot_reuse,
            &mut extra_bins,
            &mut declared_here,
        )?;
        for c in declared_here {
            if !consts.iter().any(|x| x.name == c.name) {
                consts.push(c);
            }
        }
        slots.push(slot_name(&slot_base[i]));
        body.push_str(&ops);
        // 这一层处理完了：把"最后一个读者就是这一层"的前驱槽回收。
        // （必须在用完它们之后再放回 —— 那一层的输入就是它们。）
        if opts.slot_reuse {
            for &p in &l.preds {
                if last_use_set[p] && last_use[p] == pos {
                    pool.entry(layers[p].out_dim).or_default().push(slot_base[p].clone());
                }
            }
            // 汇入的中间槽也活到"这一层读完它"为止（就在上面那一步之前它已被读走）
            if input_is_merge_scratch {
                scratch.entry((1, l.in_dim)).or_default().push(input_slot.clone());
            }
        }
    }
    // 循环控制常量**放在池尾**：池里"第 r 份 → 第 r+1 份"的偏移差必须恒等于
    // 步长，中间插一个常量就把这条不变式破坏了。
    consts.extend(loop_ctl);

    // ---- 6. 主入口序列收尾 ----
    let out_layer = match &forward {
        Some((_, out)) => match last_of.get(out) {
            Some(&i) => layers[i].name.clone(),
            None => out.clone(),
        },
        None => layers[*order.last().unwrap()].name.clone(),
    };
    // 出口槽的**物理名**（复用开/关都对）：关闭时 == `<出口层>_out`，
    // 开启时可能是回收来的槽名（它被钉住不会被回收，见上面的 last_use）。
    let out_slot = match out_idx {
        Some(i) => slot_name(&slot_base[i]),
        None => {
            return Err(Diag::error(
                "E105",
                format!("`前向` 的出口层 `{out_layer}` 不是本模型的层"),
                Span::ZERO,
            )
            .with_candidates(last_of.keys().cloned().collect::<Vec<_>>()))
        }
    };
    if !slots.iter().any(|s| *s == out_slot) {
        return Err(Diag::error(
            "E105",
            format!("`前向` 的出口层 `{out_layer}` 没有输出槽"),
            Span::ZERO,
        )
        .with_candidates(slots.clone()));
    }
    // ---- 6.6 反向遍历（V0.1.6）：`损失` + `反向` ----
    let mut grad_prints: Vec<String> = Vec::new();
    if let Some(bspan) = backward {
        let Some(loss_kind) = loss.clone() else {
            return Err(Diag::error(
                "E113",
                "`反向` 需要一个起点：模型里没有声明损失",
                bspan,
            )
            .with_hint("先写一行 `损失 均方`（引擎侧是 `OP_MSE_BWD`），再写 `反向`"));
        };
        let (text, prints) = emit_backward(
            &model,
            &loss_kind,
            &out_slot,
            out_idx.map(|i| layers[i].out_dim).unwrap_or(0),
            t_seq,
            &order,
            &layers,
            &layer_input,
            &opts,
            &mut consts,
        )?;
        body.push_str(&text);
        grad_prints = prints;
    }

    body.push_str(&format!("    PRINT   {out_slot}\n"));
    for g in &grad_prints {
        body.push_str(&format!("    PRINT   {g}\n"));
    }
    body.push_str("    HALT\n");

    // ---- 7. 拼装文本 ----
    let mut vml = String::new();
    vml.push_str("; sparrow v0.1.2 — opcode .vml (generated from .spar)\n");
    vml.push_str(&format!("; model: {model}\n"));
    if !loop_groups.is_empty() {
        vml.push_str("; 循环化：");
        for g in &loop_groups {
            vml.push_str(&format!(
                "{}×{} 步长 {} B（预测，汇编后按 .vmb META 对账）；",
                g.cprefix, g.iters, g.stride
            ));
        }
        vml.push('\n');
    }
    vml.push_str("; 常量清单（同 .args）：\n");
    for c in &consts {
        vml.push_str(&format!(
            ";   {:<16} {:<20} [{}, {}] {:<4} {}\n",
            c.name,
            c.file,
            c.rows,
            c.cols,
            c.dtype.name(),
            c.what
        ));
    }
    vml.push_str(".section forward\n");
    vml.push_str(&body);
    vml.push_str(".end\n");

    let mut args_text = String::new();
    for c in &consts {
        args_text.push_str(&c.arg());
        args_text.push('\n');
    }

    // 循环组在 Plan 里的投影：与文本后端**同一套推导**（实例集合、链输入、外部前驱）。
    // 这里不重新推图，只是把 `layers` / `loop_groups` 已有的信息摆成 Rust 后端要的形状。
    //
    // 注意 `Layer::group` 的语义（**踩过**）：它是**每个层**自增的计数器，不是循环组编号
    // —— 非 `重复` 的层也有自己的 group 值（n=1）。所以不能拿 `Plan::groups` 的下标去
    // 匹配 `Layer::group`；正确做法是先用模板实例名（`LoopGroup::cprefix`）**定位**那一层，
    // 再取它的 `group` 值当组号（文本后端也是用 `l.group` 取成员的）。
    let mut plan_groups: Vec<PlanGroup> = Vec::new();
    for lg in &loop_groups {
        let first = layers
            .iter()
            .position(|l| l.name == lg.cprefix)
            .unwrap_or_else(|| panic!("循环组的模板实例 `{}` 必须存在", lg.cprefix));
        let gid = layers[first].group;
        let insts: Vec<usize> = (0..layers.len()).filter(|&j| layers[j].group == gid).collect();
        debug_assert_eq!(insts.len(), lg.iters as usize, "实例数必须等于循环次数");
        debug_assert!(!layers[first].preds.is_empty(), "循环组的第一份必须有链输入");
        plan_groups.push(PlanGroup {
            seed: layers[first].preds[0],
            ext: layers[first].preds.iter().skip(1).copied().collect(),
            merge: layers[first].merge,
            in_dim: layers[first].in_dim,
            out_dim: layers[first].out_dim,
            insts,
        });
    }

    Ok(Generated {
        vml,
        args_text,
        consts: consts.clone(),
        slots,
        loop_groups,
        extra_bins,
        plan: Plan {
            model: model.clone(),
            layers: layers
                .iter()
                .map(|l| PlanLayer {
                    base: l.name.clone(),
                    orig: l.orig.clone(),
                    kind: l.kind.clone(),
                    attrs: l.attrs.clone(),
                    in_dim: l.in_dim,
                    out_dim: l.out_dim,
                    preds: l.preds.clone(),
                    merge: l.merge,
                    group: l.group,
                    gn: l.gn,
                })
                .collect(),
            order: order.clone(),
            slot_base: slot_base.clone(),
            consts: consts.clone(),
            t_seq,
            out_idx: out_idx.unwrap_or(0),
            groups: plan_groups,
            loss: loss.clone(),
            backward: backward.is_some(),
        },
    })
}

// ===========================================================================
// V0.1.3：汇入（一个层有多个前驱）
// ===========================================================================

/// 汇入方式：显式 `合并=相加|拼接` 优先，否则按前驱宽度自动定。
///
/// 自动规则只有两条，都是确定的、可预测的：
/// * **等宽 → 相加**（`EWADD`）—— 残差连接就是这种，也是绝大多数情况；
/// * **不等宽 → 拼接**（`CONCAT`）—— 相加在不等宽下没有定义。
///
/// 为什么不做"更聪明"的推断（比如按名字猜残差）：推断错就是**静默算错**，
/// 而宽度的相等/不等是**可判定**的事实。
fn merge_mode(l: &Layer, pdims: &[u32]) -> Result<MergeMode> {
    let explicit = attr_ident(&l.attrs, "merge").map(|s| crate::keywords::norm(&s));
    let all_equal = pdims.iter().all(|d| *d == pdims[0]);
    let mode = match explicit.as_deref() {
        Some(v)
            if matches!(
                v,
                "相加" | "加" | "求和" | "残差" | "add" | "sum" | "plus" | "residual"
            ) =>
        {
            MergeMode::Add
        }
        Some(v)
            if matches!(v, "拼接" | "拼" | "串接" | "连接" | "concat" | "cat" | "join" | "stack") =>
        {
            MergeMode::Concat
        }
        Some(v) => {
            return Err(Diag::error(
                "E100",
                format!("层 `{}` 的 `合并=` 值 `{v}` 不认识", l.name),
                l.span,
            )
            .with_candidates(vec![
                "相加 (add/residual)".to_string(),
                "拼接 (concat)".to_string(),
            ]))
        }
        None if all_equal => MergeMode::Add,
        None => MergeMode::Concat,
    };
    if mode == MergeMode::Add && !all_equal {
        return Err(Diag::error(
            "E100",
            format!(
                "层 `{}` 要相加，但前驱宽度不等（{}）—— 相加没有定义",
                l.name,
                pdims
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(" / ")
            ),
            l.span,
        )
        .with_hint("等宽才能相加；宽度不同要写 `合并=拼接`（CONCAT，宽度相加）"));
    }
    Ok(mode)
}

/// 发射汇入：把多个操作数合成 `dst`。
///
/// 3 个以上前驱时左折叠（2 个 2 个地合）。相加可以就地折叠 —— 引擎的
/// `EWADD` 对每个元素是"先读两个、再写同下标"，`dst == a` 是安全的
/// （`interp.rs` 的 `OP_EWADD` 分支）。拼接不行（宽度在长大），所以
/// 每折一次都要一个临时槽 + `COPY` 搬回来。
///
/// `operands` 是**已经解析好的槽位名**：展开版传各前驱的输出槽；
/// 循环版传 `[进位槽, 残差槽…]`（进位槽每圈代表"上一圈的输出"）。
fn emit_merge_ops(
    dst: &str,
    orig: &str,
    rows: u32,
    dim: u32,
    operands: &[String],
    pdims: &[u32],
    mode: MergeMode,
) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "    ; --- 汇入 `{}`：{} 个前驱，{} → 宽度 {dim} ---\n",
        orig,
        operands.len(),
        mode.name()
    ));
    s.push_str(&format!("    SHAPE   {dst}, {rows}, {dim}\n"));
    match mode {
        MergeMode::Add => {
            s.push_str(&format!("    EWADD   {dst}, {}, {}\n", operands[0], operands[1]));
            for p in &operands[2..] {
                s.push_str(&format!("    EWADD   {dst}, {dst}, {p}\n"));
            }
        }
        MergeMode::Concat => {
            s.push_str(&format!(
                "    CONCAT  {dst}, {}, {}\n",
                operands[0], operands[1]
            ));
            let mut w = pdims[0] + pdims[1];
            for (k, p) in operands[2..].iter().enumerate() {
                let tmp = format!("{dst}_f{}", k + 1);
                w += pdims[2 + k];
                // CONCAT 的 dst 宽度必须显式声明（汇编器按 ac+bc 走，不查 dc）
                s.push_str(&format!("    SHAPE   {tmp}, {rows}, {w}\n"));
                s.push_str(&format!("    CONCAT  {tmp}, {dst}, {p}\n"));
                s.push_str(&format!("    COPY    {dst}, {tmp}\n"));
            }
        }
    }
    s
}

/// 展开路径的汇入：操作数 = 各前驱的输出槽。
///
/// `dst` 由调用方给：`--slot-reuse` 时它可能是从暂存池里借来的同形状槽名
/// （形状固定 `(1, in_dim)`，与池的桶键一致）。
fn emit_merge_into(
    l: &Layer,
    pred_slots: &[String],
    pdims: &[u32],
    mode: MergeMode,
    dst: &str,
) -> String {
    emit_merge_ops(dst, &l.orig, 1, l.in_dim, pred_slots, pdims, mode)
}

/// 注意力块的参数（`维度` / `头` / `序列` / `因果`）。
///
/// 硬约束：`D` 必须能被 `头数` 整除（否则"每个头 hd 维"没有意义）。
/// 序列长度只影响**输入激活的行数**与 causal 掩码的语义。
fn attn_params(l: &Layer, d: u32) -> Result<(u32, u32, u32, u32, bool)> {
    let heads = attr_int(&l.attrs, "heads").ok_or_else(|| {
        Diag::error(
            "E105",
            format!("注意力层 `{}` 缺 `头`（注意力头数）", l.name),
            l.span,
        )
        .with_hint("写 `注意力 维度=64 头=4`")
    })?;
    if heads <= 0 {
        return Err(Diag::error(
            "E108",
            format!("注意力层 `{}` 的头数是 {heads}，必须 ≥1", l.name),
            l.span,
        ));
    }
    let heads = heads as u32;
    if d % heads != 0 {
        return Err(Diag::error(
            "E108",
            format!(
                "注意力层 `{}`：维度 {d} 不能被头数 {heads} 整除（每个头 {}.{} 维没有意义）",
                l.name,
                d / heads,
                d % heads
            ),
            l.span,
        )
        .with_candidates(vec![
            format!("维度=D 取 {heads} 的倍数，例如 {}", (d / heads + 1) * heads),
            "或改头数（1/2/4/8…）".to_string(),
        ]));
    }
    let seq = attr_int(&l.attrs, "seq").unwrap_or(1).max(1) as u32;
    let causal = attr_flag(&l.attrs, "causal", l.span, &l.name)?;
    Ok((d, heads, d / heads, seq, causal))
}

// ===========================================================================
// V0.1.3：注意力块（`注意力 维度=D 头=H [序列=T] [因果=是]`）
// ===========================================================================
//
// 汇编器里**没有** `ATTEN`/`HEAD` 这类助记符（76 个助记符里 grep 不到），
// 所以必须自己组合，而且每一段都要照顾到已知的契约/雷：
//
// ```text
//   Q = GEMM(x, Wq) + BIAS bq            ; [T,D]
//   K = GEMM(x, Wk) + BIAS bk
//   V = GEMM(x, Wv) + BIAS bv
//   每个头 h：                            ; hd = D/H
//     SLICE    q_h,  Q, h*hd, hd          ; 第 4 个操作数是**汇编期专用**的宽度
//     SLICE    k_h,  K, h*hd, hd          ;（它把 dst 的形状钉死；不进 .vmb）
//     SLICE    v_h,  V, h*hd, hd          ; SLICE 运行时**没有**任何守卫
//     TRANSPOSE kT, k_h                   ; [T,hd] → [hd,T]（不能用 RESHAPE：未实现）
//     GEMM     s_h, q_h, kT, nn           ; NN 布局：[T,hd]·[hd,T] → [T,T]
//     SCALE    s_h, s_h, scale            ; 1/sqrt(hd)（就地安全）
//     SOFTMAX  s_h, s_h [, causal]        ; causal 是**裸 flag**；它会清零上三角
//     GEMM     c_h, s_h, v_h, nn          ; [T,T]·[T,hd] → [T,hd]
//   CONCAT 拼接各头 → [T,D]（宽度在长，用临时槽左折叠）
//   GEMM out, ctx, Wo + BIAS bo
// ```
//
// 两个必须记住的雷：
// * `BMM` 也能做批量矩阵乘，但它在 `k==0` 时**整除零 panic**（不是 exit(9)），
//   而且完全忽略 `nn`/`acc` flag ⇒ 这里一律用 `GEMM ... nn`，不用 BMM。
// * `SOFTMAX` 的行宽来自 **dst**；causal 时 `lim = 行号+1`，把 `[lim, 列数)`
//   显式清零 ⇒ 正好是下三角掩码（第 0 行只看第 0 个 token）。

/// 一个头的中间槽名（都挂在层自己的槽基名上，槽位复用不会撞车）。
fn head_slot(base: &str, h: u32, what: &str) -> String {
    format!("{base}_{what}{h}")
}

/// 每头的**暂存槽**名（V0.1.6）。
///
/// `--slot-reuse` 开启时按**角色**命名（`<层>_hq` / `_hk` / `_hkt` / `_hv` / `_hs`）：
/// 这 5 个槽都是"写完立刻读、读完就不再有人读"，所以 H 个头可以**轮流借同一批**。
/// 关闭时保持 V0.1.5 的按头命名（`<层>_q0` …）—— 两条路的指令**语义完全相同**，
/// 只是"写到哪个物理槽"不同，因此数值必须逐字符一致（判据见 tools/slot_reuse.ps1 的注意力两例）。
///
/// 注意 `c`（每头输出）**不在**此列：H 个头的 `c` 必须一起活到拼头那一刻。
fn head_scratch(base: &str, h: u32, what: &str, reuse: bool) -> String {
    if reuse {
        format!("{base}_h{what}")
    } else {
        head_slot(base, h, what)
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_attn(
    l: &Layer,
    input_slot: &str,
    slot: &str,
    cprefix: &str,
    wd: DataType,
    reuse: bool,
    bins: &mut Vec<(String, [u8; 4])>,
    consts: &mut Vec<ConstDecl>,
) -> Result<String> {
    let d = l.out_dim;
    let (_, heads, hd, t, causal) = attn_params(l, d)?;
    let n = cprefix;
    let mut out = String::new();

    // 四个投影 + 四个偏置。只有权重能吃 bf16（GEMM 的 b）；
    // 偏置走 BIAS 指令，引擎装载期只允许 f32。
    for (tag, what) in [
        ("Wq", "查询投影权重 [D, D]"),
        ("Wk", "键投影权重 [D, D]"),
        ("Wv", "值投影权重 [D, D]"),
        ("Wo", "输出投影权重 [D, D]"),
    ] {
        if wd == DataType::Int8 {
            // V0.1.6：投影权重吃 int8（四个投影都是 `gemm_nt` 形态）。
            // 反量化 scale 的名字必须带投影名：`{层}_scale` 已经被 softmax 的 1/sqrt(hd) 占了。
            consts.push(ConstDecl {
                name: format!("{n}_{tag}"),
                file: format!("{n}_{tag}.bin"),
                rows: d,
                cols: d,
                what,
                dtype: DataType::Int8,
            });
            consts.push(ConstDecl {
                name: format!("{n}_s{tag}"),
                file: format!("{n}_s{tag}.bin"),
                rows: 1,
                cols: d,
                what: "每输出行的反量化 scale（f32[out]）",
                dtype: DataType::F32,
            });
        } else {
            consts.push(ConstDecl {
                name: format!("{n}_{tag}"),
                file: format!("{n}_{tag}.bin"),
                rows: d,
                cols: d,
                what,
                dtype: wd,
            });
        }
        consts.push(ConstDecl {
            name: format!("{n}_b{tag}"),
            file: format!("{n}_b{tag}.bin"),
            rows: 1,
            cols: d,
            what: "投影偏置 [1, D]",
            dtype: DataType::F32,
        });
    }
    // 缩放 1/sqrt(hd)：`SCALE` 的第 3 个操作数是**槽**，所以值得是个常量。
    // 它是 per-layer 的（不是共享常量）—— 循环化时循环体内不能读共享常量
    // （`layer_base` 会把共享常量一起平移，见 docs/TRAPS.md E3）。
    let scale_name = format!("{n}_scale");
    consts.push(ConstDecl {
        name: scale_name.clone(),
        file: format!("{scale_name}.bin"),
        rows: 1,
        cols: 1,
        what: "注意力缩放 1/sqrt(head_dim)",
        dtype: DataType::F32,
    });
    let sv = 1.0f32 / (hd as f32).sqrt();
    bins.push((format!("{scale_name}.bin"), sv.to_le_bytes()));

    out.push_str(&format!(
        "    ; --- 注意力 `{}`：D={d} 头={heads} 每头={hd} 序列={t}{} ---\n",
        l.orig,
        if causal { " 因果掩码" } else { "" }
    ));
    // int8 只服务 `--backend lib`：`.vmb`/汇编器那条路没有反量化 GEMM 指令，
    // main.rs 在进到这里之前就已经拒了（E109）。与 LINEAR 的 int8 一样，
    // 这里**不发射指令**，免得生成一份"看起来能汇编、其实跑不起来"的 .vml。
    if wd == DataType::Int8 {
        out.push_str(
            "    ; int8 量化投影：只服务 `--backend lib`（`--backend vml`/`--asm` 会 E109）\n",
        );
        return Ok(out);
    }
    if t == 1 {
        out.push_str(
            "    ; 提示：序列=1 时 softmax 只有一个元素参与，结果恒等于 V ——\n\
             ;        这个形状主要用来验**接线**（拆头/转置/拼接），值上没有惊喜。\n",
        );
    }
    for (src, dst) in [("q", "q"), ("k", "k"), ("v", "v")] {
        let col = format!("{slot}_{src}");
        out.push_str(&format!("    GEMM    {col}, {input_slot}, {n}_W{src}\n"));
        out.push_str(&format!("    BIAS    {col}, {n}_bW{src}\n"));
    }
    // 注意：BIAS 的偏置常量名是 `{n}_bWq`（`b` + `Wq`），与上面声明的
    // `{n}_bWq` 对齐 —— 见循环里的 tag 拼接。

    // 逐头：切、转、乘、缩、softmax、再乘
    for h in 0..heads {
        let off = h * hd;
        let qh = head_scratch(slot, h, "q", reuse);
        let kh = head_scratch(slot, h, "k", reuse);
        let kt = head_scratch(slot, h, "kt", reuse);
        let vh = head_scratch(slot, h, "v", reuse);
        let sh = head_scratch(slot, h, "s", reuse);
        let ch = head_slot(slot, h, "c"); // 每头独立：拼头时 H 份必须同时活着
        out.push_str(&format!("    SLICE   {qh}, {slot}_q, {off}, {hd}\n"));
        out.push_str(&format!("    SLICE   {kh}, {slot}_k, {off}, {hd}\n"));
        out.push_str(&format!("    SLICE   {vh}, {slot}_v, {off}, {hd}\n"));
        out.push_str(&format!("    TRANSPOSE {kt}, {kh}\n"));
        out.push_str(&format!("    GEMM    {sh}, {qh}, {kt}, nn\n"));
        out.push_str(&format!("    SCALE   {sh}, {sh}, {scale_name}\n"));
        out.push_str(&format!(
            "    SOFTMAX {sh}, {sh}{}\n",
            if causal { ", causal" } else { "" }
        ));
        out.push_str(&format!("    GEMM    {ch}, {sh}, {vh}, nn\n"));
    }

    // 拼头（宽度在长 ⇒ 临时槽 + COPY 搬回来）
    //
    // `--slot-reuse` 时目标**借 `_q`**：最后一次读 `_q` 是**最后一个头**的
    // `GEMM s, qh, kt`，正好在拼头之前；两者形状也相同（都是 [T, D]）。
    let ctx = if reuse {
        format!("{slot}_q")
    } else {
        format!("{slot}_ctx")
    };
    let head_slots: Vec<String> = (0..heads).map(|h| head_slot(slot, h, "c")).collect();
    if heads == 1 {
        out.push_str(&format!("    SHAPE   {ctx}, {t}, {hd}\n"));
        out.push_str(&format!("    COPY    {ctx}, {}\n", head_slots[0]));
    } else {
        let dims: Vec<u32> = vec![hd; heads as usize];
        out.push_str(&emit_merge_ops(
            &ctx,
            &format!("{} 的 {} 个头", l.orig, heads),
            t,
            d,
            &head_slots,
            &dims,
            MergeMode::Concat,
        ));
    }

    out.push_str(&format!("    GEMM    {slot}_out, {ctx}, {n}_Wo\n"));
    out.push_str(&format!("    BIAS    {slot}_out, {n}_bWo\n"));
    Ok(out)
}

/// 由层类型与上游维度推出 (输入维度, 输出维度)。
fn dims_of(l: &Layer, prev: Option<u32>) -> Result<(u32, u32)> {
    let dim = || -> Option<u32> {
        attr_int(&l.attrs, "dim").map(|n| n as u32).or(prev)
    };
    match l.kind.as_str() {
        "EMBED" => {
            let d = attr_int(&l.attrs, "dim").ok_or_else(|| {
                Diag::error("E105", format!("嵌入层 `{}` 缺 `dim`", l.name), l.span)
            })? as u32;
            Ok((0, d)) // EMBED 的"输入"是词元 id，不是激活向量
        }
        "SPARSE" => {
            let d = dim().ok_or_else(|| {
                Diag::error(
                    "E103",
                    format!("稀疏层 `{}` 的维度无法推断：既没写 `dim`，上游也没给出宽度", l.name),
                    l.span,
                )
            })?;
            Ok((d, d))
        }
        "LINEAR" => {
            let out = attr_int(&l.attrs, "out")
                .or_else(|| attr_int(&l.attrs, "in"))
                .ok_or_else(|| {
                    Diag::error("E105", format!("线性层 `{}` 缺 `out`", l.name), l.span)
                        .with_hint("写 `线性 输出=4054` 或 `Linear(out=4054)`")
                })? as u32;
            let ind = attr_int(&l.attrs, "in").map(|n| n as u32).or(prev).ok_or_else(|| {
                Diag::error(
                    "E103",
                    format!("线性层 `{}` 的输入维度无法推断：既没写 `in`，上游也没给出宽度", l.name),
                    l.span,
                )
                .with_hint("在最前面加一层 `嵌入`/`稀疏`，或写 `线性 输入=1536 输出=4054`")
            })?;
            Ok((ind, out))
        }
        "NORM" => {
            let d = dim().ok_or_else(|| {
                Diag::error("E103", format!("归一化层 `{}` 的维度无法推断", l.name), l.span)
            })?;
            Ok((d, d))
        }
        "ACT" => {
            let d = prev.ok_or_else(|| {
                Diag::error("E103", format!("激活层 `{}` 的维度无法推断（链首不能是激活）", l.name), l.span)
            })?;
            Ok((d, d))
        }
        "ATTN" => {
            // 注意力不改变宽度：输出 = 输入 = dim（`维度=` 缺省时沿用上游）
            let d = attr_int(&l.attrs, "dim").map(|n| n as u32).or(prev).ok_or_else(|| {
                Diag::error(
                    "E103",
                    format!("注意力层 `{}` 的维度推不出来：写 `维度=D`，或在它前面放一层", l.name),
                    l.span,
                )
            })?;
            let _ = attn_params(l, d)?; // 头数/整除/序列在这里就报错，别等到发射
            Ok((d, d))
        }
        other => Err(Diag::error(
            "E104",
            format!(
                "层类型 `{other}` 还不能编到 opcode（当前支持 嵌入/稀疏/线性/归一化/激活/注意力）"
            ),
            l.span,
        )
        .with_hint("自定义算子（`算子` / 裸 `@opcode`）排在 V0.2")),
    }
}

/// 反向遍历（V0.1.6）：从损失梯度出发，按**拓扑反序**把梯度传回去。
///
/// 契约**逐条来自引擎源码**（`core/src/interp.rs` 的操作数映射 + `core/src/train.rs`
/// 的数学），并在 `vmlm/examples/rnn/train/bptt_t2.vml` 的头部写成过一张表：
///
/// | 指令 | 操作数 | 语义 |
/// |---|---|---|
/// | `MSE_BWD` | `dst=预测, a=目标, b=梯度(写)` | 梯度落在**第 3 个**操作数（与 `CE_BWD` 不同：那边是第 2 个） |
/// | `GEMM_BWD_W` | `dst=dW, a=dy, b=x` | `dW[N,K] += dy^T x`（**累加** ⇒ 先 `ZERO`） |
/// | `GEMM_BWD_X` | `dst=dx, a=dy, b=W` | `dx = dy·W`（覆盖） |
/// | `BIAS_BWD` | `dst=db, a=dy` | `db += Σ dy`（**累加** ⇒ 先 `ZERO`） |
/// | `ACT_BWD` | `dst=dx, a=dy, b=ypre, kind` | `dx = dy ⊙ act'(ypre)` |
///
/// 为什么反向只能是**展开**路径：反向要读每一层的输入/输出激活，而
/// `--slot-reuse` 会把已经"读完"的槽回收掉、`--loop-repeat` 会让循环体每圈
/// 覆盖同一批槽 —— 两者都会让反向读到**被改写过的**激活（静默错答案）。
/// 所以它们在 `反向` 下都是硬错误（`E114`），而不是"小心一点"。
///
/// 返回值：`(指令文本, 要 PRINT 的梯度槽名)`。梯度打出来是**训练步唯一的可观测
/// 输出** —— 没有它，"反向算对了"就只能在驱动里凭引擎内部状态自证。
#[allow(clippy::too_many_arguments)]
fn emit_backward(
    model: &str,
    loss_kind: &str,
    out_slot: &str,
    out_dim: u32,
    t_seq: u32,
    order: &[usize],
    layers: &[Layer],
    layer_input: &[String],
    opts: &GenOptions,
    consts: &mut Vec<ConstDecl>,
) -> Result<(String, Vec<String>)> {
    if loss_kind != "mse" {
        return Err(Diag::error(
            "E112",
            format!("反向只接了均方误差（`mse`），不认识 `{loss_kind}`"),
            Span::ZERO,
        )
        .with_hint("写 `损失 均方`"));
    }
    if opts.loop_repeat && layers.iter().any(|l| l.gn >= 2) {
        return Err(Diag::error(
            "E114",
            "`反向` 不支持循环化的 `重复` 组（`--loop-repeat`）",
            Span::ZERO,
        )
        .with_hint(
            "循环体每圈覆盖同一批激活槽，反向读到的就不是第 k 圈的输入了。\
             去掉 `--loop-repeat`（展开成 N 份，前向行为不变），反向即可用。",
        ));
    }
    if opts.slot_reuse {
        return Err(Diag::error(
            "E114",
            "`反向` 与 `--slot-reuse` 不能同时用",
            Span::ZERO,
        )
        .with_hint(
            "槽位复用会把已经读完的激活槽回收给后面的层 —— 反向要读的正是那些激活。\
             去掉 `--slot-reuse` 就能训练（激活区会变大，这是正确性的代价）。",
        ));
    }
    if opts.weight_dtype != DataType::F32 {
        return Err(Diag::error(
            "E114",
            format!(
                "`反向` 只支持 f32 权重（当前 `{}`）",
                opts.weight_dtype.name()
            ),
            Span::ZERO,
        )
        .with_hint("引擎的反向内核按 f32 读写权重；bf16/int8 的梯度路径没有做过"));
    }

    let rows = t_seq.max(1);
    // 目标常量：与出口同形状。它是**输入**（和 f32/bf16 权重一样由使用者提供），
    // 编译器不做任何"造数据"的事。
    consts.push(ConstDecl {
        name: format!("{model}_tgt"),
        file: format!("{model}_tgt.bin"),
        rows,
        cols: out_dim,
        what: "损失目标（MSE 的目标值，与出口同形状）",
        dtype: DataType::F32,
    });

    let mut s = String::new();
    let mut prints: Vec<String> = Vec::new();
    let names: Vec<&str> = order.iter().map(|&i| layers[i].name.as_str()).collect();
    s.push_str("    ; ================= 反向（V0.1.6）=================\n");
    s.push_str(&format!(
        "    ; 损失 {}（拓扑反序：{}）\n",
        loss_kind,
        order
            .iter()
            .rev()
            .map(|&i| layers[i].name.as_str())
            .collect::<Vec<_>>()
            .join(" → ")
    ));
    s.push_str("    ; MSE_BWD 的梯度落在**第 3 个**操作数（与 CE_BWD 不同）；\n");
    s.push_str("    ; GEMM_BWD_W / BIAS_BWD 是**累加**的，所以每个参数先 ZERO。\n");
    let _ = names;
    s.push_str(&format!("    SHAPE   {model}_dloss, {rows}, {out_dim}\n"));
    s.push_str(&format!(
        "    MSE_BWD {out_slot}, {model}_tgt, {model}_dloss\n"
    ));

    let mut dy = format!("{model}_dloss");
    for &i in order.iter().rev() {
        let l = &layers[i];
        let n = &l.name;
        let input = &layer_input[i];
        s.push_str(&format!(
            "    ; --- 反向 {}（{}，入口槽 {}）---\n",
            n, l.kind, input
        ));
        match l.kind.as_str() {
            "LINEAR" => {
                s.push_str(&format!("    ZERO    g_{n}_W\n"));
                s.push_str(&format!("    ZERO    g_{n}_b\n"));
                s.push_str(&format!("    GEMM_BWD_W g_{n}_W, {dy}, {input}\n"));
                s.push_str(&format!("    BIAS_BWD   g_{n}_b, {dy}\n"));
                prints.push(format!("g_{n}_W"));
                prints.push(format!("g_{n}_b"));
                // 链首没有上游梯度要传（它的输入是模型输入常量）⇒ 不需要 GEMM_BWD_X
                if !l.preds.is_empty() {
                    let din = format!("{n}_din");
                    s.push_str(&format!("    SHAPE   {din}, {rows}, {}\n", l.in_dim));
                    s.push_str(&format!("    GEMM_BWD_X {din}, {dy}, {n}_W\n"));
                    dy = din;
                }
            }
            "ACT" => {
                let act = attr_ident(&l.attrs, "act").unwrap_or_else(|| "relu".to_string());
                let dact = format!("{n}_dact");
                s.push_str(&format!("    SHAPE   {dact}, {rows}, {}\n", l.in_dim));
                // 前激活就是这一层的输入槽（`ACT {n}_out, <input>, <act>`），它还在
                s.push_str(&format!("    ACT_BWD {dact}, {dy}, {input}, {act}\n"));
                dy = dact;
            }
            other => {
                return Err(Diag::error(
                    "E114",
                    format!("`{other}` 层没有接反向内核，`反向` 走不过去（层 `{}`）", l.name),
                    l.span,
                )
                .with_hint(
                    "本版反向只支持 `线性`（GEMM_BWD_W/X + BIAS_BWD）与 `激活`（ACT_BWD）。\
                     引擎里还有 LN_BWD / SOFTMAX_BWD / EMB_BWD / OUTER_BWD，\
                     但它们的槽位约定（gamma/beta、逐头、词表行号）各自不同，本版没有接。",
                ));
            }
        }
    }
    Ok((s, prints))
}

/// 单层展开：返回该层的 opcode 文本，并把用到的常量推进 `consts`。
/// `cprefix` 是**常量名**前缀，`slot` 是**槽位名**前缀。展开路径上两者相同
/// （= `l.name`）；循环化路径上槽位用最后一份、常量引用第 1 份 —— 这正是
/// `LAYERB` 能靠"基址平移"把第 1 份的常量名当模板用的原因。
#[allow(clippy::too_many_arguments)]
fn emit_layer_body(
    l: &Layer,
    input_slot: &str,
    slot: &str,
    cprefix: &str,
    rows: u32,
    wd: DataType,
    reuse: bool,
    bins: &mut Vec<(String, [u8; 4])>,
    consts: &mut Vec<ConstDecl>,
) -> Result<String> {
    let n = cprefix;
    let s = slot;
    let mut out = String::new();
    match l.kind.as_str() {
        "EMBED" => {
            let vocab = attr_int(&l.attrs, "vocab").unwrap_or(0) as u32;
            if vocab == 0 {
                return Err(Diag::error(
                    "E105",
                    format!("嵌入层 `{n}` 缺 `vocab`（词表大小）"),
                    l.span,
                )
                .with_hint("写 `嵌入 词表=8 维度=4`"));
            }
            consts.push(ConstDecl {
                name: format!("{n}_E"),
                file: format!("{n}_E.bin"),
                rows: vocab,
                cols: l.out_dim,
                what: "嵌入表 [vocab, dim]（EMB 的表）",
                // int8 只作用于**线性层权重**（`gemm_nt_int8` 的 q/scale 布局）：
                // 引擎的 `EMB` 不吃 int8，`.vmb` 装载期也会拒。这里回落成 f32。
                dtype: if wd == DataType::Int8 { DataType::F32 } else { wd },
            });
            if rows <= 1 {
                // 单行：输入槽就是那个词元 id（[1,1] 常量，直接当行号用）
                out.push_str(&format!("    SHAPE   {s}_out, 1, {}\n", l.out_dim));
                out.push_str(&format!("    EMB     {s}_out, {n}_E, {input_slot}\n"));
            } else {
                // 序列 T>1（V0.1.5）：**逐行查表**再拼成 [T, D]。
                //
                // 为什么不能指望一条指令搞定：`EMB` 一次只查一个行号、只写一行
                // （引擎侧 OP_EMB 用**表的列数**决定写多少，dst 形状不参与）。
                // 所以这里 T 次 SLICE+EMB，把每行转成列、按列左折叠拼接成 [D, T]，
                // 最后一次 TRANSPOSE 得到 [T, D]（dst 形状由 TRANSPOSE 推断）。
                // 输入常量因此是**一行 T 个词元 id**：`SLICE x, in, k, 1` 取第 k 个。
                //
                // 拼接不能就地写（`CONCAT dst, a, b` 的 dst 行跨步 = ac+bc，
                // 与 a 的行跨步不同，原地写会互相覆盖）⇒ 每一步一个新槽名。
                let d = l.out_dim;
                for k in 0..rows {
                    out.push_str(&format!("    ; --- 词元 {k}（共 {rows}）---\n"));
                    out.push_str(&format!("    SHAPE   {s}_tok{k}, 1, 1\n"));
                    out.push_str(&format!("    SLICE   {s}_tok{k}, {input_slot}, {k}, 1\n"));
                    out.push_str(&format!("    SHAPE   {s}_row{k}, 1, {d}\n"));
                    out.push_str(&format!("    EMB     {s}_row{k}, {n}_E, {s}_tok{k}\n"));
                    out.push_str(&format!("    SHAPE   {s}_c{k}, {d}, 1\n"));
                    out.push_str(&format!("    TRANSPOSE {s}_c{k}, {s}_row{k}\n"));
                }
                let mut prev = format!("{s}_c0");
                for k in 1..rows {
                    let cur = format!("{s}_stack{k}");
                    out.push_str(&format!("    SHAPE   {cur}, {d}, {}\n", k + 1));
                    out.push_str(&format!("    CONCAT  {cur}, {prev}, {s}_c{k}\n"));
                    prev = cur;
                }
                out.push_str(&format!("    TRANSPOSE {s}_out, {prev}\n"));
            }
        }
        "SPARSE" => {
            let k0 = attr_int(&l.attrs, "k0").unwrap_or(0) as u32;
            if k0 == 0 {
                return Err(Diag::error(
                    "E105",
                    format!("稀疏层 `{n}` 缺 `k0`（每个神经元的连接数）"),
                    l.span,
                )
                .with_hint("写 `稀疏 维度=4 连接=2`"));
            }
            let d = l.out_dim;
            let nnz = d * k0;
            consts.push(ConstDecl {
                name: format!("{n}_off"),
                file: format!("{n}_off.bin"),
                rows: d + 1,
                cols: 1,
                what: "CSR 行偏移（f32 存整数）",
                dtype: DataType::F32,
            });
            consts.push(ConstDecl {
                name: format!("{n}_src"),
                file: format!("{n}_src.bin"),
                rows: nnz,
                cols: 1,
                what: "CSR 列索引（f32 存整数）",
                dtype: DataType::F32,
            });
            consts.push(ConstDecl {
                name: format!("{n}_W"),
                file: format!("{n}_W.bin"),
                rows: nnz,
                cols: 1,
                what: "CSR 连接权重",
                dtype: DataType::F32,
            });
            // 引擎硬契约：SPARSE_MATVEC 的 dst 必须是**列向量** [n,1]
            //（interp.rs: "dst must be [n_neurons(,1)]"，且 offsets/src 必须是列向量）。
            // 之后的层要的是行向量，所以紧跟一条 TRANSPOSE —— 不能用 RESHAPE，
            // 汇编器把 RESHAPE 列在 UNIMPLEMENTED_OPS 里（"形状重解释未实现"）。
            out.push_str(&format!("    SHAPE   {s}_col, {d}, 1\n"));
            out.push_str(&format!(
                "    SPARSE_MATVEC {s}_col, {n}_off, {n}_src, {n}_W, {input_slot}\n"
            ));
            out.push_str(&format!("    TRANSPOSE {s}_out, {s}_col\n"));
        }
        "LINEAR" => {
            if wd == DataType::Int8 {
                // int8（V0.1.5 ⑥）：两个常量 —— 量化权重 `_Wq`（i8）+ 每输出行一个 `_scale`。
                // **只服务 `--backend lib`**：`.vmb`/汇编器那条路没有反量化 GEMM 指令，
                // main.rs 在进这里之前就会拒（E109）。所以这里不发射 GEMM/BIAS，
                // 只留一条注释，免得生成一份"看起来能汇编、其实跑不起来"的 `.vml`。
                consts.push(ConstDecl {
                    name: format!("{n}_Wq"),
                    file: format!("{n}_Wq.bin"),
                    rows: l.out_dim,
                    cols: l.in_dim,
                    what: "int8 量化权重 [out, in]（q: i8）",
                    dtype: DataType::Int8,
                });
                consts.push(ConstDecl {
                    name: format!("{n}_scale"),
                    file: format!("{n}_scale.bin"),
                    rows: 1,
                    cols: l.out_dim,
                    what: "每输出行的反量化 scale（f32[out]）",
                    dtype: DataType::F32,
                });
                out.push_str(&format!(
                    "    ; 层 `{n}`：int8 权重（--backend lib 专用；`.vmb` 路径没有反量化 GEMM）\n"
                ));
            } else {
                consts.push(ConstDecl {
                    name: format!("{n}_W"),
                    file: format!("{n}_W.bin"),
                    rows: l.out_dim,
                    cols: l.in_dim,
                    what: "权重 [out, in]（NT 布局；GEMM 的 b 操作数）",
                    dtype: wd,
                });
                out.push_str(&format!("    GEMM    {s}_out, {input_slot}, {n}_W\n"));
            }
            consts.push(ConstDecl {
                name: format!("{n}_b"),
                file: format!("{n}_b.bin"),
                rows: 1,
                cols: l.out_dim,
                what: "偏置 [1, out]",
                dtype: DataType::F32,
            });
            if wd != DataType::Int8 {
                out.push_str(&format!("    BIAS    {s}_out, {n}_b\n"));
            }
        }
        "NORM" => {
            let d = l.out_dim;
            consts.push(ConstDecl {
                name: format!("{n}_g"),
                file: format!("{n}_g.bin"),
                rows: 1,
                cols: d,
                what: "gamma [1, dim]",
                dtype: DataType::F32,
            });
            consts.push(ConstDecl {
                name: format!("{n}_b"),
                file: format!("{n}_b.bin"),
                rows: 1,
                cols: d,
                what: "beta [1, dim]",
                dtype: DataType::F32,
            });
            out.push_str(&format!("    LN      {s}_out, {input_slot}, {n}_g, {n}_b\n"));
        }
        "ACT" => {
            let act = attr_ident(&l.attrs, "act").unwrap_or_else(|| "relu".to_string());
            out.push_str(&format!("    ACT     {s}_out, {input_slot}, {act}\n"));
        }
        "ATTN" => {
            out.push_str(&emit_attn(l, input_slot, s, n, wd, reuse, bins, consts)?);
        }
        other => {
            return Err(Diag::error("E104", format!("层类型 `{other}` 不能编到 opcode"), l.span));
        }
    }
    Ok(out)
}

/// 展开路径的薄包装：槽位与常量同名。**不带槽位复用**（= V0.1.5 的默认行为）。
fn emit_layer(
    l: &Layer,
    input_slot: &str,
    wd: DataType,
    bins: &mut Vec<(String, [u8; 4])>,
    consts: &mut Vec<ConstDecl>,
) -> Result<String> {
    let name = l.name.clone();
    emit_layer_body(l, input_slot, &name, &name, 1, wd, false, bins, consts)
}

// ===========================================================================
// P0-2：循环发射
// ===========================================================================

/// 一个循环组发射出来的几样东西。
struct LoopEmit {
    body: String,
    /// 计数器/步长常量声明（要追加到池尾）。
    ctl: Vec<ConstDecl>,
    /// 前端自己产出的极小 `.bin`（f32 小端）。
    bins: Vec<(String, [u8; 4])>,
    slots: Vec<String>,
    group: LoopGroup,
    /// 循环**结束后**就死掉的暂存槽 `(行, 列, 槽名)` —— 进 `--slot-reuse` 的暂存池。
    /// 进位槽 / 汇入槽 / 计数器 / 两个步长槽都是"循环期间活着、循环后没人读"，
    /// 于是可以被后面的汇入借用（形状按 (行,列) 精确匹配，见调用点注释）。
    pool_in: Vec<(u32, u32, String)>,
}

/// 把声明推进 `consts`（同名去重），返回它们各自的下标。
fn push_decls(consts: &mut Vec<ConstDecl>, decls: Vec<ConstDecl>) -> Vec<usize> {
    let mut idx = Vec::new();
    for c in decls {
        match consts.iter().position(|x| x.name == c.name) {
            Some(p) => idx.push(p),
            None => {
                idx.push(consts.len());
                consts.push(c);
            }
        }
    }
    idx
}

/// 一个常量块的补齐字节数（汇编器对**每块**补齐到 `CONST_ALIGN`）。
fn padded_bytes(c: &ConstDecl) -> u64 {
    align_up(c.bytes(), CONST_ALIGN as u64)
}

/// 循环化的一个 `重复` 组。
///
/// `inst_names` 是组内各份实例名（`t__1 … t__N`）：常量名引用**第一份**，
/// 槽位名用**最后一份**（这样下游 `连接`/`前向` 的接线与展开版一字不改）。
///
/// `t_seq` 是序列长度（每层的行数）：循环体的进位槽/汇入槽都必须是 `[t_seq, D]`。
/// **实测踩过**：这里原来把行数写死成 1，于是 `序列=3` + `--loop-repeat` 的模型
/// 只算出**第一行**（输出 4 个数而不是 12 个），退出码 0 —— 静默错答案。
#[allow(clippy::too_many_arguments)]
fn emit_loop_group(
    l: &Layer,
    input_slot: &str,
    ext_slots: &[String],
    ext_dims: &[u32],
    inst_names: &[String],
    t_seq: u32,
    wd: DataType,
    reuse: bool,
    bins: &mut Vec<(String, [u8; 4])>,
    consts: &mut Vec<ConstDecl>,
) -> Result<LoopEmit> {
    let rows = t_seq.max(1);
    let n = inst_names.len() as u64;
    let cprefix = &inst_names[0];
    let slot_prefix = inst_names.last().unwrap();

    // 进位槽：循环体的**进**与**出**都走它；不用前驱槽直接当输入，
    // 是因为展开版里 `t__2` 读的是 `t__1_out`，循环版必须自己把上一圈
    // 的结果搬回来（`COPY` 是逐位拷贝，数值上与原版完全相同）。
    let carry = format!("{slot_prefix}__in");
    // 汇入（残差）时：进位槽再与外部前驱汇入一次，落在 `__m`；没有外部前驱
    // 就直接用进位槽，不多一条指令。
    let merge_slot = format!("{slot_prefix}__m");
    let body_input = if ext_slots.is_empty() { carry.clone() } else { merge_slot.clone() };
    let ctr = format!("{slot_prefix}__ctr");
    let ls = format!("{slot_prefix}__ls");
    let ns = format!("{slot_prefix}__ns");
    let lbl = format!("{slot_prefix}__l1");
    let iters_name = format!("{cprefix}_iters");
    let ls_name = format!("{cprefix}_lstride");
    let ns_name = format!("{cprefix}_nstride");

    // 循环体的输入必须是**激活槽**：常量槽会被 layer_base 平移，
    // 而输入常量（模型输入/词元 id）不是 per-layer 的，一起被平移就读错了。
    let in_cols = if l.kind == "EMBED" { 1 } else { l.in_dim };
    if in_cols == 0 {
        return Err(Diag::error(
            "E103",
            format!("循环组 `{}` 的输入维度是 0，无法建进位槽", l.orig),
            l.span,
        ));
    }

    let mut decl1: Vec<ConstDecl> = Vec::new();
    let ops = emit_layer_body(l, &body_input, slot_prefix, cprefix, rows, wd, reuse, bins, &mut decl1)?;

    // 每份实例各自的常量块（名字/文件名只差实例前缀）。
    let mut sets: Vec<Vec<usize>> = Vec::new();
    let mut extra_instance_bins: Vec<(String, [u8; 4])> = Vec::new();
    sets.push(push_decls(consts, decl1.clone()));
    for inst in &inst_names[1..] {
        let mut d: Vec<ConstDecl> = Vec::new();
        for c in &decl1 {
            d.push(ConstDecl {
                name: c.name.replacen(cprefix, inst, 1),
                file: c.file.replacen(cprefix, inst, 1),
                rows: c.rows,
                cols: c.cols,
                what: c.what,
                dtype: c.dtype,
            });
        }
        // 每份实例的常量块名不同 ⇒ 前端自己产出的极小 `.bin`（注意力缩放）
        // 也要按实例名复制一份，否则第 2 份的 `LAYERB` 走过去会读到不存在的文件。
        for (bn, bv) in bins.iter() {
            let renamed = bn.replacen(cprefix, inst, 1);
            if renamed != *bn {
                extra_instance_bins.push((renamed, *bv));
            }
        }
        sets.push(push_decls(consts, d));
    }
    bins.extend(extra_instance_bins);

    // 步长 = 第 1 份所有常量块的补齐尺寸之和。
    // 为什么是这个和：池里第 r+1 份的第 p 块的偏移 = 第 r 份的 + 步长，
    // 当且仅当每块的补齐尺寸在各份之间恒定 —— 形状相同必然成立，
    // 而"是否真的成立"由 `verify_loop_groups` 拿 `.vmb` 实测值对账。
    let stride = sets[0].iter().map(|&ci| padded_bytes(&consts[ci])).sum::<u64>();
    let total = stride * n;
    if !f32_exact_int(stride) || !f32_exact_int(total) {
        return Err(Diag::error(
            "E212",
            format!(
                "循环组 `{}` 的步长 {stride} B（×{n} = {total} B）不能被 f32 精确表示；\
                 `LAYERB` 的增量槽是 f32，落不到整数就会从第 2 层起静默读错权重",
                l.orig
            ),
            l.span,
        ));
    }

    let ctl = vec![
        ConstDecl {
            name: iters_name.clone(),
            file: format!("{iters_name}.bin"),
            rows: 1,
            cols: 1,
            what: "循环次数 N（f32；LOOP_CNT 是减完再判，初值就写 N）",
            dtype: DataType::F32,
        },
        ConstDecl {
            name: ls_name.clone(),
            file: format!("{ls_name}.bin"),
            rows: 1,
            cols: 1,
            what: "每层常量块字节步长（LAYERB +）",
            dtype: DataType::F32,
        },
        ConstDecl {
            name: ns_name.clone(),
            file: format!("{ns_name}.bin"),
            rows: 1,
            cols: 1,
            what: "步长还原 -N×步长（LAYERB −）",
            dtype: DataType::F32,
        },
    ];
    let bins = vec![
        (format!("{iters_name}.bin"), (n as f32).to_le_bytes()),
        (format!("{ls_name}.bin"), (stride as f32).to_le_bytes()),
        (format!("{ns_name}.bin"), (-(total as f32)).to_le_bytes()),
    ];

    let mut body = String::new();
    body.push_str(&format!(
        "    ; --- 重复组 `{}`：{} 层 → LOOP_CNT + LAYERB（指令流与层数无关）---\n",
        l.orig, n
    ));
    body.push_str(&format!("    SHAPE   {carry}, {rows}, {in_cols}\n"));
    body.push_str(&format!("    COPY    {carry}, {input_slot}\n"));
    // 常量槽直接写名字，**不能**写 `@name`：汇编器把 `@` 当跳转标签，
    // 只有 JMP/BRANCH/CMPLE/LOOP_CNT 接受标签，`COPY x, @c` 会 panic
    //（实测 `指令 COPY 不支持跳转标签`）。`@` 只出现在 LOOP_CNT 的目标上。
    body.push_str(&format!("    SHAPE   {ctr}, 1, 1\n    COPY    {ctr}, {iters_name}\n"));
    body.push_str(&format!("    SHAPE   {ls}, 1, 1\n    COPY    {ls}, {ls_name}\n"));
    body.push_str(&format!("    SHAPE   {ns}, 1, 1\n    COPY    {ns}, {ns_name}\n"));
    body.push_str(&format!("{lbl}:\n"));
    if !ext_slots.is_empty() {
        // 循环体内的汇入：进位槽（= 上一圈的输出）就是那条"链前驱"，
        // 外部前驱每圈都参与 ⇒ 展开版与循环版语义一致（见组归一化）。
        let mut operands = vec![carry.clone()];
        operands.extend(ext_slots.iter().cloned());
        let mut dims = vec![l.in_dim];
        dims.extend(ext_dims.iter().copied());
        body.push_str(&emit_merge_ops(
            &merge_slot,
            &l.orig,
            rows,
            l.in_dim,
            &operands,
            &dims,
            l.merge,
        ));
    }
    body.push_str(&ops);
    body.push_str(&format!("    COPY    {carry}, {slot_prefix}_out\n"));
    body.push_str(&format!("    LAYERB  {ls}\n"));
    body.push_str(&format!("    LOOP_CNT {ctr}, @{lbl}\n"));
    body.push_str(&format!("    LAYERB  {ns}\n"));

    let mut slots = vec![
        format!("{slot_prefix}_out"),
        carry.clone(),
        ctr.clone(),
        ls.clone(),
        ns.clone(),
    ];
    // 循环结束后就死的暂存槽（形状 = 它们被 SHAPE 成什么，见各自的发射点）
    let mut pool_in = vec![
        (rows, in_cols, carry.clone()), // SHAPE carry, rows, in_cols
        (1u32, 1u32, ctr.clone()),
        (1u32, 1u32, ls.clone()),
        (1u32, 1u32, ns.clone()),
    ];
    if !ext_slots.is_empty() {
        slots.push(merge_slot.clone());
        pool_in.push((rows, l.in_dim, merge_slot));
    }
    Ok(LoopEmit {
        body,
        ctl,
        bins,
        slots,
        group: LoopGroup { cprefix: cprefix.clone(), iters: n as u32, stride, sets },
        pool_in,
    })
}

/// 拿 `.vmb` 的常量表给循环组**对账**（P0-2 的核心校验）。
///
/// 三件事，任何一条不成立都返回 `Err`（调用方报 E210 并**删掉**产出的 `.vmb`）：
///
/// 1. **池序 = 清单序**：第 i 条常量的 `(rows, cols, len)` 必须与清单第 i 行一致。
///    这一条同时守住"`LAYERB` 走到的确实是同一批常量"。
/// 2. **步长恒定**：对每一份 r、每一个位置 p，
///    `off[r+1][p] - off[r][p]` 必须都等于同一个值。
/// 3. **实测步长 = 前端预测步长**：不然 `.vml` 里那个 `_lstride` 常量就是错的。
pub fn verify_loop_groups(gen: &Generated, ents: &[ConstEnt]) -> std::result::Result<u64, String> {
    if ents.len() != gen.consts.len() {
        return Err(format!(
            "常量条数不符：清单 {} 条，.vmb 里 {} 条",
            gen.consts.len(),
            ents.len()
        ));
    }
    for (i, (c, e)) in gen.consts.iter().zip(ents.iter()).enumerate() {
        let want = c.bytes();
        if e.rows != c.rows || e.cols != c.cols || e.len as u64 != want {
            return Err(format!(
                "第 {i} 条常量 `{}` 对不上：清单 {}×{}={} B，.vmb {}×{}={} B —— \
                 常量池顺序/长度的假设被打破",
                c.name, c.rows, c.cols, want, e.rows, e.cols, e.len
            ));
        }
    }
    let mut checked = 0u64;
    for g in &gen.loop_groups {
        if g.sets.iter().all(|s| s.is_empty()) {
            continue; // 该组没有常量（例如重复的激活层）：LAYERB 是空操作
        }
        let mut actual: Option<u64> = None;
        for r in 0..g.sets.len() - 1 {
            for p in 0..g.sets[r].len() {
                let (a, b) = (g.sets[r][p], g.sets[r + 1][p]);
                let d = ents[b].off as i64 - ents[a].off as i64;
                if d <= 0 {
                    return Err(format!(
                        "循环组 `{}` 第 {}→{} 份第 {} 个常量的偏移没前进（{d} B）",
                        g.cprefix, r + 1, r + 2, p
                    ));
                }
                match actual {
                    None => actual = Some(d as u64),
                    Some(x) if x == d as u64 => {}
                    Some(x) => {
                        return Err(format!(
                            "循环组 `{}` 的步长不恒定：第 {}→{} 份的第 {} 个常量（0 起）差 {} B，\
                             同一组的其它位置差 {} B —— `LAYERB` 的单一增量假设被打破",
                            g.cprefix, r + 1, r + 2, p, d, x
                        ))
                    }
                }
            }
        }
        let Some(a) = actual else { continue };
        if a != g.stride {
            return Err(format!(
                "循环组 `{}` 的步长对不上：前端预测 {} B，.vmb 实测 {} B",
                g.cprefix, g.stride, a
            ));
        }
        checked += 1;
    }
    Ok(checked)
}

/// 便捷入口：源码 → opcode `.vml`（默认选项）。
pub fn generate_from_src(
    ast: &crate::ast::Ast,
    src_path: &str,
) -> Result<Generated> {
    generate_from_src_with(ast, src_path, GenOptions::default())
}

/// 便捷入口（带后端开关）。
pub fn generate_from_src_with(
    ast: &crate::ast::Ast,
    src_path: &str,
    opts: GenOptions,
) -> Result<Generated> {
    let lowered = lower(ast)?;
    generate_with(&lowered, src_path, opts)
}
