//! Sparrow V0.1 —— 关键字表与词汇归一化（对应任务书第四、五、九部分）
//!
//! 三个独立的表，职责不同，**容错策略也不同**（这是刻意的边界判断）：
//!
//! | 表 | 作用 | 匹配策略 | 理由 |
//! |---|---|---|---|
//! | [`KEYWORDS`] | 结构关键字（模型/连接/重复/块标记/标点） | 精确 + 唯一前缀 | 任务书第二部分：`连`=`连接`=`conn`=`connect` |
//! | [`KINDS`] | 层类型（Embed/Sparse/Linear/…） | **仅精确** | 验收 11 要求 `Spars` 报"未定义的层类型"并给建议；前缀匹配会把它当合法 |
//! | [`ATTRS`] | 属性键（词表/维度/连接/输出/…） | 精确 + 唯一前缀 | 与结构关键字同级 |
//!
//! 三张表都只做**词汇归一化**：把 N 种写法映射到 1 个 canonical form。
//! canonical form 只出现在 codegen 输出里，AST 保留原文（第八部分要求
//! "中英文关键字在 Lexer 层统一成 canonical form，AST 里只有一种写法"
//! —— 指的是同义写法归一，不是把中文改写成英文）。

use crate::ast::Expr;

// ---------------------------------------------------------------------------
// 文本归一化小工具
// ---------------------------------------------------------------------------

/// 大小写不敏感比较。只对 ASCII 折叠，中文原样。
pub fn eq_norm(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

/// ASCII 小写化（中文不受影响）。
pub fn norm(s: &str) -> String {
    s.to_ascii_lowercase()
}

/// `a` 归一化之后是否以 `w` 开头（`w` 必须**已经**归一化）。
///
/// 为什么不写 `norm(a).starts_with(w)`：那会为**每一个别名**分配一个 `String`。
/// 关键字表有 21 类、上百个别名，而 `classes()` 在解析每个词时都会被调用 ——
/// 实测 1600 层模型（3600 个词）里，这几行分配占了**解析时间的 ~95%**
/// （365 ms → 13 ms，见 `tools/perf_frontend.ps1`）。
///
/// 等价性：`to_ascii_lowercase` 只改 ASCII 字节且不改变长度，所以
/// "折叠后前缀" ⟺ "逐字节 ASCII 不敏感前缀"。
pub fn starts_with_norm(a: &str, w: &str) -> bool {
    a.len() > w.len() && a.as_bytes()[..w.len()].eq_ignore_ascii_case(w.as_bytes())
}

/// `s` 是否以 `pat` 开头（ASCII 大小写不敏感，**不分配**，允许长度相等）。
pub fn starts_with_ci(s: &str, pat: &str) -> bool {
    s.len() >= pat.len() && s.as_bytes()[..pat.len()].eq_ignore_ascii_case(pat.as_bytes())
}

/// 前缀匹配资格：中文 ≥1 字即可（`连` → `连接`），ASCII 需 ≥3 字符
/// （`conn` → `connect`），否则 `e`/`t`/`o`/`a` 这类正常标识符会被吞成关键字。
pub fn prefix_eligible(w: &str) -> bool {
    let n = w.chars().count();
    if w.is_ascii() {
        n >= 3
    } else {
        n >= 1
    }
}

/// 稳定的浮点文本（`2.0` → `2`），保证 .vml 输出可复现。
pub fn fmt_f64(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
}

// ---------------------------------------------------------------------------
// 结构关键字
// ---------------------------------------------------------------------------

/// canonical 结构关键字类。
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum Kw {
    Model,
    Layer,
    Connect,
    Forward,
    Repeat,
    To,
    And,
    Start,
    End,
    Colon,
    Semicolon,
    Comma,
    LParen,
    RParen,
    // --- V0.1 扩展类（任务书示例里出现、但第九部分的表未列出）---
    Op,
    Opcode,
    Args,
    As,
    From,
    Then,
    Filler,
}

impl Kw {
    pub const ALL: [Kw; 21] = [
        Kw::Model,
        Kw::Layer,
        Kw::Connect,
        Kw::Forward,
        Kw::Repeat,
        Kw::To,
        Kw::And,
        Kw::Start,
        Kw::End,
        Kw::Colon,
        Kw::Semicolon,
        Kw::Comma,
        Kw::LParen,
        Kw::RParen,
        Kw::Op,
        Kw::Opcode,
        Kw::Args,
        Kw::As,
        Kw::From,
        Kw::Then,
        Kw::Filler,
    ];

    pub const fn bit(self) -> u32 {
        1u32 << (self as u8)
    }

    /// 报错/文档里用的名字。
    pub fn name(self) -> &'static str {
        match self {
            Kw::Model => "model",
            Kw::Layer => "layer",
            Kw::Connect => "connect",
            Kw::Forward => "forward",
            Kw::Repeat => "repeat",
            Kw::To => "to",
            Kw::And => "and",
            Kw::Start => "start",
            Kw::End => "end",
            Kw::Colon => "colon",
            Kw::Semicolon => "semicolon",
            Kw::Comma => "comma",
            Kw::LParen => "lparen",
            Kw::RParen => "rparen",
            Kw::Op => "op",
            Kw::Opcode => "opcode",
            Kw::Args => "args",
            Kw::As => "as",
            Kw::From => "from",
            Kw::Then => "then",
            Kw::Filler => "filler",
        }
    }
}

/// 关键字类集合（一个词可以同时属于多类，例如 `到` ∈ {connect, to}、`,` ∈ {and, comma}）。
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct KwSet(pub u32);

impl KwSet {
    pub const fn empty() -> KwSet {
        KwSet(0)
    }

    pub const fn of(kw: Kw) -> KwSet {
        KwSet(kw.bit())
    }

    pub fn insert(&mut self, kw: Kw) {
        self.0 |= kw.bit();
    }

    pub fn union(self, other: KwSet) -> KwSet {
        KwSet(self.0 | other.0)
    }

    pub fn contains(self, kw: Kw) -> bool {
        self.0 & kw.bit() != 0
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    pub fn iter(self) -> impl Iterator<Item = Kw> {
        Kw::ALL.into_iter().filter(move |k| self.contains(*k))
    }
}

impl std::fmt::Debug for KwSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = self.iter().map(|k| k.name()).collect();
        write!(f, "{{{}}}", names.join("|"))
    }
}

/// 结构关键字表（第九部分的表 + 示例里实际用到的虚词）。
pub struct KeywordTable {
    pub model: &'static [&'static str],
    pub layer: &'static [&'static str],
    pub connect: &'static [&'static str],
    pub forward: &'static [&'static str],
    pub repeat: &'static [&'static str],
    pub to: &'static [&'static str],
    pub and: &'static [&'static str],
    pub start: &'static [&'static str],
    pub end: &'static [&'static str],
    pub colon: &'static [&'static str],
    pub semicolon: &'static [&'static str],
    pub comma: &'static [&'static str],
    pub lparen: &'static [&'static str],
    pub rparen: &'static [&'static str],
    // 扩展类
    pub op: &'static [&'static str],
    pub opcode: &'static [&'static str],
    pub args: &'static [&'static str],
    pub as_kw: &'static [&'static str],
    pub from: &'static [&'static str],
    pub then: &'static [&'static str],
    pub filler: &'static [&'static str],
}

impl KeywordTable {
    /// 21 个类与别名，一次遍历用。
    pub fn classes_iter(&self) -> [(Kw, &'static [&'static str]); 21] {
        [
            (Kw::Model, self.model),
            (Kw::Layer, self.layer),
            (Kw::Connect, self.connect),
            (Kw::Forward, self.forward),
            (Kw::Repeat, self.repeat),
            (Kw::To, self.to),
            (Kw::And, self.and),
            (Kw::Start, self.start),
            (Kw::End, self.end),
            (Kw::Colon, self.colon),
            (Kw::Semicolon, self.semicolon),
            (Kw::Comma, self.comma),
            (Kw::LParen, self.lparen),
            (Kw::RParen, self.rparen),
            (Kw::Op, self.op),
            (Kw::Opcode, self.opcode),
            (Kw::Args, self.args),
            (Kw::As, self.as_kw),
            (Kw::From, self.from),
            (Kw::Then, self.then),
            (Kw::Filler, self.filler),
        ]
    }

    /// `word` 归属的所有关键字类。
    ///
    /// 先精确匹配（大小写不敏感）；没有精确匹配时再做**唯一前缀**匹配：
    /// `word` 是某个别名的真前缀。前缀匹配受 [`prefix_eligible`] 约束。
    pub fn classes(&self, word: &str) -> KwSet {
        let w = norm(word);
        let mut exact = KwSet::empty();
        for (kw, list) in self.classes_iter() {
            for a in list {
                if eq_norm(a, word) {
                    exact.insert(kw);
                }
            }
        }
        if !exact.is_empty() {
            return exact;
        }
        if !prefix_eligible(&w) {
            return KwSet::empty();
        }
        let mut pref = KwSet::empty();
        for (kw, list) in self.classes_iter() {
            for a in list {
                if starts_with_norm(a, &w) {
                    pref.insert(kw);
                }
            }
        }
        pref
    }

    pub fn is(&self, word: &str, kw: Kw) -> bool {
        self.classes(word).contains(kw)
    }

    /// 全部 `(类, canonical 名, 别名)`，报告用。
    pub fn dump(&self) -> Vec<(Kw, Vec<&'static str>)> {
        self.classes_iter()
            .into_iter()
            .map(|(kw, list)| (kw, list.to_vec()))
            .collect()
    }
}

/// 训练/反向相关词的**类别**（V0.1.6）。
///
/// V0.1.5 时这三类**都**是"语言没有这个功能"的边界；V0.1.6 起 `损失` / `反向`
/// 已经是**真语法**（`Stmt::Loss` / `Stmt::Backward`），只有"参数更新 / 训练循环"
/// 一类仍然是可判定的边界（`E111`）。
///
/// 保留这张表而不是删掉它，是因为它同时服务两件事：
/// ① 边界诊断仍然要点名"缺的是哪一类"；② 把这些词当**实例名**用时不能被误伤。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TrainingWord {
    /// `损失` / `误差` / `loss` …（V0.1.6 起是真语法）
    Loss,
    /// `反向` / `反传` / `梯度` / `backward` …（V0.1.6 起是真语法）
    Backward,
    /// `训练` / `优化` / `train` / `optimizer` …（仍是边界）
    Train,
}

/// 词 → 类别（见 [`TrainingWord`]）。
pub fn training_word_kind(name: &str) -> Option<TrainingWord> {
    const LOSS: &[&str] = &["损失", "误差", "loss", "目标函数", "objective"];
    const BACKWARD: &[&str] = &["反向", "反传", "梯度", "backward", "bwd", "grad", "gradient"];
    const TRAIN: &[&str] = &["训练", "训练器", "优化", "优化器", "train", "trainer", "optimizer"];
    let n = norm(name);
    if LOSS.iter().any(|w| n == *w) {
        return Some(TrainingWord::Loss);
    }
    if BACKWARD.iter().any(|w| n == *w) {
        return Some(TrainingWord::Backward);
    }
    if TRAIN.iter().any(|w| n == *w) {
        return Some(TrainingWord::Train);
    }
    None
}

/// 损失函数的规范名（V0.1.6）。
///
/// 现在只有**均方误差**（引擎的 `OP_MSE_BWD`）；`均方` / `平方误差` / `mse`
/// 都归到 `mse`。不认识就返回 `None`，由调用方报 `E112` 并列出候选
/// —— 与层类型"不认识的写法要报错 + 给候选"一致。
pub fn loss_kind_canonical(name: &str) -> Option<&'static str> {
    const MSE: &[&str] = &["均方", "均方差", "均方误差", "平方误差", "mse", "mse_loss", "l2"];
    let n = norm(name);
    if MSE.iter().any(|w| n == *w) {
        return Some("mse");
    }
    None
}

/// 损失函数的候选名（报错提示用）。
pub const LOSS_CANDIDATES: &[&str] = &["均方", "mse"];

/// 训练/反向相关的词（V0.1.5 ⑦）：它们**不是**层类型，也不该被当成层引用。
///
/// 返回"缺的是什么"，供诊断提示用。V0.1.6 起 `损失`/`反向` 已经是真语法，
/// 因此本函数只在**边界诊断**里用（`训练` 一类），见 [`training_word_kind`]。
///
/// 只收**明确**的训练词（不收 `步`/`step`/`fit` 这类可能撞上用户命名的词）。
pub fn training_word(name: &str) -> Option<&'static str> {
    const LOSS: &[&str] = &["损失", "误差", "loss", "目标函数", "objective"];
    const BACKWARD: &[&str] = &["反向", "反传", "梯度", "backward", "bwd", "grad", "gradient"];
    const TRAIN: &[&str] = &["训练", "训练器", "优化", "优化器", "train", "trainer", "optimizer"];
    let n = norm(name);
    if LOSS.iter().any(|w| n == *w) {
        return Some("损失声明");
    }
    if BACKWARD.iter().any(|w| n == *w) {
        return Some("反向遍历（按拓扑反序）");
    }
    if TRAIN.iter().any(|w| n == *w) {
        return Some("参数更新（优化器 / 训练循环）");
    }
    None
}

/// 唯一的全局关键字表。
pub static KEYWORDS: KeywordTable = KeywordTable {
    model: &["模型", "网络", "model", "net", "network", "M"],
    layer: &["层", "定义", "layer", "def", "L"],
    connect: &["连接", "连", "接", "串", "到", "->", "→", "connect", "conn", "link"],
    forward: &["前向", "预测", "推理", "forward", "fwd", "infer", "predict"],
    repeat: &["重复", "复用", "叠", "堆", "循环", "repeat", "rep", "x", "*", "做"],
    to: &["到", "至", "->", "→", "to"],
    and: &["和", "与", "以及", "、", ",", "，", ";", "；", "and"],
    start: &["开始", "起", "{", "【", "begin", "start"],
    end: &["结束", "止", "完", "}", "】", "。", "end", "stop"],
    colon: &[":", "："],
    semicolon: &[";", "；"],
    comma: &[",", "，", "、"],
    lparen: &["(", "（", "["],
    rparen: &[")", "）", "]"],
    op: &["算子", "自定义算子", "op", "operator", "算子定义"],
    opcode: &["opcode", "操作码", "指令", "指令号"],
    args: &["args", "参数", "形参", "arity"],
    as_kw: &["用", "为", "把", "以", "as", "with", "using"],
    from: &["从", "自", "from", "src"],
    then: &["再", "然后", "接着", "then", "next"],
    filler: &["的", "个", "次", "就", "了", "让", "使", "将", "一下"],
};

// ---------------------------------------------------------------------------
// 层类型
// ---------------------------------------------------------------------------

/// 一个层类型：canonical 名 + 别名 + 位置参数绑定规则。
pub struct KindDef {
    /// .vml 里唯一写法（全大写 ASCII，语言无关）。
    pub canonical: &'static str,
    /// 报错建议里显示的名字（`Sparse` 而不是 `SPARSE`）。
    pub display: &'static str,
    pub aliases: &'static [&'static str],
    /// arity ≥ 2 时，位置参数按此序列绑定规范属性键。
    pub params: &'static [&'static str],
    /// arity == 1 时的快捷绑定（`嵌入 512` → `dim=512`）。
    pub single_key: Option<&'static str>,
}

impl KindDef {
    pub fn matches(&self, word: &str) -> bool {
        self.aliases.iter().any(|a| eq_norm(a, word))
    }
}

/// 层类型表。`稀疏块` 这类用户自定义层类型不在这里 —— 它们在 `sema.rs` 里解析。
pub static KINDS: &[KindDef] = &[
    KindDef {
        canonical: "EMBED",
        display: "Embed",
        aliases: &["嵌入", "嵌", "词嵌入", "embed", "embedding", "emb"],
        params: &["vocab", "dim"],
        single_key: Some("dim"),
    },
    KindDef {
        canonical: "SPARSE",
        display: "Sparse",
        aliases: &["稀疏", "稀", "稀疏层", "sparse", "sp"],
        params: &["dim", "k0"],
        single_key: Some("dim"),
    },
    KindDef {
        canonical: "LINEAR",
        display: "Linear",
        aliases: &["线性", "线", "全连接", "linear", "lin", "dense", "fc"],
        params: &["in", "out"],
        single_key: Some("out"),
    },
    KindDef {
        canonical: "NORM",
        display: "Norm",
        aliases: &["归一化", "层归一化", "范数", "norm", "layernorm", "rmsnorm", "ln"],
        params: &["dim"],
        single_key: None,
    },
    KindDef {
        canonical: "ACT",
        display: "Activation",
        aliases: &["激活", "激活函数", "act", "activation"],
        params: &["act"],
        single_key: None,
    },
    KindDef {
        canonical: "ATTN",
        display: "Attention",
        aliases: &["注意力", "自注意力", "多头注意力", "attn", "attention", "mha"],
        params: &["dim", "heads"],
        single_key: Some("dim"),
    },
];

/// 层类型精确查表。**不做前缀匹配**（见模块头部说明）。
pub fn kind_lookup(word: &str) -> Option<&'static KindDef> {
    KINDS.iter().find(|k| k.matches(word))
}

pub fn kind_is_known(word: &str) -> bool {
    kind_lookup(word).is_some()
}

/// 层类型建议：`Spars` → `Sparse`。
/// 候选同时考虑 canonical 名与全部别名，返回 canonical 名（去重、稳定排序）。
pub fn kind_suggest(word: &str) -> Vec<&'static str> {
    let mut scored: Vec<(usize, &'static str)> = Vec::new();
    for k in KINDS {
        let mut best = levenshtein(&norm(word), &norm(k.canonical));
        for a in k.aliases {
            let d = levenshtein(&norm(word), &norm(a));
            if d < best {
                best = d;
            }
        }
        let limit = suggest_limit(word);
        if best <= limit {
            scored.push((best, k.display));
        }
    }
    scored.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(b.1)));
    scored.into_iter().map(|(_, c)| c).collect()
}

/// canonical → 显示名（报错建议用 `Sparse` 而不是 `SPARSE`）。
pub fn kind_display(canonical: &str) -> &str {
    KINDS
        .iter()
        .find(|k| k.canonical == canonical)
        .map(|k| k.display)
        .unwrap_or(canonical)
}

/// 建议阈值：短词允许距离 1，长词允许 2。
pub fn suggest_limit(word: &str) -> usize {
    let n = word.chars().count();
    if n <= 3 {
        1
    } else {
        2
    }
}

// ---------------------------------------------------------------------------
// 属性键
// ---------------------------------------------------------------------------

pub struct AttrDef {
    pub canonical: &'static str,
    pub aliases: &'static [&'static str],
}

pub static ATTRS: &[AttrDef] = &[
    AttrDef { canonical: "vocab", aliases: &["词表", "词汇表", "词表大小", "vocab", "vocab_size", "vsize", "v"] },
    AttrDef { canonical: "dim", aliases: &["维度", "维", "隐藏维度", "dim", "d", "d_model", "model_dim", "hidden_size", "hsize"] },
    AttrDef { canonical: "k0", aliases: &["连接", "连", "连接数", "稀疏度", "k0", "k", "sparse_k", "nnz"] },
    AttrDef { canonical: "in", aliases: &["输入", "入", "输入维度", "in", "input", "in_dim", "in_features"] },
    AttrDef { canonical: "out", aliases: &["输出", "出", "输出维度", "out", "output", "out_dim", "out_features"] },
    AttrDef { canonical: "hidden", aliases: &["隐藏", "隐", "隐藏层", "中间维度", "hidden", "hidden_dim", "hid", "ffn_dim"] },
    AttrDef { canonical: "heads", aliases: &["头", "头数", "注意力头", "heads", "n_head", "n_heads", "num_heads", "nh"] },
    AttrDef { canonical: "act", aliases: &["激活", "激活函数", "act", "activation", "fn"] },
    AttrDef { canonical: "norm", aliases: &["归一化类型", "norm_type", "norm"] },
    AttrDef { canonical: "eps", aliases: &["极小值", "精度", "eps", "epsilon"] },
    AttrDef { canonical: "layers", aliases: &["层数", "深度", "layers", "n_layer", "n_layers", "num_layers", "depth"] },
    AttrDef { canonical: "bias", aliases: &["偏置", "bias"] },
    AttrDef { canonical: "tie", aliases: &["权重共享", "共享权重", "tie", "tied"] },
    AttrDef { canonical: "scale", aliases: &["缩放", "scale"] },
    AttrDef { canonical: "seq", aliases: &["序列长度", "长度", "seq", "seq_len", "ctx"] },
    AttrDef { canonical: "batch", aliases: &["批大小", "批次", "batch", "bs"] },
    // V0.1.3 汇入（多前驱合流）：一个层有两个前驱时，先把它们合起来再喂给它。
    // 值决定合的方式：相加（EWADD）/ 拼接（CONCAT）；不写就按宽度自动定。
    AttrDef { canonical: "merge", aliases: &["汇入", "合并", "合流", "残差", "融合", "merge", "combine", "residual", "fuse"] },
    // V0.1.3 注意力块：因果掩码（`causal`）。值是"是/真/1"这类真假词。
    AttrDef { canonical: "causal", aliases: &["因果", "因果掩码", "掩码", "下三角", "causal", "mask"] },
];

/// canonical 输出顺序（未知键排在后面，按名字典序）。
pub const ATTR_ORDER: &[&str] = &[
    "vocab", "dim", "k0", "in", "out", "hidden", "heads", "act", "norm", "eps", "layers", "seq",
    "batch", "bias", "tie", "scale", "merge", "causal",
];

pub fn attr_rank(key: &str) -> usize {
    ATTR_ORDER.iter().position(|k| *k == key).unwrap_or(usize::MAX)
}

/// 属性键归一化：`词表`/`vocab`/`v` → `vocab`。未知键原样返回（宽容：不报错）。
pub fn attr_canonical(key: &str) -> String {
    for d in ATTRS {
        if d.aliases.iter().any(|a| eq_norm(a, key)) {
            return d.canonical.to_string();
        }
    }
    if prefix_eligible(key) {
        let w = norm(key);
        let mut hit: Option<&'static str> = None;
        let mut count = 0;
        for d in ATTRS {
            let mut m = false;
            for a in d.aliases {
                if starts_with_norm(a, &w) {
                    m = true;
                    break;
                }
            }
            if m {
                if hit != Some(d.canonical) {
                    count += 1;
                    hit = Some(d.canonical);
                }
            }
        }
        if count == 1 {
            if let Some(h) = hit {
                return h.to_string();
            }
        }
    }
    key.to_string()
}

/// 属性键建议（拼错时给候选）。
pub fn attr_suggest(word: &str) -> Vec<&'static str> {
    let limit = suggest_limit(word);
    let mut scored: Vec<(usize, &'static str)> = Vec::new();
    for d in ATTRS {
        let mut best = levenshtein(&norm(word), &norm(d.canonical));
        for a in d.aliases {
            let dd = levenshtein(&norm(word), &norm(a));
            if dd < best {
                best = dd;
            }
        }
        if best <= limit {
            scored.push((best, d.canonical));
        }
    }
    scored.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(b.1)));
    scored.into_iter().map(|(_, c)| c).collect()
}

// ---------------------------------------------------------------------------
// 激活函数取值归一化
// ---------------------------------------------------------------------------

pub static ACT_VALUES: &[&str] = &["relu", "relu6", "gelu", "silu", "swish", "tanh", "sigmoid", "softmax", "leaky_relu"];

/// `ReLU`/`relu`/`RELU` → `relu`；未知取值只做 ASCII 小写（宽容）。
pub fn act_value_canonical(v: &str) -> String {
    norm(v)
}

// ---------------------------------------------------------------------------
// 通用建议
// ---------------------------------------------------------------------------

/// 字符级编辑距离（中文按字算）。
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// 把中文复合词贪心切成关键字别名序列（要求 ≥2 段）：`连到` = `连`+`到`。
///
/// 只在 parser 期待关键字的位置调用。**不能放进 lexer**：`嵌入层` 也能切成
/// `嵌入`+`层`，但第四/七部分要求 `嵌入层` 是一个标识符；而且这种切分是有
/// 上下文的（只在连接词/前向词该出现的地方才做）。
pub fn split_aliases(word: &str) -> Option<Vec<Kw>> {
    let w = norm(word);
    let mut rest: &str = &w;
    let mut out: Vec<Kw> = Vec::new();
    while !rest.is_empty() {
        let mut best: Option<(usize, Kw)> = None;
        for (kw, list) in KEYWORDS.classes_iter() {
            for a in list {
                // 不分配：`rest` 已经归一化，别名逐字节 ASCII 不敏感比较即可
                if !a.is_empty() && starts_with_ci(rest, a) {
                    let l = a.len();
                    if best.map(|(bl, _)| l > bl).unwrap_or(true) {
                        best = Some((l, kw));
                    }
                }
            }
        }
        let (len, kw) = best?;
        rest = &rest[len..];
        out.push(kw);
    }
    if out.len() >= 2 {
        Some(out)
    } else {
        None
    }
}

/// 连接类复合词：`连到` = `连`+`到`，`再连到` = `再`+`连`+`到`。
pub fn split_connectors(word: &str) -> Option<Vec<Kw>> {
    const CONN: [Kw; 4] = [Kw::Connect, Kw::To, Kw::Then, Kw::And];
    let parts = split_aliases(word)?;
    if parts.iter().all(|k| CONN.contains(k)) {
        Some(parts)
    } else {
        None
    }
}

/// 前向类复合词：`前向到` = `前向`+`到`。
pub fn split_forward(word: &str) -> Option<(Kw, Vec<Kw>)> {
    const CONN: [Kw; 4] = [Kw::Connect, Kw::To, Kw::Then, Kw::And];
    let mut parts = split_aliases(word)?;
    if parts.first() != Some(&Kw::Forward) {
        return None;
    }
    let head = parts.remove(0);
    if parts.iter().all(|k| CONN.contains(k)) {
        Some((head, parts))
    } else {
        None
    }
}

/// 从候选里挑最像 `word` 的若干个（用于"候选："行）。
pub fn suggest_from<'a>(word: &str, candidates: &[&'a str], max: usize) -> Vec<&'a str> {
    let limit = suggest_limit(word);
    let mut scored: Vec<(usize, &'a str)> = candidates
        .iter()
        .map(|c| (levenshtein(&norm(word), &norm(c)), *c))
        .filter(|(d, c)| *d <= limit || norm(c).starts_with(&norm(word).chars().take(2).collect::<String>()))
        .collect();
    scored.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(b.1)));
    scored.into_iter().map(|(_, c)| c).take(max).collect()
}

// ---------------------------------------------------------------------------
// 位置参数绑定（KindDef::params / single_key 的消费者）
// ---------------------------------------------------------------------------

/// 位置参数 → 规范属性键。返回 `(键, 值)`。
///
/// 绑定规则（刻意的确定性设计，见报告"边界判断"）：
/// - `EMBED`: 1 个 → `(dim,)`；2 个 → `(vocab, dim)`
/// - `SPARSE`: 1 个 → `(dim,)`；2 个 → `(dim, k0)`
/// - `LINEAR`: 1 个 → `(out,)`；2 个 → `(in, out)`
/// - 其余：按 `params` 顺序，多余的丢弃（宽容）
pub fn bind_kind_positional(kind: &str, args: &[Expr]) -> Vec<(String, Expr)> {
    let Some(def) = kind_lookup(kind) else {
        return args
            .iter()
            .enumerate()
            .map(|(i, e)| (format!("arg{i}"), e.clone()))
            .collect();
    };
    if args.len() == 1 {
        if let Some(k) = def.single_key {
            return vec![(k.to_string(), args[0].clone())];
        }
    }
    args.iter()
        .enumerate()
        .map(|(i, e)| {
            let key = def
                .params
                .get(i)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("arg{i}"));
            (key, e.clone())
        })
        .collect()
}

/// 位置参数是否被显式命名覆盖 —— 用于报"重复给参"。
pub fn kind_param_keys(kind: &str) -> &'static [&'static str] {
    kind_lookup(kind).map(|d| d.params).unwrap_or(&[])
}

// ---------------------------------------------------------------------------
// 词表自检（模块 1 验收：多套写法 → 同一 canonical 类）
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keywords_exact_and_prefix() {
        let t = &KEYWORDS;
        for w in ["连接", "连", "接", "connect", "conn"] {
            assert!(t.is(w, Kw::Connect), "{w} 应为 connect");
        }
        for w in ["模型", "网络", "model", "M", "模"] {
            assert!(t.is(w, Kw::Model), "{w} 应为 model");
        }
        for w in ["前向", "预测", "forward", "fwd"] {
            assert!(t.is(w, Kw::Forward), "{w} 应为 forward");
        }
        for w in ["重复", "叠", "堆", "repeat", "x", "*", "做"] {
            assert!(t.is(w, Kw::Repeat), "{w} 应为 repeat");
        }
        // 到 ∈ {connect, to}，`再` 是 then
        assert!(t.is("到", Kw::Connect) && t.is("到", Kw::To));
        assert!(t.is("再", Kw::Then));
        assert!(t.is("conn", Kw::Connect), "ASCII 三字符前缀应生效");
        assert!(t.is("conne", Kw::Connect));
    }

    #[test]
    fn short_ascii_identifiers_are_not_keywords() {
        // 关键边界：e/t/o/a 是常见实例名，绝不能被前缀匹配吞掉
        for w in ["e", "t", "o", "a", "i", "s1", "x1", "xx", "input", "net1"] {
            let set = KEYWORDS.classes(w);
            assert!(set.is_empty(), "{w} 不应是关键字，却得到 {set:?}");
        }
        // 但 spec 明列的单字母别名要生效
        assert!(KEYWORDS.is("x", Kw::Repeat));
        assert!(KEYWORDS.is("m", Kw::Model));
        assert!(KEYWORDS.is("l", Kw::Layer));
    }

    #[test]
    fn chinese_identifiers_are_not_swallowed() {
        for w in ["嵌入层", "主干", "输出层", "入", "干", "出", "_tmp", "层数", "连接数", "维"] {
            let set = KEYWORDS.classes(w);
            assert!(set.is_empty(), "{w} 不应是关键字，却得到 {set:?}");
        }
    }

    #[test]
    fn kinds_are_exact_only() {
        assert_eq!(kind_lookup("稀疏").unwrap().canonical, "SPARSE");
        assert_eq!(kind_lookup("稀").unwrap().canonical, "SPARSE");
        assert_eq!(kind_lookup("Sparse").unwrap().canonical, "SPARSE");
        assert_eq!(kind_lookup("嵌").unwrap().canonical, "EMBED");
        assert_eq!(kind_lookup("线").unwrap().canonical, "LINEAR");
        // 验收 11：Spars 必须失败并给出 Sparse 建议
        assert!(kind_lookup("Spars").is_none());
        assert!(kind_suggest("Spars").contains(&"Sparse"));
        // 用户自定义类型名不能被前缀吞成内置类型
        assert!(kind_lookup("稀疏块").is_none());
    }

    #[test]
    fn attrs_canonicalize() {
        assert_eq!(attr_canonical("词表"), "vocab");
        assert_eq!(attr_canonical("Vocab"), "vocab");
        assert_eq!(attr_canonical("v"), "vocab");
        assert_eq!(attr_canonical("维度"), "dim");
        assert_eq!(attr_canonical("维"), "dim");
        assert_eq!(attr_canonical("连接"), "k0");
        assert_eq!(attr_canonical("输出"), "out");
        assert_eq!(attr_canonical("入"), "in");
        assert_eq!(attr_canonical("未知键"), "未知键");
    }

    #[test]
    fn positional_binding_is_deterministic() {
        let one = vec![Expr::Int(1536)];
        let two = vec![Expr::Int(4054), Expr::Int(1536)];
        assert_eq!(bind_kind_positional("嵌入", &one)[0].0, "dim");
        let b = bind_kind_positional("嵌入", &two);
        assert_eq!(b[0].0, "vocab");
        assert_eq!(b[1].0, "dim");
        let b = bind_kind_positional("稀疏", &two);
        assert_eq!((b[0].0.as_str(), b[1].0.as_str()), ("dim", "k0"));
        let b = bind_kind_positional("线性", &two);
        assert_eq!((b[0].0.as_str(), b[1].0.as_str()), ("in", "out"));
        assert_eq!(bind_kind_positional("线性", &one)[0].0, "out");
    }

    #[test]
    fn connector_compounds() {
        // `到` 同时属于 connect 与 to 两个类（第九部分的表就是这样），
        // 所以只断言"切成了 2 段且都是连接类"，不断言具体取哪一类。
        let parts = split_connectors("连到").expect("连到 应切成 2 段");
        assert_eq!(parts.len(), 2);
        assert!(parts.iter().all(|k| [Kw::Connect, Kw::To, Kw::Then, Kw::And].contains(k)));
        assert_eq!(split_connectors("再连到").map(|v| v.len()), Some(3));
        // 单个连接词由 kw_set 负责，不走切分
        assert_eq!(split_connectors("连"), None);
        // 标识符不能被切碎
        assert_eq!(split_connectors("嵌入层"), None);
        assert_eq!(split_connectors("主干"), None);
        // `前向到` 是"前向 + 到"，但它不是连接类复合词
        assert_eq!(split_connectors("前向到"), None);
        // `到` 同属 connect/to，两种归法都算对
        let (head, tail) = split_forward("前向到").expect("前向到 应切开");
        assert_eq!(head, Kw::Forward);
        assert_eq!(tail.len(), 1);
        assert!([Kw::Connect, Kw::To].contains(&tail[0]));
        assert_eq!(split_forward("前向"), None, "单段不走切分");
        assert_eq!(split_forward("嵌入层"), None);
    }

    #[test]
    fn levenshtein_basics() {
        assert_eq!(levenshtein("spars", "sparse"), 1);
        assert_eq!(levenshtein("稀疏", "稀疏"), 0);
        assert_eq!(levenshtein("稀疏", "稀"), 1);
    }
}
