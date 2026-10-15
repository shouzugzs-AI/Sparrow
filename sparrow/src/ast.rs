//! Sparrow V0.1 —— AST（对应任务书第八部分）
//!
//! 四条硬约束：
//! 1. 每个节点携带 [`Span`]（源文件字节区间），用于错误定位；
//! 2. 参数列表同时支持位置参数与命名参数，命名参数在 AST 里是 [`Expr::Named`]；
//! 3. [`Stmt::Repeat`] 是结构化节点，不是文本替换；
//! 4. 中英文关键字/标点在 lexer 层归一化，AST 里只有一种写法。
//!
//! AST 只承载语义。写法差异（缩进、标点、关键字语言、块机制）一律不进 AST。

use std::fmt;

/// 源文件字节区间 `[start, end)`。
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub const ZERO: Span = Span { start: 0, end: 0 };

    pub fn new(start: usize, end: usize) -> Span {
        Span { start: start as u32, end: end as u32 }
    }

    /// 覆盖两个 span 的最小 span。
    pub fn join(self, other: Span) -> Span {
        Span { start: self.start.min(other.start), end: self.end.max(other.end) }
    }

    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }

    pub fn contains(&self, other: Span) -> bool {
        self.start <= other.start && other.end <= self.end
    }
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

/// 标识符。中文标识符是一等公民：`name` 就是源码里的原文，不做任何转写。
///
/// 同义词归一化**不**发生在 AST 里；`Embed`/`嵌入`/`嵌` 是三个不同的 `Ident`
/// 文本，归一化发生在 codegen 的 canonical form（见 `keywords.rs`）。
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

impl Ident {
    pub fn new(name: impl Into<String>, span: Span) -> Ident {
        Ident { name: name.into(), span }
    }

    pub fn as_str(&self) -> &str {
        &self.name
    }

    /// 大小写不敏感的文本比较（中文不受影响）。
    pub fn is(&self, other: &str) -> bool {
        crate::keywords::eq_norm(&self.name, other)
    }
}

impl fmt::Debug for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}@{}..{}", self.name, self.span.start, self.span.end)
    }
}

impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

/// 属性列表：`(键, 值)`，保持源码顺序。
pub type Attrs = Vec<(Ident, Expr)>;

/// 顶层 AST。
#[derive(Clone, Debug, PartialEq)]
pub enum Ast {
    Program { items: Vec<Item> },
}

/// 顶层条目。
#[derive(Clone, Debug, PartialEq)]
pub enum Item {
    /// `模型 小: ...` / `model small: ...`
    ModelDef {
        name: Ident,
        body: Vec<Stmt>,
        span: Span,
    },
    /// `定义 稀疏块 用 维 连 为: 稀疏 维度=维 连接=连`
    LayerDef {
        name: Ident,
        params: Vec<Ident>,
        kind: Ident,
        attrs: Attrs,
        span: Span,
    },
}

/// 语句。
#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    /// `e = 嵌入 1536` / `嵌入 1536` / `Sparse(1536, 8)`
    ///
    /// 没有显式名字时，`name` == `kind` 的原文（`嵌入 512` → name=`嵌入`, kind=`嵌入`），
    /// parser 因此不需要往 AST 里塞 `explicit_name` 标志位。
    Instantiate {
        name: Ident,
        kind: Ident,
        args: Vec<Expr>,
        span: Span,
    },
    /// `t = 稀疏块(1536, 8) 重复24次` / `... x 24` / `... * 24`
    Repeat {
        body: Box<Stmt>,
        count: Expr,
        span: Span,
    },
    /// `连接 嵌入 稀疏 线性` / `把 e 连到 t 再连到 o` / `a -> b -> c`
    Connect {
        path: Vec<Ident>,
        span: Span,
    },
    /// `前向 嵌入 线性` / `从 e 前向到 o`
    Forward {
        input: Ident,
        output: Ident,
        span: Span,
    },
    /// `算子 我的激活 = opcode(200) args(2)`
    CustomOpDef {
        name: Ident,
        opcode: Expr,
        args: Vec<Ident>,
        span: Span,
    },
    /// `@load r0 cpool+0`
    RawOp {
        opcode: Ident,
        operands: Vec<Expr>,
        span: Span,
    },
    /// `损失 均方` —— 声明训练用的损失函数（V0.1.6）。
    ///
    /// `kind` 是**原文**（`均方` / `mse` / `平方误差` …），canonical 化在 sema
    /// （与层类型同一套做法：AST 保持原文，规范化只发生一次）。
    Loss {
        kind: Ident,
        span: Span,
    },
    /// `反向` —— 要求按拓扑反序发射反向遍历（V0.1.6）。
    ///
    /// 它**不是**层，也不参与 `连接`；它是模型级的一条陈述：有了损失之后，
    /// 反向从损失梯度出发，逐层把梯度传回去。
    Backward {
        span: Span,
    },
}

/// 表达式。位置参数就是裸 `Expr`；命名参数是 [`Expr::Named`]。
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Int(i64),
    Float(f64),
    Ident(String),
    Named { key: Ident, value: Box<Expr> },
    Arrow(Vec<Expr>),
}

impl Ast {
    pub fn items(&self) -> &[Item] {
        match self {
            Ast::Program { items } => items,
        }
    }

    pub fn items_mut(&mut self) -> &mut Vec<Item> {
        match self {
            Ast::Program { items } => items,
        }
    }

    /// 第一个 `ModelDef`（codegen 只认单模型文件）。
    pub fn main_model(&self) -> Option<&Item> {
        self.items().iter().find(|i| matches!(i, Item::ModelDef { .. }))
    }
}

impl Item {
    pub fn span(&self) -> Span {
        match self {
            Item::ModelDef { span, .. } | Item::LayerDef { span, .. } => *span,
        }
    }

    pub fn name(&self) -> &Ident {
        match self {
            Item::ModelDef { name, .. } | Item::LayerDef { name, .. } => name,
        }
    }

    pub fn body(&self) -> &[Stmt] {
        match self {
            Item::ModelDef { body, .. } => body,
            Item::LayerDef { .. } => &[],
        }
    }

    pub fn is_model(&self) -> bool {
        matches!(self, Item::ModelDef { .. })
    }

    pub fn is_layer_def(&self) -> bool {
        matches!(self, Item::LayerDef { .. })
    }
}

impl Stmt {
    pub fn span(&self) -> Span {
        match self {
            Stmt::Instantiate { span, .. }
            | Stmt::Repeat { span, .. }
            | Stmt::Connect { span, .. }
            | Stmt::Forward { span, .. }
            | Stmt::CustomOpDef { span, .. }
            | Stmt::RawOp { span, .. }
            | Stmt::Loss { span, .. }
            | Stmt::Backward { span, .. } => *span,
        }
    }

    /// 语句类型名，报错/JSON 用。
    pub fn kind(&self) -> &'static str {
        match self {
            Stmt::Instantiate { .. } => "Instantiate",
            Stmt::Repeat { .. } => "Repeat",
            Stmt::Connect { .. } => "Connect",
            Stmt::Forward { .. } => "Forward",
            Stmt::CustomOpDef { .. } => "CustomOpDef",
            Stmt::RawOp { .. } => "RawOp",
            Stmt::Loss { .. } => "Loss",
            Stmt::Backward { .. } => "Backward",
        }
    }

    /// 该语句是否声明了一个实例（`Instantiate` / `Repeat{Instantiate}`）。
    pub fn declared_instance(&self) -> Option<(&Ident, &Ident, &[Expr])> {
        match self {
            Stmt::Instantiate { name, kind, args, .. } => Some((name, kind, args)),
            Stmt::Repeat { body, .. } => body.declared_instance(),
            _ => None,
        }
    }

    /// 结构化遍历直接子语句（`Repeat` 递归进 body）。
    pub fn walk<'a>(&'a self, f: &mut impl FnMut(&'a Stmt)) {
        f(self);
        if let Stmt::Repeat { body, .. } = self {
            body.walk(f);
        }
    }
}

impl Expr {
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Expr::Int(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_ident(&self) -> Option<&str> {
        match self {
            Expr::Ident(s) => Some(s),
            _ => None,
        }
    }

    /// 命名参数取值；位置参数返回 `None`。
    pub fn named(&self, key: &str) -> Option<&Expr> {
        match self {
            Expr::Named { key: k, value } if k.is(key) => Some(value),
            _ => None,
        }
    }

    pub fn key(&self) -> Option<&Ident> {
        match self {
            Expr::Named { key, .. } => Some(key),
            _ => None,
        }
    }

    pub fn is_named(&self) -> bool {
        matches!(self, Expr::Named { .. })
    }

    /// 稳定的文本形态，用于 Canonical 输出与 JSON。
    pub fn render(&self) -> String {
        match self {
            Expr::Int(n) => n.to_string(),
            Expr::Float(x) => crate::keywords::fmt_f64(*x),
            Expr::Ident(s) => s.clone(),
            Expr::Named { key, value } => format!("{}={}", key.name, value.render()),
            Expr::Arrow(items) => items.iter().map(|e| e.render()).collect::<Vec<_>>().join(" -> "),
        }
    }
}

/// 统计信息（报告用）。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AstStats {
    pub models: usize,
    pub layer_defs: usize,
    pub stmts: usize,
    pub instantiates: usize,
    pub repeats: usize,
    pub connects: usize,
    pub forwards: usize,
    pub custom_ops: usize,
    pub raw_ops: usize,
    pub named_args: usize,
}

impl Ast {
    pub fn stats(&self) -> AstStats {
        let mut s = AstStats::default();
        for item in self.items() {
            match item {
                Item::ModelDef { body, .. } => {
                    s.models += 1;
                    for st in body {
                        count_stmt(st, &mut s);
                    }
                }
                Item::LayerDef { attrs, .. } => {
                    s.layer_defs += 1;
                    s.named_args += attrs.iter().filter(|(_, v)| v.is_named()).count();
                }
            }
        }
        s
    }
}

fn count_stmt(st: &Stmt, s: &mut AstStats) {
    s.stmts += 1;
    match st {
        Stmt::Instantiate { args, .. } => {
            s.instantiates += 1;
            s.named_args += args.iter().filter(|a| a.is_named()).count();
        }
        Stmt::Repeat { body, .. } => {
            s.repeats += 1;
            count_stmt(body, s);
        }
        Stmt::Connect { .. } => s.connects += 1,
        Stmt::Forward { .. } => s.forwards += 1,
        Stmt::CustomOpDef { .. } => s.custom_ops += 1,
        Stmt::RawOp { .. } => s.raw_ops += 1,
        // V0.1.6：训练陈述。`stmts` 已经计过数，这里不再单列计数器
        // （AstStats 是对外契约，加字段要动 JSON 形状；这两条按"其它语句"归类）。
        Stmt::Loss { .. } | Stmt::Backward { .. } => {}
    }
}
