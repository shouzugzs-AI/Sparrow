//! Sparrow V0.1 —— Parser（对应任务书第十部分）
//!
//! Parser 不预期特定 token 序列，而是**模式识别**。`parse_stmt` 的五种模式：
//!
//! | 模式 | 触发 | 例 |
//! |---|---|---|
//! | Instantiate | `词 [= \| :] 词 [(]` 或 词 (｜是层类型) | `e = 嵌入 1536`、`稀疏块(1536, 8)`、`嵌入 512` |
//! | Repeat | 语句后跟 `重复/x/*/24层` | `t = 稀疏块(1536, 8) 重复24次` |
//! | Connect | `连接`/`把…连到…再连到…`/裸标识符列表/`a -> b -> c` | `连接 嵌入 稀疏 线性` |
//! | Forward | `前向`/`从 X 前向到 Y` | `前向 嵌入 线性` |
//! | CustomOpDef | `算子 名 = opcode(N) args(M)` | `算子 我的激活 = opcode(200) args(2)` |
//! | RawOp | `@op ...` | `@load r0 cpool+0` |
//!
//! 宽容的语法在 lexer/词表层归一化，Parser 只处理 canonical 形式。
//!
//! 两处刻意的边界判断：
//! 1. **裸标识符列表用"整行前瞻"判定**：`嵌入层;主干;输出层` 里的 `;` 既是语句分隔符
//!    也是列表分隔符（第二、三部分冲突）。规则：若从当前位置到逻辑行结束只有
//!    ≥2 个词 + 分隔/连接词，则整行是一个 Connect 路径；只要出现数字/括号就不再是列表。
//! 2. **隐式模型**：Level 0 只有 `嵌入 512`，没有 `模型` 头。按第八部分
//!    `Item` 只有 ModelDef/LayerDef，所以这些顶层语句被包进一个名字为
//!    `main` 的隐式 ModelDef（语言中立的名字，中英对照才能产出同一个 .vml）。

use crate::ast::{Ast, Attrs, Expr, Ident, Item, Span, Stmt};
use crate::diag::{Diag, Result, SourceFile};
use crate::keywords::{self, Kw, KwSet, KEYWORDS, KINDS};
use crate::lexer::{lex, cn_number_value, is_cn_digit, Punct, TokKind, Token};

const IMPLICIT_MODEL: &str = "main";

/// 实例化的书写顺序。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Order {
    /// `e = 嵌入 1536`
    NameEqKind,
    /// `e 嵌入 1536`（示例 G）
    NameKind,
    /// `嵌 e 512`（Level 3 / 示例 E）
    KindName,
    /// `嵌入 512`
    KindOnly,
}

/// 不能作为参数继续吃进去的结构关键字（遇到就说明下一条语句开始了）。
fn arg_stop_kw() -> KwSet {
    let mut s = KwSet::empty();
    for k in [
        Kw::Model,
        Kw::Layer,
        Kw::Connect,
        Kw::Forward,
        Kw::Repeat,
        Kw::Op,
        Kw::Start,
        Kw::End,
        Kw::To,
        Kw::Then,
        Kw::From,
        Kw::As,
    ] {
        s.insert(k);
    }
    s
}

pub struct Parser<'a> {
    src: &'a SourceFile,
    toks: Vec<Token>,
    i: usize,
    /// 刚吃掉了 `@` 转义符：下一个标识符即使撞上关键字也放行（第四部分）。
    escaped_ident: bool,
}

// ---------------------------------------------------------------------------
// 入口
// ---------------------------------------------------------------------------

pub fn parse(src: &SourceFile) -> Result<Ast> {
    let toks = lex(src)?;
    Parser::new(src, toks).parse_program()
}

pub fn parse_tokens<'a>(src: &'a SourceFile, toks: Vec<Token>) -> Result<Ast> {
    Parser::new(src, toks).parse_program()
}

impl<'a> Parser<'a> {
    fn new(src: &'a SourceFile, toks: Vec<Token>) -> Parser<'a> {
        Parser { src, toks, i: 0, escaped_ident: false }
    }

    // -- 基础游标 ----------------------------------------------------------

    fn peek(&self) -> &Token {
        &self.toks[self.i.min(self.toks.len() - 1)]
    }

    fn at(&self, k: usize) -> &Token {
        &self.toks[(self.i + k).min(self.toks.len() - 1)]
    }

    fn bump(&mut self) -> Token {
        let t = self.peek().clone();
        if self.i + 1 < self.toks.len() {
            self.i += 1;
        }
        t
    }

    fn eat_punct(&mut self, p: Punct) -> bool {
        if self.peek().is_punct(p) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn eat_kw(&mut self, kw: Kw) -> bool {
        if self.peek().is_kw(kw) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn is_eof(&self) -> bool {
        matches!(self.peek().kind, TokKind::Eof)
    }

    /// 语句结束：换行、分号、句号、感叹号、反缩进、块闭合、EOF。
    fn at_stmt_end(&self) -> bool {
        match &self.peek().kind {
            TokKind::Eof | TokKind::Newline | TokKind::Dedent => true,
            TokKind::Punct(p) => matches!(p, Punct::Semi | Punct::Dot | Punct::Bang | Punct::RBrace | Punct::RBracket),
            _ => false,
        }
    }

    /// 跳过纯分隔符（不动 Indent/Dedent，块边界由块循环处理）。
    fn skip_separators(&mut self) {
        while matches!(self.peek().kind, TokKind::Newline)
            || self.peek().is_punct(Punct::Semi)
            || self.peek().is_punct(Punct::Dot)
            || self.peek().is_punct(Punct::Bang)
        {
            self.bump();
        }
    }

    /// 跳过虚词（`的`/`个`/`次`/`一下`…）。
    fn skip_fillers(&mut self) {
        while self.peek().is_kw(Kw::Filler) && self.peek().is_word() {
            // 注意：仅当它不是 `key=` 的键时才跳
            if self.at(1).is_punct(Punct::Eq) {
                break;
            }
            self.bump();
        }
    }

    fn error(&self, code: &'static str, msg: impl Into<String>, span: Span) -> Diag {
        Diag::error(code, msg, span)
    }

    fn err_here(&self, code: &'static str, msg: impl Into<String>) -> Diag {
        let t = self.peek();
        Diag::error(code, msg, t.span)
    }

    /// 期望一个标识符：裸词或引号转义词都算。结构关键字在此报 E007。
    ///
    /// 成功时**消费**该 token（早期版本漏了 `bump()`，导致模型名/层类型名留在
    /// 游标上被当成参数再吃一遍）。
    fn expect_ident(&mut self, what: &str) -> Result<Ident> {
        let t = self.peek().clone();
        match &t.kind {
            TokKind::Word(w) => {
                let set = t.kw_set();
                let escaped = std::mem::take(&mut self.escaped_ident);
                if !set.is_empty() && !escaped && !self.ident_ok_keyword(w) {
                    let kws: Vec<&str> = set.iter().map(|k| k.name()).collect();
                    return Err(Diag::error(
                        "E007",
                        format!("\"{w}\" 是关键字（{}），不能作为{what}", kws.join("/")),
                        t.span,
                    )
                    .with_hint(format!("改用 `@{w}` 或 `\"{w}\"` 转义，或者换个名字"))
                    .with_note("第四部分：关键字保留，除非用 @ 或引号转义"));
                }
                self.bump();
                Ok(Ident::new(w.clone(), t.span))
            }
            TokKind::Str(w) => {
                self.bump();
                Ok(Ident::new(w.clone(), t.span))
            }
            _ => Err(Diag::error(
                "E005",
                format!("这里需要{what}，实际是{}", t.describe()),
                t.span,
            )),
        }
    }

    /// 关键字里允许当标识符的例外（避免把 `维`/`连` 这类常见参数名一刀切掉）。
    fn ident_ok_keyword(&self, w: &str) -> bool {
        // 只有"层类型/参数名"用途的词才放行；纯结构词（模型/连接/重复/块标记）一律拒绝
        const ALLOW: &[&str] = &["连", "维", "到", "至", "用", "为", "把", "从", "再", "和", "与", "接", "起", "止", "做"];
        ALLOW.iter().any(|a| keywords::eq_norm(a, w))
    }

    // -- 顶层 --------------------------------------------------------------

    fn parse_program(mut self) -> Result<Ast> {
        let mut items: Vec<Item> = Vec::new();
        loop {
            self.skip_separators();
            // 块结尾残留（`模型 小 结束` 之类）不应出现在顶层
            if self.is_eof() {
                break;
            }
            if self.is_item_header(Kw::Model) {
                items.push(self.parse_model()?);
                continue;
            }
            if self.is_item_header(Kw::Layer) {
                items.push(self.parse_layer_def()?);
                continue;
            }
            // 隐式模型：收集连续的非 item 语句
            let start = self.peek().span;
            let mut body = Vec::new();
            loop {
                self.skip_separators();
                if self.is_eof() || self.is_item_header(Kw::Model) || self.is_item_header(Kw::Layer) {
                    break;
                }
                let before = self.i;
                body.push(self.parse_stmt()?);
                if self.i == before {
                    return Err(self.err_here("E005", "无法识别的语句（解析器没有前进）"));
                }
            }
            if body.is_empty() {
                break;
            }
            let span = start.join(body.last().unwrap().span());
            items.push(Item::ModelDef { name: Ident::new(IMPLICIT_MODEL, start), body, span });
        }
        Ok(Ast::Program { items })
    }

    // -- 模型 / 层定义 ------------------------------------------------------

    fn parse_model(&mut self) -> Result<Item> {
        let start = self.peek().span;
        self.bump(); // 模型
        self.skip_fillers();
        let name = self.expect_ident("模型名")?;
        self.skip_fillers();
        let body = self.parse_block_after_header()?;
        let span = start.join(self.prev_span());
        Ok(Item::ModelDef { name, body, span })
    }

    fn parse_layer_def(&mut self) -> Result<Item> {
        let start = self.peek().span;
        self.bump(); // 层/定义
        self.skip_fillers();
        let name = self.expect_ident("层类型名")?;
        // 参数列表：`稀疏块(维, 连)` 或 `稀疏块 用 维 连 为:`
        let mut params: Vec<Ident> = Vec::new();
        if self.peek().is_punct(Punct::LParen) {
            self.bump();
            loop {
                self.skip_separators();
                if self.eat_punct(Punct::RParen) || self.is_eof() {
                    break;
                }
                if self.peek().is_punct(Punct::Comma) {
                    self.bump();
                    continue;
                }
                params.push(self.expect_ident("参数名")?);
            }
        }
        // `用 p1 p2 (为:)?`
        loop {
            self.skip_fillers();
            if self.eat_kw(Kw::As) {
                continue;
            }
            if self.peek().is_punct(Punct::Comma) {
                self.bump();
                continue;
            }
            if self.peek().is_word() {
                params.push(self.expect_ident("参数名")?);
                continue;
            }
            break;
        }
        // 冒号 / `为` 之后是定义体
        self.eat_punct(Punct::Colon);
        let body = self.parse_block_after_header()?;
        let span = start.join(self.prev_span());
        // 定义体的第一条语句就是模板
        let (kind, attrs) = match body.first() {
            Some(Stmt::Instantiate { kind, args, .. }) => {
                let bound = self.bind_args_of(kind, args);
                (kind.clone(), bound)
            }
            Some(other) => {
                return Err(self.error(
                    "E005",
                    format!("层定义 `{}` 的体内第一条语句必须是层实例化，实际是{}", name.name, other.kind()),
                    other.span(),
                )
                .with_note("例：`定义 稀疏块 用 维 连 为:` 换行 `稀疏 维度=维 连接=连`"));
            }
            None => {
                return Err(self.error(
                    "E003",
                    format!("层定义 `{}` 是空的：缺少模板语句", name.name),
                    span,
                )
                .with_hint("体内写一条实例化，例如 `稀疏 维度=维 连接=连`"));
            }
        };
        Ok(Item::LayerDef { name, params, kind, attrs, span })
    }

    fn prev_span(&self) -> Span {
        let k = self.i.saturating_sub(1);
        self.toks.get(k).map(|t| t.span).unwrap_or(Span::ZERO)
    }

    /// 头部（模型名 / 层定义名 + 冒号）之后的块。
    ///
    /// 四套机制 + 容错：
    /// - `:` + 换行 + 缩进 → 缩进块
    /// - `:` + 同行语句 → 内联块（分号分隔）
    /// - `:` 或 `开始`/`{`/`【` → 标记块（`开始`…`结束`、`{}`、`【】`）
    /// - `:` + 换行但**没有缩进** → 宽容模式：文件剩余部分都是块体（第三部分要求
    ///   "缩进错了不报错，只要层级能从冒号和块标记推断出来"）
    fn parse_block_after_header(&mut self) -> Result<Vec<Stmt>> {
        // 冒号（可重复出现，容错）
        while self.peek().is_punct(Punct::Colon) {
            self.bump();
        }
        if self.at_stmt_end() {
            // 头部后有分隔符：可能是内联块（分号/同行）或换行+缩进
            let mut saw_newline = false;
            while matches!(self.peek().kind, TokKind::Newline)
                || self.peek().is_punct(Punct::Semi)
            {
                if matches!(self.peek().kind, TokKind::Newline) {
                    saw_newline = true;
                }
                self.bump();
            }
            if !saw_newline {
                // `模型 小; a; b` —— 同行内联
                return self.parse_inline_block();
            }
        }
        // 标记块
        if self.peek().is_kw(Kw::Start) || self.peek().is_punct(Punct::LBrace) || self.peek().is_punct(Punct::LBracket) {
            return self.parse_marker_block();
        }
        if matches!(self.peek().kind, TokKind::Indent) {
            self.bump();
            return self.parse_indent_block();
        }
        if self.peek().is_kw(Kw::End) || self.peek().is_punct(Punct::RBrace) || self.peek().is_punct(Punct::RBracket) || self.is_eof() {
            return Ok(Vec::new());
        }
        // 同行内联 + 宽容续行。
        //
        // 例（第五部分"标点宽容"）：`模型 小 嵌入 512` 换行 `线性 4054` 换行
        // `连接 嵌入 线性` —— 没有冒号缩进也没有分号，层级只能靠"模型头 + 续行"推断。
        let mut body = Vec::new();
        if !self.at_stmt_end() {
            body = self.parse_inline_block()?;
        }
        let tail = self.parse_rest_block()?;
        body.extend(tail);
        Ok(body)
    }

    /// 缩进块。
    ///
    /// 关键取舍：**缩进只是建议性的**（第三部分："缩进错了不报错，只要层级能从
    /// 冒号和块标记推断出来"）。所以这里吞掉一切多余的 Indent/Dedent/分隔符，
    /// 块体一直延续到：文件结束、块结束标记、或下一个 `模型`/`层` 条目。
    /// 这样 2/4/8/Tab 混用、同级宽度不统一（`  稀疏` 与 `     线性`）都不会出问题。
    fn parse_indent_block(&mut self) -> Result<Vec<Stmt>> {
        let mut body = Vec::new();
        loop {
            loop {
                let skip = matches!(
                    self.peek().kind,
                    TokKind::Newline | TokKind::Indent | TokKind::Dedent
                ) || self.peek().is_punct(Punct::Semi)
                    || self.peek().is_punct(Punct::Dot)
                    || self.peek().is_punct(Punct::Bang);
                if !skip {
                    break;
                }
                self.bump();
            }
            if self.is_eof()
                || self.peek().is_kw(Kw::End)
                || self.peek().is_punct(Punct::RBrace)
                || self.peek().is_punct(Punct::RBracket)
                || self.is_item_header(Kw::Model)
                || self.is_item_header(Kw::Layer)
            {
                break;
            }
            let before = self.i;
            body.push(self.parse_stmt()?);
            if self.i == before {
                return Err(self.err_here("E005", "块内语句没有推进"));
            }
        }
        Ok(body)
    }

    fn parse_marker_block(&mut self) -> Result<Vec<Stmt>> {
        // `{`/`【` 是括号式块（词法层已吃掉换行）；`开始`/`起` 是词式块。
        let brace = self.peek().is_punct(Punct::LBrace) || self.peek().is_punct(Punct::LBracket);
        self.bump();
        let mut body = Vec::new();
        loop {
            // 块内的缩进同样只是建议性的（`模型 裸 开始` 后写缩进或不缩进都行）
            loop {
                let skip = matches!(
                    self.peek().kind,
                    TokKind::Newline | TokKind::Indent | TokKind::Dedent
                ) || self.peek().is_punct(Punct::Semi)
                    || self.peek().is_punct(Punct::Dot)
                    || self.peek().is_punct(Punct::Bang);
                if !skip {
                    break;
                }
                self.bump();
            }
            if self.is_eof() {
                break;
            }
            if brace {
                if self.peek().is_punct(Punct::RBrace) || self.peek().is_punct(Punct::RBracket) {
                    self.bump();
                    break;
                }
            } else if self.peek().is_kw(Kw::End) {
                self.bump();
                break;
            } else if self.line_has_block_end() {
                // `模型 小 结束`：本行就是块结束标记，整行吞掉
                while !self.at_stmt_end() && !self.is_eof() {
                    self.bump();
                }
                break;
            }
            let before = self.i;
            body.push(self.parse_stmt()?);
            if self.i == before {
                return Err(self.err_here("E005", "块内语句没有推进"));
            }
        }
        Ok(body)
    }

    /// 本逻辑行里是否出现块结束标记（词式块专用；括号式块不能用，
    /// 因为括号内换行被词法层吃掉了，会误判整块）。
    fn line_has_block_end(&self) -> bool {
        let mut k = 0;
        loop {
            let t = self.at(k);
            if matches!(t.kind, TokKind::Eof | TokKind::Newline | TokKind::Dedent) {
                return false;
            }
            if t.is_kw(Kw::End) || t.is_punct(Punct::RBrace) || t.is_punct(Punct::RBracket) {
                return true;
            }
            k += 1;
            if k > 256 {
                return false;
            }
        }
    }

    /// 吃掉连接类词：`连`/`到`/`再`/`和`/`,`/`->`，以及 `连到` 这种中文复合词。
    fn eat_connectors(&mut self) {
        loop {
            let t = self.peek().clone();
            if t.is_punct(Punct::Arrow) || t.is_punct(Punct::Comma) {
                self.bump();
                continue;
            }
            if let Some(w) = t.word() {
                let set = t.kw_set();
                let connector = set.contains(Kw::Connect)
                    || set.contains(Kw::To)
                    || set.contains(Kw::Then)
                    || set.contains(Kw::And);
                if connector {
                    self.bump();
                    continue;
                }
                if set.is_empty() && keywords::split_connectors(w).is_some() {
                    self.bump();
                    continue;
                }
            }
            break;
        }
    }

    /// 同行内联块：`模型 小：嵌入 4054 1536；稀疏 1536 8；…`
    ///
    /// 关键：`;`/`，` 在内联块里是**语句分隔符**，不能像换行那样结束整块
    /// （早期版本一遇 `;` 就 break，导致示例 D 的后半段掉进隐式模型 `main`）。
    fn parse_inline_block(&mut self) -> Result<Vec<Stmt>> {
        let mut body = Vec::new();
        loop {
            while self.peek().is_punct(Punct::Semi)
                || self.peek().is_punct(Punct::Comma)
                || self.peek().is_punct(Punct::Dot)
                || self.peek().is_punct(Punct::Bang)
            {
                self.bump();
            }
            if matches!(
                self.peek().kind,
                TokKind::Newline | TokKind::Eof | TokKind::Dedent
            ) || self.peek().is_punct(Punct::RBrace)
                || self.peek().is_punct(Punct::RBracket)
            {
                break;
            }
            let before = self.i;
            body.push(self.parse_stmt()?);
            if self.i == before {
                return Err(self.err_here("E005", "同行语句没有推进"));
            }
        }
        Ok(body)
    }

    /// 宽容模式：从当前位置吃到文件末尾（或块结束标记）。
    fn parse_rest_block(&mut self) -> Result<Vec<Stmt>> {
        let mut body = Vec::new();
        loop {
            self.skip_separators();
            if matches!(self.peek().kind, TokKind::Indent | TokKind::Dedent) {
                self.bump();
                continue;
            }
            if self.is_eof() || self.peek().is_kw(Kw::End) || self.peek().is_punct(Punct::RBrace) || self.peek().is_punct(Punct::RBracket) {
                break;
            }
            if self.is_item_header(Kw::Model) || self.is_item_header(Kw::Layer) {
                break;
            }
            let before = self.i;
            body.push(self.parse_stmt()?);
            if self.i == before {
                return Err(self.err_here("E005", "块内语句没有推进"));
            }
        }
        Ok(body)
    }

    // -- 语句分发 ----------------------------------------------------------

    fn parse_stmt(&mut self) -> Result<Stmt> {
        self.skip_separators();
        self.skip_fillers();
        let t = self.peek().clone();
        // 1. `@op ...` 裸指令；但 `@名字 = 类型` 是第四部分的"转义关键字当标识符"
        if t.is_punct(Punct::At) {
            if self.at(2).is_punct(Punct::Eq) || self.at(2).is_punct(Punct::Colon) {
                self.bump(); // 吃掉转义符 `@`，剩下的按实例化解析
                self.escaped_ident = true;
            } else {
                return self.parse_raw_op();
            }
        }
        // 2. 训练相关的词（V0.1.6）：
        //    * `损失 均方` / `反向` 现在是**真语法**（V0.1.5 里它们报 E111）；
        //    * `训练` / `优化` 一类仍是可判定的边界（E111）。
        //    必须放在"自定义算子/前向/连接"之前：否则 `损失 均方` 会被当成裸标识符列表
        //    去当层引用，报出 E006「引用了未定义的层」—— 那是在暗示"你打错了"。
        //    排除 `X = ...` / `X: ...`（那是拿这些词当实例名）。
        if let Some(w) = t.word() {
            if !self.at(1).is_punct(Punct::Eq) && !self.at(1).is_punct(Punct::Colon) {
                match crate::keywords::training_word_kind(w) {
                    Some(crate::keywords::TrainingWord::Loss) => return self.parse_loss(),
                    Some(crate::keywords::TrainingWord::Backward) => return self.parse_backward(),
                    Some(crate::keywords::TrainingWord::Train) => {
                        return Err(self
                            .err_here(
                                "E111",
                                &format!("`{w}` 是训练相关的词，但本语言还没有训练循环/参数更新陈述"),
                            )
                            .with_hint(
                                "V0.1.6 起支持「损失 → 反向」：把 `损失 均方` 与 `反向` 写进模型，\
                                 前端就会按拓扑反序发射 `MSE_BWD` / `GEMM_BWD_W` / `GEMM_BWD_X` / \
                                 `BIAS_BWD` / `ACT_BWD`，并打印每个参数的梯度。\
                                 但**参数更新**（`OP_ADAM`/`OP_ADAMW`/`OP_SGD`）还不是语言的一部分：\
                                 优化器状态、学习率、步数、数据从哪来，都是训练循环的语义，\
                                 本版没有设计（边界见 REPORT_V015.md §11.5）。",
                            ));
                    }
                    None => {}
                }
            }
        }
        // 2. 自定义算子 `算子 名 = opcode(N) args(M)`
        if t.is_kw(Kw::Op) {
            return self.parse_custom_op();
        }
        // 3. 前向 `前向 a b` / `从 a 前向到 b`
        if t.is_kw(Kw::From) {
            return self.parse_forward();
        }
        if t.is_kw(Kw::Forward) {
            return self.parse_forward();
        }
        // 4. 连接 `连接 a b c` / `connect(a, b, c)`
        if t.is_kw(Kw::Connect) {
            return self.parse_connect();
        }
        // 5. `把 e 连到 t 再连到 o`
        if t.is_kw(Kw::As) {
            return self.parse_connect();
        }
        // 6. 裸标识符列表（`a;b;c`、`a,b,c`、`a -> b -> c`）
        if self.lookahead_is_ident_list() {
            return self.parse_connect();
        }
        // 7. 真·关键字开头但是其他结构词 → 让报错说清楚
        //
        // 例外：`m = 嵌入 512` 这种"关键字被当实例名"要落到实例化分支，
        // 由 expect_ident 报 E007（关键字不能作标识符），比"模型不能出现在语句位置"准。
        if t.is_kw(Kw::Model) && !(self.at(1).is_punct(Punct::Eq) || (self.at(1).is_word() && !self.at(2).is_punct(Punct::Colon))) {
            return Err(self.err_here("E005", "`模型` 不能出现在语句位置")
                .with_hint("模型定义要写在最外层，不能嵌套"));
        }
        if t.is_kw(Kw::Layer)
            && !(self.at(1).is_punct(Punct::Eq)
                || self.at(1).is_punct(Punct::Colon)
                || self.at(1).is_punct(Punct::LParen))
        {
            return Err(self.err_here("E005", "层定义（`定义`/`层`）只能写在最外层"));
        }
        if t.is_kw(Kw::Repeat) {
            return Err(self
                .err_here("E005", "`重复` 前面缺少被重复的层")
                .with_hint("先写层实例化，再跟 `重复24次`"));
        }
        if t.is_kw(Kw::Start) || t.is_kw(Kw::End) {
            return Err(self.err_here("E005", format!("孤立的块标记 {}", t.describe())));
        }
        // 8. 实例化 + 可选的重复后缀
        self.parse_instantiate_or_connect()
    }

    /// 是否是"顶层条目头"（`模型 名` / `层 名` / `定义 名`），而不是
    /// 恰好撞上关键字的标识符。
    ///
    /// 边界：第九部分把 `M` 列为 `model` 别名，`L` 列为 `layer` 别名，于是
    /// `m = 嵌入 512` 里的 `m` 会被判成 model 关键字。这里用"后面跟 `=`/`(`/`:`"
    /// 把实例化与条目头区分开：块循环不会因为 `m = …` 提前结束（否则它会退化成
    /// 一个新的模型头，报出莫名其妙的"需要模型名"）。
    fn is_item_header(&self, kw: Kw) -> bool {
        if !self.peek().is_kw(kw) {
            return false;
        }
        !(self.at(1).is_punct(Punct::Eq)
            || self.at(1).is_punct(Punct::LParen)
            || self.at(1).is_punct(Punct::Colon))
    }

    /// 整行前瞻：是否"只有 ≥2 个词 + 分隔/连接词"。
    fn lookahead_is_ident_list(&self) -> bool {
        let mut words = 0usize;
        let mut k = 0usize;
        let stop = arg_stop_kw();
        loop {
            let t = self.at(k);
            match &t.kind {
                TokKind::Eof | TokKind::Newline | TokKind::Dedent => break,
                TokKind::Punct(p) => match p {
                    Punct::RBrace | Punct::RBracket => break,
                    Punct::Comma | Punct::Semi | Punct::Dot | Punct::Bang | Punct::Arrow => {}
                    _ => return false,
                },
                TokKind::Word(w) => {
                    let set = t.kw_set();
                    let connector = set.contains(Kw::Connect) || set.contains(Kw::To) || set.contains(Kw::Then) || set.contains(Kw::And);
                    if connector {
                        // 连接词可以夹在中间
                    } else if set.is_empty() && keywords::split_connectors(w).is_some() {
                        // `e 连到 t`：中文复合连接词也算连接词，不算标识符
                    } else if set.is_empty() {
                        words += 1;
                    } else if set.contains(Kw::Filler) {
                        // 虚词忽略
                    } else if set.intersects(stop) {
                        // 结构化词（`层`/`到`除外）出现即说明不是裸列表
                        return false;
                    } else {
                        return false;
                    }
                }
                TokKind::Str(_) => words += 1,
                _ => return false,
            }
            k += 1;
            if k > 512 {
                return false;
            }
        }
        words >= 2
    }

    // -- 实例化 ------------------------------------------------------------

    fn parse_instantiate_or_connect(&mut self) -> Result<Stmt> {
        self.skip_fillers();
        let t0 = self.peek().clone();
        let t1 = self.at(1).clone();

        // Is `t0` a kind word?
        let t0_word = t0.word().map(|s| s.to_string());
        let t0_is_kind = t0_word.as_deref().map(keywords::kind_is_known).unwrap_or(false);

        // 三种书写顺序（第三部分/Level 3/示例 E/G 都要支持）：
        //   `name = kind args`、`name kind args`（示例 G：`e 嵌入 1536`）
        //   `kind name args` （Level 3：`嵌 e 512`）
        //   `kind args`      （`嵌入 512`，隐式实例名 = 层类型）
        let order = if t1.is_punct(Punct::Eq) || (t1.is_punct(Punct::Colon) && self.at(2).is_word()) {
            Order::NameEqKind
        } else if t0_is_kind
            && t1.is_word()
            && !t1.word().map(keywords::kind_is_known).unwrap_or(false)
            && !self.at(2).is_punct(Punct::Eq)
            && !self.at(2).is_punct(Punct::Colon)
        {
            Order::KindName
        } else if !t0_is_kind
            && t1.is_word()
            && (t1.word().map(keywords::kind_is_known).unwrap_or(false)
                // 用户 `定义` 的层类型在 parse 期还认不出来，但 `名字 类型(...)` 的括号能定性
                || self.at(2).is_punct(Punct::LParen))
        {
            Order::NameKind
        } else {
            Order::KindOnly
        };

        let consumed = match order {
            Order::NameEqKind => 3,
            Order::NameKind => 2,
            Order::KindName => 2,
            Order::KindOnly => 1,
        };
        let (name, kind) = match order {
            Order::NameEqKind | Order::NameKind => {
                let name = self.expect_ident("实例名")?;
                if self.peek().is_punct(Punct::Eq) || self.peek().is_punct(Punct::Colon) {
                    self.bump();
                }
                self.skip_fillers();
                let kind_tok = self.peek().clone();
                if kind_tok.word().is_none() {
                    return Err(Diag::error("E005", "层类型必须是标识符", kind_tok.span));
                }
                let kind = self.expect_ident("层类型")?;
                (name, kind)
            }
            Order::KindName => {
                let kind = self.expect_ident("层类型")?;
                self.skip_fillers();
                let name = self.expect_ident("实例名")?;
                (name, kind)
            }
            Order::KindOnly => {
                // `嵌入 512` / `稀疏块(1536,8)` / `我的激活`
                let kind = self.expect_ident("层类型")?;
                let name = kind.clone();
                (name, kind)
            }
        };

        let start = self.toks[self.i.saturating_sub(consumed)].span;
        self.skip_fillers();

        // 参数：括号形式或位置/命名混合
        let args: Vec<Expr>;
        if self.peek().is_punct(Punct::LParen) {
            self.bump();
            args = self.parse_arg_list(Punct::RParen)?;
            if !self.eat_punct(Punct::RParen) && !self.is_eof() {
                return Err(self.err_here("E005", "参数列表没有闭合的 `)`"));
            }
        } else {
            args = self.parse_arg_list_bare()?;
        }

        let mut span = start.join(self.prev_span());
        let stmt = Stmt::Instantiate { name, kind, args, span };

        // 重复后缀
        if let Some(rep) = self.parse_repeat_suffix()? {
            let count = rep;
            span = span.join(self.prev_span());
            return Ok(Stmt::Repeat { body: Box::new(stmt), count, span });
        }
        let _ = span;
        Ok(stmt)
    }

    /// 括号内参数列表：位置 + 命名混用，逗号/空白分隔。
    fn parse_arg_list(&mut self, close: Punct) -> Result<Vec<Expr>> {
        let mut args = Vec::new();
        loop {
            self.skip_separators();
            if self.peek().is_punct(close) || self.is_eof() {
                break;
            }
            if self.peek().is_punct(Punct::Comma) {
                self.bump();
                continue;
            }
            let before = self.i;
            args.push(self.parse_arg()?);
            if self.i == before {
                break;
            }
            if self.peek().is_punct(Punct::Comma) {
                self.bump();
            }
        }
        Ok(args)
    }

    /// 括号外的参数列表：吃到语句结束 / 下一条语句 / 重复后缀。
    fn parse_arg_list_bare(&mut self) -> Result<Vec<Expr>> {
        let mut args = Vec::new();
        let stop = arg_stop_kw();
        loop {
            if self.at_stmt_end() || self.is_eof() {
                break;
            }
            if self.peek().is_punct(Punct::Comma) {
                self.bump();
                continue;
            }
            if self.peek().is_punct(Punct::RParen) {
                break;
            }
            // 重复后缀
            if self.peek().is_kw(Kw::Repeat) {
                break;
            }
            // 已知层类型词 → 下一条语句
            if let Some(w) = self.peek().word() {
                if !self.peek().is_punct(Punct::Eq) && keywords::kind_is_known(w) && !args.is_empty() {
                    break;
                }
            }
            // 结构化关键字 → 下一条语句（`#key=value` 除外）
            let set = self.peek().kw_set();
            if !set.is_empty() && !self.peek().is_kw(Kw::Filler) {
                if self.at(1).is_punct(Punct::Eq) {
                    // `连接=8` 这种把关键字当参数键
                } else if set.intersects(stop) {
                    break;
                }
            }
            let before = self.i;
            args.push(self.parse_arg()?);
            if self.i == before {
                break;
            }
        }
        Ok(args)
    }

    /// 单个参数：位置值 或 `键=值`。
    fn parse_arg(&mut self) -> Result<Expr> {
        let t = self.peek().clone();
        // 命名参数
        if t.is_word() && self.at(1).is_punct(Punct::Eq) {
            let key = Ident::new(t.word().unwrap().to_string(), t.span);
            self.bump(); // key
            self.bump(); // =
            let value = self.parse_arg_value()?;
            return Ok(Expr::Named { key, value: Box::new(value) });
        }
        // `键: 值`（容错；仅当冒号后还有内容且不是块头）
        if t.is_word() && self.at(1).is_punct(Punct::Colon) && self.at(2).is_word() {
            let key = Ident::new(t.word().unwrap().to_string(), t.span);
            self.bump();
            self.bump();
            let value = self.parse_arg_value()?;
            return Ok(Expr::Named { key, value: Box::new(value) });
        }
        self.parse_arg_value()
    }

    fn parse_arg_value(&mut self) -> Result<Expr> {
        let t = self.peek().clone();
        match &t.kind {
            TokKind::Int(n) => {
                self.bump();
                Ok(Expr::Int(*n))
            }
            TokKind::Float(x) => {
                self.bump();
                Ok(Expr::Float(*x))
            }
            TokKind::Str(s) => {
                self.bump();
                Ok(Expr::Ident(s.clone()))
            }
            TokKind::Word(w) => {
                // 中文数字（`二十四` → 24）
                if w.chars().all(is_cn_digit) {
                    if let Some(n) = cn_number_value(w) {
                        self.bump();
                        return Ok(Expr::Int(n));
                    }
                }
                self.bump();
                Ok(Expr::Ident(w.clone()))
            }
            TokKind::Punct(Punct::LParen) => {
                self.bump();
                let items = self.parse_arg_list(Punct::RParen)?;
                self.eat_punct(Punct::RParen);
                if items.len() == 1 {
                    Ok(items.into_iter().next().unwrap())
                } else {
                    Ok(Expr::Arrow(items))
                }
            }
            TokKind::Punct(Punct::Minus) => {
                self.bump();
                let v = self.parse_arg_value()?;
                match v {
                    Expr::Int(n) => Ok(Expr::Int(-n)),
                    Expr::Float(x) => Ok(Expr::Float(-x)),
                    other => Ok(other),
                }
            }
            _ => Err(Diag::error(
                "E005",
                format!("这里需要参数值，实际是{}", t.describe()),
                t.span,
            )),
        }
    }

    /// 重复后缀：`重复24次` / `x 24` / `* 24` / `24层` / `重复 二十四 次`。
    fn parse_repeat_suffix(&mut self) -> Result<Option<Expr>> {
        // `*` 是标点而不是词，所以走单独分支（`×` 在词法层已归一成 Star）
        if self.peek().is_punct(Punct::Star) {
            self.bump();
            self.skip_fillers();
            let count = self.parse_arg_value()?;
            self.skip_fillers();
            if self.peek().is_kw(Kw::Layer) {
                self.bump();
            }
            return Ok(Some(count));
        }
        if self.peek().is_kw(Kw::Repeat) {
            self.bump();
            self.skip_fillers();
            let count = self.parse_arg_value()?;
            self.skip_fillers();
            // `24次` 里的 `次` 已由 Filler 吃掉；`层` 也吞掉
            if self.peek().is_kw(Kw::Layer) {
                self.bump();
            }
            return Ok(Some(count));
        }
        // `24层`
        if let TokKind::Int(n) = self.peek().kind {
            if self.at(1).is_kw(Kw::Layer) || self.at(1).is_kw(Kw::Repeat) {
                self.bump();
                if self.peek().is_kw(Kw::Layer) {
                    self.bump();
                }
                return Ok(Some(Expr::Int(n)));
            }
        }
        Ok(None)
    }

    // -- 连接 --------------------------------------------------------------

    fn parse_connect(&mut self) -> Result<Stmt> {
        let start = self.peek().span;
        // 可选的引导词
        if self.peek().is_kw(Kw::Connect) || self.peek().is_kw(Kw::As) {
            // `connect(...)`
            if self.at(1).is_punct(Punct::LParen) {
                self.bump();
                self.bump();
                let mut path = Vec::new();
                loop {
                    self.skip_separators();
                    if self.peek().is_punct(Punct::RParen) || self.is_eof() {
                        break;
                    }
                    if self.peek().is_punct(Punct::Comma) {
                        self.bump();
                        continue;
                    }
                    path.push(self.expect_ident("层名")?);
                }
                self.eat_punct(Punct::RParen);
                return self.finish_connect(path, start, None);
            }
            self.bump();
        }
        // 连接路径的元素分隔符：`,`/`、`/`；`/`。`/`!`/`->`/`连`/`到`/`再`/`和`。
        //
        // 注意 `;` 在这里是**列表分隔符**而不是语句分隔符：第九部分把 `;`/`；`
        // 同时列在 `and` 与 `semicolon` 两类里，验收 8 要求
        // `嵌入层;主干;输出层` 解析成 3 个标识符的列表。
        let mut path: Vec<Ident> = Vec::new();
        let mut blocked: Option<String> = None;
        loop {
            if self.is_eof() || matches!(self.peek().kind, TokKind::Newline | TokKind::Dedent) {
                break;
            }
            let t = self.peek().clone();
            match &t.kind {
                TokKind::Punct(p)
                    if matches!(
                        p,
                        Punct::Comma | Punct::Semi | Punct::Dot | Punct::Bang | Punct::Arrow
                    ) =>
                {
                    self.bump();
                    continue;
                }
                TokKind::Punct(Punct::RBrace) | TokKind::Punct(Punct::RBracket) => break,
                TokKind::Word(w) => {
                    let set = t.kw_set();
                    let connector = set.contains(Kw::Connect)
                        || set.contains(Kw::To)
                        || set.contains(Kw::Then)
                        || set.contains(Kw::And);
                    if connector {
                        self.bump();
                        continue;
                    }
                    if set.contains(Kw::Filler) {
                        self.bump();
                        continue;
                    }
                    if set.is_empty() && keywords::split_connectors(w).is_some() {
                        self.bump();
                        continue;
                    }
                    if set.contains(Kw::Model)
                        || set.contains(Kw::Layer)
                        || set.contains(Kw::Forward)
                        || set.contains(Kw::Repeat)
                        || set.contains(Kw::Start)
                        || set.contains(Kw::End)
                        || set.contains(Kw::Op)
                    {
                        blocked = Some(w.clone());
                        break;
                    }
                    // `c = 嵌入 512` 之类：下一条语句开始了
                    if self.at(1).is_punct(Punct::Eq) || self.at(1).is_punct(Punct::Colon) {
                        break;
                    }
                    path.push(Ident::new(w.clone(), t.span));
                    self.bump();
                }
                // `@名字` 转义引用（第四部分）
                TokKind::Punct(Punct::At) if self.at(1).is_word() => {
                    self.bump();
                    let t2 = self.peek().clone();
                    path.push(Ident::new(t2.word().unwrap().to_string(), t2.span));
                    self.bump();
                }
                TokKind::Str(w) => {
                    path.push(Ident::new(w.clone(), t.span));
                    self.bump();
                }
                TokKind::Int(_) | TokKind::Float(_) => {
                    return Err(Diag::error(
                        "E005",
                        format!("连接路径里出现了{}，这里只能是层名", t.describe()),
                        t.span,
                    )
                    .with_hint("连接写成 `连接 a b c` 或 `a -> b -> c`"));
                }
                _ => break,
            }
        }
        self.finish_connect(path, start, blocked)
    }

    fn finish_connect(
        &mut self,
        path: Vec<Ident>,
        start: Span,
        blocked: Option<String>,
    ) -> Result<Stmt> {
        if path.len() < 2 {
            let span = start.join(self.prev_span());
            let got = if path.is_empty() {
                "空".to_string()
            } else {
                format!("\"{}\"", path[0].name)
            };
            let mut d = Diag::error("E005", format!("连接至少需要 2 个层，实际是 {got}"), span)
                .with_hint("例：`连接 嵌入 稀疏 线性` 或 `a -> b -> c`");
            if let Some(w) = blocked {
                d = d.with_note(format!(
                    "`{w}` 是关键字；要把它当层名引用请转义：`@{w}` 或 `\"{w}\"`"
                ));
            }
            return Err(d);
        }
        let span = start.join(self.prev_span());
        Ok(Stmt::Connect { path, span })
    }

    // -- 前向 --------------------------------------------------------------

    /// `损失 均方` / `损失 mse`（V0.1.6）。
    ///
    /// 只解析**语法**：损失函数名是否认识由 sema 判定（与层类型同一套做法 ——
    /// AST 保留原文，规范化与校验只发生一次，见 `keywords::loss_kind_canonical`）。
    fn parse_loss(&mut self) -> Result<Stmt> {
        let start = self.peek().span;
        self.bump(); // 吃掉 `损失` / `loss` …
        self.skip_fillers();
        let Some(w) = self.peek().word() else {
            return Err(self.err_here(
                "E112",
                "`损失` 后面要跟损失函数名",
            ).with_hint("现在只有均方误差：写 `损失 均方`（或 `损失 mse`）"));
        };
        let kind = Ident { name: w.to_string(), span: self.peek().span };
        self.bump();
        let span = start.join(self.prev_span());
        Ok(Stmt::Loss { kind, span })
    }

    /// `反向`（V0.1.6）：只吃这一个词，不需要操作数。
    fn parse_backward(&mut self) -> Result<Stmt> {
        let span = self.peek().span;
        self.bump();
        Ok(Stmt::Backward { span })
    }

    fn parse_forward(&mut self) -> Result<Stmt> {
        let start = self.peek().span;
        if self.peek().is_kw(Kw::Forward) && self.at(1).is_punct(Punct::LParen) {
            self.bump();
            self.bump();
            let a = self.expect_ident("输入层")?;
            self.skip_separators();
            self.eat_punct(Punct::Comma);
            let b = self.expect_ident("输出层")?;
            self.skip_separators();
            self.eat_punct(Punct::RParen);
            let span = start.join(self.prev_span());
            return Ok(Stmt::Forward { input: a, output: b, span });
        }
        let mut input: Option<Ident> = None;
        if self.peek().is_kw(Kw::From) {
            self.bump();
            self.skip_fillers();
            input = Some(self.expect_ident("输入层")?);
        }
        let saw_forward = if self.peek().is_kw(Kw::Forward) {
            self.bump();
            true
        } else if let Some(w) = self.peek().word() {
            // `前向到` 这种把前向词和连接词粘在一起的写法
            if self.peek().kw_set().is_empty() && keywords::split_forward(w).is_some() {
                self.bump();
                true
            } else {
                false
            }
        } else {
            false
        };
        if !saw_forward && input.is_none() {
            return Err(self.err_here("E005", "前向语句缺少 `前向`"));
        }
        self.eat_connectors();
        self.skip_fillers();
        let first = self.expect_ident("输出层")?;
        let (input, output) = match input {
            Some(i) => (i, first),
            None => {
                self.skip_fillers();
                // `前向 嵌入 线性`
                self.eat_connectors();
                let second = self.expect_ident("输出层")?;
                (first, second)
            }
        };
        let span = start.join(self.prev_span());
        Ok(Stmt::Forward { input, output, span })
    }

    // -- 自定义算子 --------------------------------------------------------

    fn parse_custom_op(&mut self) -> Result<Stmt> {
        let start = self.peek().span;
        self.bump(); // 算子
        self.skip_fillers();
        let name = self.expect_ident("算子名")?;
        // 引导符 `=` / `为` / `用` / `:`
        loop {
            if self.peek().is_punct(Punct::Eq) || self.peek().is_punct(Punct::Colon) || self.peek().is_kw(Kw::As) {
                self.bump();
                continue;
            }
            break;
        }
        self.skip_fillers();
        // opcode(200) / opcode 200 / 操作码 200
        let opcode = if self.peek().is_kw(Kw::Opcode) || self.peek().is_word() {
            if self.peek().is_word() && !self.peek().is_kw(Kw::Opcode) {
                // 宽容：`算子 f 200`
                self.parse_arg_value()?
            } else {
                self.bump();
                self.skip_fillers();
                if self.peek().is_punct(Punct::LParen) {
                    self.bump();
                    let v = self.parse_arg_value()?;
                    self.skip_separators();
                    self.eat_punct(Punct::RParen);
                    v
                } else {
                    self.parse_arg_value()?
                }
            }
        } else {
            return Err(self.err_here("E003", "自定义算子缺少 opcode")
                .with_hint("写成 `算子 我的激活 = opcode(200) args(2)`"));
        };
        self.skip_fillers();
        // args(2) / args(a, b) / 参数 a b
        let mut arg_names: Vec<Ident> = Vec::new();
        if self.peek().is_kw(Kw::Args) || (self.peek().is_word() && keywords::eq_norm(self.peek().word().unwrap(), "args")) {
            self.bump();
            self.skip_fillers();
            if self.peek().is_punct(Punct::LParen) {
                self.bump();
                loop {
                    self.skip_separators();
                    if self.peek().is_punct(Punct::RParen) || self.is_eof() {
                        break;
                    }
                    if self.peek().is_punct(Punct::Comma) {
                        self.bump();
                        continue;
                    }
                    let t = self.peek().clone();
                    match &t.kind {
                        TokKind::Int(n) => {
                            self.bump();
                            // `args(2)` = 声明 2 个实参 → arg0, arg1
                            for k in 0..*n {
                                arg_names.push(Ident::new(format!("arg{k}"), t.span));
                            }
                        }
                        _ => arg_names.push(self.expect_ident("参数名")?),
                    }
                }
                self.eat_punct(Punct::RParen);
            } else {
                // 非括号形式：只在同一行内收参数名（**不能**跳过换行，
                // 否则下一行的语句会被吃进形参列表）
                while self.peek().is_word() && !self.at_stmt_end() {
                    arg_names.push(self.expect_ident("参数名")?);
                    if self.peek().is_punct(Punct::Comma) {
                        self.bump();
                    }
                }
            }
        }
        let span = start.join(self.prev_span());
        Ok(Stmt::CustomOpDef { name, opcode, args: arg_names, span })
    }

    // -- 裸指令 ------------------------------------------------------------

    fn parse_raw_op(&mut self) -> Result<Stmt> {
        let start = self.peek().span;
        self.bump(); // @
        let opcode = self.expect_ident("指令名")?;
        let mut operands: Vec<Expr> = Vec::new();
        loop {
            if self.at_stmt_end() || self.is_eof() {
                break;
            }
            // 操作数分隔符：逗号、`->`、以及括号（`@gemm(r0, x, r0)` 等价于
            // `@gemm r0 x r0`；裸指令里括号不参与语义，只当分组括号用）
            if self.peek().is_punct(Punct::Arrow)
                || self.peek().is_punct(Punct::Comma)
                || self.peek().is_punct(Punct::LParen)
                || self.peek().is_punct(Punct::RParen)
            {
                self.bump();
                continue;
            }
            // 按"有无空白间隙"切操作数：`cpool+0` 是一片
            let first = self.peek().clone();
            let mut last = first.span;
            self.bump();
            while !self.at_stmt_end()
                && !self.is_eof()
                && !self.peek().is_punct(Punct::Arrow)
                && !self.peek().is_punct(Punct::Comma)
                && !self.peek().is_punct(Punct::LParen)
                && !self.peek().is_punct(Punct::RParen)
                && self.peek().span.start == last.end
            {
                last = self.peek().span;
                self.bump();
            }
            let text = self
                .src
                .slice(Span::new(first.span.start as usize, last.end as usize))
                .to_string();
            let expr = match &first.kind {
                TokKind::Int(n) if first.span == last => Expr::Int(*n),
                TokKind::Float(x) if first.span == last => Expr::Float(*x),
                _ => Expr::Ident(text),
            };
            operands.push(expr);
        }
        let span = start.join(self.prev_span());
        Ok(Stmt::RawOp { opcode, operands, span })
    }

    // -- 参数绑定（层定义模板 / 实例化）------------------------------------

    /// 把 `args` 绑定成规范 `(键, 值)` 列表。
    fn bind_args_of(&self, kind: &Ident, args: &[Expr]) -> Attrs {
        let positional_hint = if keywords::kind_is_known(&kind.name) {
            Some(keywords::bind_kind_positional(&kind.name, &[]))
        } else {
            None
        };
        let _ = positional_hint;
        let mut out: Attrs = Vec::new();
        let mut positional_index = 0usize;
        let schema: Vec<String> = if keywords::kind_is_known(&kind.name) {
            keywords::kind_param_keys(&kind.name).iter().map(|s| s.to_string()).collect()
        } else {
            Vec::new()
        };
        for a in args {
            match a {
                Expr::Named { key, value } => {
                    out.push((key.clone(), (**value).clone()));
                }
                other => {
                    let key = if schema.len() == 1 {
                        schema[0].clone()
                    } else if let Some(k) = schema.get(positional_index) {
                        k.clone()
                    } else {
                        format!("arg{positional_index}")
                    };
                    positional_index += 1;
                    out.push((Ident::new(key, kind.span), other.clone()));
                }
            }
        }
        out
    }
}

impl KwSet {
    /// 与另一集合是否有交集。
    pub fn intersects(self, other: KwSet) -> bool {
        self.0 & other.0 != 0
    }
}

/// `KeywordTable` 便捷入口（报告用）。
pub fn keyword_table() -> &'static keywords::KeywordTable {
    &KEYWORDS
}

/// 已知层类型名（报告用）。
pub fn known_kinds() -> Vec<&'static str> {
    KINDS.iter().map(|k| k.display).collect()
}
