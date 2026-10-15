//! Sparrow V0.1 —— Lexer（对应任务书第二、三、四、九部分）
//!
//! 设计要点（与任务书第十部分"Lexer 归一化、Parser 只认 canonical"等价，但更稳）：
//!
//! * **标点归一化在这里**：全角 `：；，、（）【】` 与半角 `:;,()[]` 落到同一个
//!   [`Punct`]；`->`/`→` 都落到 [`Punct::Arrow`]。
//! * **关键字归一化在词表查询**（`KEYWORDS.classes`），Token 只存裸词原文。
//!   原因是关键字判定**要上下文**：`x` 既是 repeat 关键字又是合法标识符，
//!   `到` 同属 connect/to，`,` 同属 and/comma。词的类集合在这里算好，Parser 挑。
//! * **缩进宽容**：宽度比较制（tab=4），同级对齐即可，宽度不统一不报错。
//!   括号内不产生 `Indent`/`Dedent`/`Newline`（`Sparse(1536,\n 8)` 合法）。
//! * Token 的 `span` 是**源码原始字节偏移**，因此 `@gemm r0 x r0 -> h`
//!   这类裸指令可以按"有没有空白间隙"切操作数，`cpool+0` 保持一片。
//!
//! 边界判断（第九部分把 `。` 列在 `end`，第四部分说中文标点"绝不"是标识符、
//! "只作为分隔符"）：取第四部分。`。` 是**语句分隔符**，不是块结束标记。

use crate::ast::Span;
use crate::diag::{Diag, Result, SourceFile};
use crate::keywords::KwSet;

/// 归一化后的标点。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Punct {
    At,
    Eq,
    Arrow,
    Colon,
    Semi,
    Comma,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Star,
    Plus,
    Minus,
    Slash,
    Dot,
    Bang,
    Lt,
    Gt,
    Percent,
    Amp,
    Pipe,
    Other,
}

impl Punct {
    pub fn str(self) -> &'static str {
        match self {
            Punct::At => "@",
            Punct::Eq => "=",
            Punct::Arrow => "->",
            Punct::Colon => ":",
            Punct::Semi => ";",
            Punct::Comma => ",",
            Punct::LParen => "(",
            Punct::RParen => ")",
            Punct::LBrace => "{",
            Punct::RBrace => "}",
            Punct::LBracket => "[",
            Punct::RBracket => "]",
            Punct::Star => "*",
            Punct::Plus => "+",
            Punct::Minus => "-",
            Punct::Slash => "/",
            Punct::Dot => ".",
            Punct::Bang => "!",
            Punct::Lt => "<",
            Punct::Gt => ">",
            Punct::Percent => "%",
            Punct::Amp => "&",
            Punct::Pipe => "|",
            Punct::Other => "?",
        }
    }

    /// 语句分隔符。
    pub fn is_separator(self) -> bool {
        matches!(self, Punct::Semi | Punct::Dot | Punct::Bang)
    }

    /// 括号配对。
    pub fn closer(self) -> Option<Punct> {
        match self {
            Punct::LParen => Some(Punct::RParen),
            Punct::LBrace => Some(Punct::RBrace),
            Punct::LBracket => Some(Punct::RBracket),
            _ => None,
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum TokKind {
    /// 裸词：可能是标识符，也可能是关键字（用 [`Token::kw_set`] 判定）。
    Word(String),
    Int(i64),
    Float(f64),
    /// `"..."` 引号转义的标识符/关键字。
    Str(String),
    Punct(Punct),
    Newline,
    Indent,
    Dedent,
    Eof,
}

#[derive(Clone, Debug)]
pub struct Token {
    pub kind: TokKind,
    pub span: Span,
}

impl Token {
    pub fn is_word(&self) -> bool {
        matches!(self.kind, TokKind::Word(_) | TokKind::Str(_))
    }

    pub fn word(&self) -> Option<&str> {
        match &self.kind {
            TokKind::Word(w) | TokKind::Str(w) => Some(w),
            _ => None,
        }
    }

    /// 该词归属的关键字类（引号转义的词按第四部分"不可是关键字"处理，返回空集）。
    pub fn kw_set(&self) -> KwSet {
        match &self.kind {
            TokKind::Word(w) => crate::keywords::KEYWORDS.classes(w),
            _ => KwSet::empty(),
        }
    }

    pub fn is_kw(&self, kw: crate::keywords::Kw) -> bool {
        self.kw_set().contains(kw)
    }

    pub fn punct(&self) -> Option<Punct> {
        match self.kind {
            TokKind::Punct(p) => Some(p),
            _ => None,
        }
    }

    pub fn is_punct(&self, p: Punct) -> bool {
        self.punct() == Some(p)
    }

    pub fn describe(&self) -> String {
        match &self.kind {
            TokKind::Word(w) => format!("标识符/关键字 \"{w}\""),
            TokKind::Str(w) => format!("转义标识符 \"{w}\""),
            TokKind::Int(n) => format!("整数 {n}"),
            TokKind::Float(x) => format!("浮点数 {x}"),
            TokKind::Punct(p) => format!("\"{}\"", p.str()),
            TokKind::Newline => "换行".to_string(),
            TokKind::Indent => "缩进".to_string(),
            TokKind::Dedent => "反缩进".to_string(),
            TokKind::Eof => "文件结束".to_string(),
        }
    }
}

const TAB_WIDTH: usize = 4;

struct Lexer<'a> {
    src: &'a SourceFile,
    chars: Vec<(usize, char)>,
    i: usize,
    toks: Vec<Token>,
    levels: Vec<usize>,
    paren: usize,
    /// 未闭合的括号：`(开括号, 偏移)`，用于"括号不匹配"错误定位。
    paren_stack: Vec<(Punct, usize)>,
    paren_kinds: Vec<Punct>,
    at_line_start: bool,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a SourceFile) -> Lexer<'a> {
        Lexer {
            src,
            chars: src.text.char_indices().collect(),
            i: 0,
            toks: Vec::new(),
            levels: vec![0],
            paren: 0,
            paren_stack: Vec::new(),
            paren_kinds: Vec::new(),
            at_line_start: true,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.i).map(|(_, c)| *c)
    }

    fn peek_at(&self, k: usize) -> Option<char> {
        self.chars.get(self.i + k).map(|(_, c)| *c)
    }

    fn offset(&self) -> usize {
        self.chars.get(self.i).map(|(o, _)| *o).unwrap_or(self.src.text.len())
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.i += 1;
        }
        c
    }

    fn push(&mut self, kind: TokKind, start: usize, end: usize) {
        self.toks.push(Token { kind, span: Span::new(start, end) });
    }

    fn error(&self, code: &'static str, msg: impl Into<String>, start: usize, end: usize) -> Diag {
        Diag::error(code, msg, Span::new(start, end))
    }

    fn run(mut self) -> Result<Vec<Token>> {
        loop {
            if self.at_line_start && self.paren == 0 {
                if !self.handle_indent()? {
                    continue; // 空行/注释行已整行跳过
                }
            }
            // 注意：必须在缩进处理**之后**再取当前字符，
            // 否则 `match c` 用的是缩进前的过期字符，会多消费一个字符。
            let Some(c) = self.peek() else { break };
            match c {
                ' ' | '\t' | '\u{3000}' => {
                    self.bump();
                }
                '\n' => {
                    self.bump();
                    if self.paren == 0 {
                        let off = self.offset();
                        self.push(TokKind::Newline, off, off);
                        self.at_line_start = true;
                    }
                }
                '#' | '＃' => self.skip_line_comment(),
                '/' if self.peek_at(1) == Some('/') || self.peek_at(1) == Some('／') => {
                    self.skip_line_comment()
                }
                '/' if self.peek_at(1) == Some('*') => self.skip_block_comment()?,
                '"' | '\'' | '“' | '”' | '「' | '」' => self.read_string()?,
                c if c.is_ascii_digit() => self.read_number()?,
                c if is_ident_start(c) => self.read_word(),
                c => self.read_punct(c)?,
            }
        }
        // 收尾：括号未闭合
        if let Some((open, off)) = self.paren_stack.last().copied() {
            return Err(self
                .error(
                    "E004",
                    format!("括号不匹配：`{}` 没有对应的 `{}`", open.str(), open.closer().unwrap().str()),
                    off,
                    off + 1,
                )
                .with_hint(format!("在文件末尾补一个 `{}`", open.closer().unwrap().str())));
        }
        // 收尾：补齐 Dedent
        let off = self.src.text.len();
        while self.levels.len() > 1 {
            self.levels.pop();
            self.push(TokKind::Dedent, off, off);
        }
        self.push(TokKind::Eof, off, off);
        Ok(self.toks)
    }

    /// 行首缩进处理。返回 false 表示"整行被吞掉"（空行或注释行）。
    fn handle_indent(&mut self) -> Result<bool> {
        self.at_line_start = false;
        let start = self.offset();
        let mut width = 0usize;
        while let Some(c) = self.peek() {
            match c {
                ' ' => width += 1,
                '\t' => width += TAB_WIDTH,
                '\u{3000}' => width += 2,
                _ => break,
            }
            self.bump();
        }
        // 空行或整行注释：跳过，不产生任何 token（含换行）
        match self.peek() {
            None => return Ok(true),
            Some('\n') => {
                self.bump();
                self.at_line_start = true;
                return Ok(false);
            }
            Some('#') | Some('＃') => {
                self.skip_line_comment();
                if self.peek() == Some('\n') {
                    self.bump();
                }
                self.at_line_start = true;
                return Ok(false);
            }
            Some('/') if self.peek_at(1) == Some('/') || self.peek_at(1) == Some('*') => {
                if self.peek_at(1) == Some('*') {
                    self.skip_block_comment()?;
                } else {
                    self.skip_line_comment();
                }
                if self.peek() == Some('\n') {
                    self.bump();
                }
                self.at_line_start = true;
                return Ok(false);
            }
            _ => {}
        }
        let _ = start;
        let top = *self.levels.last().unwrap();
        if width > top {
            self.levels.push(width);
            let off = self.offset();
            self.push(TokKind::Indent, off, off);
        } else if width < top {
            while self.levels.len() > 1 && width < *self.levels.last().unwrap() {
                self.levels.pop();
                let off = self.offset();
                self.push(TokKind::Dedent, off, off);
            }
        }
        Ok(true)
    }

    fn skip_line_comment(&mut self) {
        while let Some(c) = self.peek() {
            if c == '\n' {
                break;
            }
            self.bump();
        }
    }

    fn skip_block_comment(&mut self) -> Result<()> {
        let start = self.offset();
        self.bump(); // '/'
        self.bump(); // '*'
        loop {
            match self.bump() {
                None => {
                    return Err(self.error(
                        "E010",
                        "块注释没有闭合：缺少 `*/`",
                        start,
                        self.src.text.len(),
                    )
                    .with_hint("补一个 `*/`，或者改用行注释 `#` / `//`"));
                }
                Some('*') if self.peek() == Some('/') => {
                    self.bump();
                    return Ok(());
                }
                _ => {}
            }
        }
    }

    fn read_string(&mut self) -> Result<()> {
        let start = self.offset();
        let open = self.bump().unwrap();
        let close = match open {
            '“' | '”' => '”',
            '「' => '」',
            '\'' => '\'',
            _ => '"',
        };
        let mut buf = String::new();
        loop {
            match self.bump() {
                None | Some('\n') => {
                    return Err(self
                        .error("E010", "字符串没有闭合：缺少结束引号", start, self.offset())
                        .with_hint(format!("在第 {} 列的 `{}` 后面补上结束引号", 1, open)));
                }
                Some(c) if c == close || (close == '"' && c == '”') => break,
                Some('\\') => {
                    if let Some(e) = self.bump() {
                        buf.push(match e {
                            'n' => '\n',
                            't' => '\t',
                            other => other,
                        });
                    }
                }
                Some(c) => buf.push(c),
            }
        }
        let end = self.offset();
        self.push(TokKind::Str(buf), start, end);
        Ok(())
    }

    fn read_number(&mut self) -> Result<()> {
        let start = self.offset();
        let mut text = String::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                text.push(c);
                self.bump();
            } else {
                break;
            }
        }
        let mut is_float = false;
        if self.peek() == Some('.') && self.peek_at(1).map(|c| c.is_ascii_digit()).unwrap_or(false) {
            is_float = true;
            text.push('.');
            self.bump();
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    text.push(c);
                    self.bump();
                } else {
                    break;
                }
            }
        }
        if matches!(self.peek(), Some('e') | Some('E')) {
            let save = self.i;
            let mut exp = String::from("e");
            self.bump();
            if matches!(self.peek(), Some('+') | Some('-')) {
                exp.push(self.bump().unwrap());
            }
            let mut any = false;
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    exp.push(c);
                    self.bump();
                    any = true;
                } else {
                    break;
                }
            }
            if any {
                is_float = true;
                text.push_str(&exp);
            } else {
                self.i = save;
            }
        }
        let end = self.offset();
        if is_float {
            let v: f64 = text.parse().map_err(|_| {
                self.error("E010", format!("浮点数无法解析：{text}"), start, end)
            })?;
            self.push(TokKind::Float(v), start, end);
        } else {
            match text.parse::<i64>() {
                Ok(n) => self.push(TokKind::Int(n), start, end),
                Err(_) => {
                    // 超长整数：退化为浮点，宽容但不丢信息
                    let v: f64 = text.parse().unwrap_or(0.0);
                    self.push(TokKind::Float(v), start, end);
                }
            }
        }
        Ok(())
    }

    fn read_word(&mut self) {
        let start = self.offset();
        let mut buf = String::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                // 第四部分允许标识符含数字（`k0`/`x1`/`层1`），但 `重复24次`
                // 这种"中文 + 数字串 + 中文"必须切成 词/数字/词，否则示例 F 解析不了。
                let mut k = 0;
                while self
                    .peek_at(k)
                    .map(|x| x.is_ascii_digit())
                    .unwrap_or(false)
                {
                    k += 1;
                }
                if buf.is_empty() || self.peek_at(k).map(is_cjk_alpha).unwrap_or(false) {
                    break;
                }
                buf.push(c);
                self.bump();
            } else if is_ident_continue(c) {
                buf.push(c);
                self.bump();
            } else {
                break;
            }
        }
        let end = self.offset();
        self.push(TokKind::Word(buf), start, end);
    }

    fn read_punct(&mut self, c: char) -> Result<()> {
        let start = self.offset();
        self.bump();
        let end = self.offset();
        let p = match c {
            '@' | '＠' => Punct::At,
            '=' if self.peek() == Some('>') => {
                self.bump();
                Punct::Arrow
            }
            '=' | '＝' => Punct::Eq,
            '→' | '➜' | '⇒' => Punct::Arrow,
            '-' if matches!(self.peek(), Some('>') | Some('＞')) => {
                self.bump();
                Punct::Arrow
            }
            '－' if matches!(self.peek(), Some('>') | Some('＞')) => {
                self.bump();
                Punct::Arrow
            }
            ':' | '：' => Punct::Colon,
            ';' | '；' => Punct::Semi,
            ',' | '，' | '、' | '‚' => Punct::Comma,
            '(' | '（' => Punct::LParen,
            ')' | '）' => Punct::RParen,
            '{' | '｛' => Punct::LBrace,
            '}' | '｝' => Punct::RBrace,
            '[' | '【' | '〔' => Punct::LBracket,
            ']' | '】' | '〕' => Punct::RBracket,
            '*' | '＊' | '×' => Punct::Star,
            '+' | '＋' => Punct::Plus,
            '-' | '－' | '−' => Punct::Minus,
            '/' | '／' => Punct::Slash,
            '.' => Punct::Dot,
            '。' => Punct::Dot,
            '!' | '！' => Punct::Bang,
            '<' | '＜' => Punct::Lt,
            '>' | '＞' => Punct::Gt,
            '%' | '％' => Punct::Percent,
            '&' | '＆' => Punct::Amp,
            '|' | '｜' => Punct::Pipe,
            '?' | '？' | '~' | '～' | '^' | '`' | '\\' => Punct::Other,
            _ => {
                return Err(self
                    .error("E010", format!("无法识别的字符 `{c}`"), start, end)
                    .with_hint("删掉它，或者用引号包成标识符：\"关键字\""));
            }
        };
        if let Some(close) = p.closer() {
            self.paren += 1;
            self.paren_kinds.push(close);
            self.paren_stack.push((p, start));
        } else if p == Punct::RParen || p == Punct::RBrace || p == Punct::RBracket {
            match self.paren_stack.pop() {
                None => {
                    return Err(self
                        .error("E004", format!("多余的 `{}`：没有与之配对的左括号", p.str()), start, end)
                        .with_hint("删掉它，或者补上对应的左括号"));
                }
                Some((open, off)) if open.closer() != Some(p) => {
                    return Err(self
                        .error(
                            "E004",
                            format!(
                                "括号不匹配：开出的是 `{}`，闭合的是 `{}`",
                                open.str(),
                                p.str()
                            ),
                            start,
                            end,
                        )
                        .with_note(format!(
                            "`{}` 开在这里：{}",
                            open.str(),
                            self.src.loc_str(Span::new(off, off + 1))
                        ))
                        .with_hint(format!("把 `{}` 改成 `{}`", p.str(), open.closer().unwrap().str())));
                }
                Some(_) => {
                    self.paren = self.paren.saturating_sub(1);
                    self.paren_kinds.pop();
                }
            }
        }
        self.push(TokKind::Punct(p), start, end);
        Ok(())
    }
}

pub fn is_ident_start(c: char) -> bool {
    c == '_' || c.is_alphabetic()
}

pub fn is_ident_continue(c: char) -> bool {
    c == '_' || c.is_alphanumeric()
}

/// 非 ASCII 字母（中文字）。
pub fn is_cjk_alpha(c: char) -> bool {
    !c.is_ascii() && c.is_alphabetic()
}

/// 中文数字字符（第十三部分：`一二三` 与 `123` 都接受）。
///
/// 注意：中文数字是**字母**，所以词法层把它们留在 `Word` 里
/// （`重复二十四次` → `["重复","二十四","次"]`），由 parser 在需要 `Expr` 的地方
/// 用 [`cn_number_value`] 转成整数。这样 `一年`/`一致` 这类正常中文标识符不受影响。
pub fn is_cn_digit(c: char) -> bool {
    matches!(
        c,
        '零' | '〇' | '一' | '二' | '两' | '三' | '四' | '五' | '六' | '七' | '八' | '九' | '十' | '百' | '千'
    )
}

fn cn_num_value(c: char) -> Option<i64> {
    Some(match c {
        '零' | '〇' => 0,
        '一' => 1,
        '二' | '两' => 2,
        '三' => 3,
        '四' => 4,
        '五' => 5,
        '六' => 6,
        '七' => 7,
        '八' => 8,
        '九' => 9,
        _ => return None,
    })
}

/// 支持 `十`/`二十四`/`一百零二`/`十二`；不支持的组合返回 `None`。
pub fn cn_number_value(s: &str) -> Option<i64> {
    let mut total = 0i64;
    let mut section = 0i64;
    let mut num = 0i64;
    let mut any = false;
    for c in s.chars() {
        any = true;
        match c {
            '十' => {
                section += if num == 0 { 10 } else { num * 10 };
                num = 0;
            }
            '百' => {
                section += if num == 0 { 100 } else { num * 100 };
                num = 0;
            }
            '千' => {
                section += if num == 0 { 1000 } else { num * 1000 };
                num = 0;
            }
            _ => {
                let d = cn_num_value(c)?;
                if d == 0 {
                    // 零：跳过（一百零二）
                } else {
                    num = d;
                }
            }
        }
    }
    if !any {
        return None;
    }
    total += section + num;
    Some(total)
}

/// 入口。
pub fn lex(src: &SourceFile) -> Result<Vec<Token>> {
    Lexer::new(src).run()
}

// ---------------------------------------------------------------------------
// Lexer 自检（模块 2 验收：标点/缩进/注释/中文数字/裸指令 归一化）
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::diag::SourceFile;

    fn toks(s: &str) -> Vec<Token> {
        lex(&SourceFile::new("t.sparrow", s)).unwrap()
    }

    fn kinds(s: &str) -> Vec<TokKind> {
        toks(s).into_iter().map(|t| t.kind).collect()
    }

    /// 忽略换行后的 token 序列（缩进/标点归一化的等价性比较）。
    fn shape(s: &str) -> Vec<TokKind> {
        kinds(s)
            .into_iter()
            .filter(|k| !matches!(k, TokKind::Newline))
            .collect()
    }

    #[test]
    fn punctuation_normalized() {
        assert_eq!(shape("a，b、c；d：e（f）"), shape("a,b,c;d:e(f)"));
        // `。` 按第四部分是分隔符（与 `.` 同类），不是块结束
        assert_eq!(shape("a。b"), shape("a.b"));
    }

    #[test]
    fn arrows_and_brackets() {
        assert_eq!(shape("a->b"), shape("a → b"));
        assert_eq!(shape("a=>b"), shape("a->b"));
        assert_eq!(shape("【x】"), shape("[x]"));
        // 开始/结束 与 { } 是两套块标记，各自保留原 token；等价性在 parser 层验证
        assert!(toks("模型 小 开始")[2].is_kw(crate::keywords::Kw::Start));
        assert!(toks("模型 小 结束")[2].is_kw(crate::keywords::Kw::End));
        assert!(toks("模型 小 { x }")[2].is_punct(Punct::LBrace));
    }

    #[test]
    fn cjk_digit_split() {
        assert_eq!(
            shape("重复24次"),
            vec![
                TokKind::Word("重复".into()),
                TokKind::Int(24),
                TokKind::Word("次".into()),
                TokKind::Eof
            ]
        );
        // 但 `k0`/`x1`/`层1` 仍是单个标识符（第四部分）
        assert_eq!(shape("k0"), vec![TokKind::Word("k0".into()), TokKind::Eof]);
        assert_eq!(shape("x1"), vec![TokKind::Word("x1".into()), TokKind::Eof]);
        assert_eq!(shape("层1"), vec![TokKind::Word("层1".into()), TokKind::Eof]);
    }

    #[test]
    fn indent_width_tolerated() {
        let two = "模型 小:\n  嵌入 512\n  线性 4054\n";
        let four = "模型 小:\n    嵌入 512\n    线性 4054\n";
        let eight = "模型 小:\n        嵌入 512\n        线性 4054\n";
        let tab = "模型 小:\n\t嵌入 512\n\t线性 4054\n";
        assert_eq!(shape(two), shape(four));
        assert_eq!(shape(two), shape(eight));
        assert_eq!(shape(two), shape(tab));
        // 同级宽度不统一也不报错
        let ragged = "模型 小:\n  嵌入 512\n     线性 4054\n";
        assert!(lex(&SourceFile::new("t", ragged)).is_ok());
    }

    #[test]
    fn comments_of_three_flavors() {
        assert_eq!(shape("# 头\n嵌入 512"), shape("嵌入 512"));
        assert_eq!(shape("// 头\n嵌入 512"), shape("嵌入 512"));
        assert_eq!(shape("/* 头\n 中间 */嵌入 512"), shape("嵌入 512"));
        assert_eq!(shape("嵌入 512 # 尾"), shape("嵌入 512"));
    }

    #[test]
    fn cn_numerals() {
        // 边界判断：中文数字是**字母**，词法层不拆词（拆了 `一致`/`一年` 就碎了）。
        // 只在 parser 的 Expr 位置用 cn_number_value 转换，所以：
        //   `重复 二十四 次` / `重复24次`  ✓
        //   `重复二十四次`（无空格）        ✗ —— 见报告"遗留项"
        assert_eq!(shape("重复24次")[1], TokKind::Int(24));
        assert_eq!(shape("重复 二十四 次")[1], TokKind::Word("二十四".into()));
        assert_eq!(shape("重复二十四次").len(), 2);
        assert_eq!(cn_number_value("二十四"), Some(24));
        assert_eq!(cn_number_value("一百零二"), Some(102));
        assert_eq!(cn_number_value("十"), Some(10));
        assert_eq!(cn_number_value("两"), Some(2));
        assert_eq!(cn_number_value("一致"), None);
        // `24层` 是第二部分明列的重复写法
        let t = toks("24层");
        assert_eq!(t[0].kind, TokKind::Int(24));
        assert_eq!(t[1].word(), Some("层"));
    }

    #[test]
    fn raw_op_adjacency_preserved() {
        let t = toks("@load r0 cpool+0");
        assert_eq!(t.len(), 7, "{:?}", t);
        assert_eq!(t[3].span.end, t[4].span.start, "cpool 与 + 相邻");
        assert_eq!(t[4].span.end, t[5].span.start, "+ 与 0 相邻");
        assert!(t[0].is_punct(Punct::At));
        assert_eq!(t[3].word(), Some("cpool"));
    }

    #[test]
    fn escaped_keyword_becomes_word() {
        let t = toks("\"连接\"");
        assert_eq!(t[0].kind, TokKind::Str("连接".into()));
        assert!(t[0].kw_set().is_empty(), "转义后不再是关键字");
    }

    #[test]
    fn keyword_classes_available_on_token() {
        let t = toks("到");
        assert!(t[0].is_kw(crate::keywords::Kw::Connect));
        assert!(t[0].is_kw(crate::keywords::Kw::To));
    }

    #[test]
    fn unbalanced_paren_reports_opener() {
        let src = SourceFile::new("t.sparrow", "x = 稀疏(1536, 8");
        let e = lex(&src).unwrap_err();
        assert_eq!(e.code, "E004");
        let out = e.render(&src);
        assert!(out.contains("括号不匹配"), "{out}");
        assert!(out.contains("t.sparrow:1:7"), "{out}");
        assert!(out.contains('^'), "{out}");
    }

    #[test]
    fn extra_closer_reports() {
        let src = SourceFile::new("t.sparrow", "x = 稀疏 1536)");
        let e = lex(&src).unwrap_err();
        assert_eq!(e.code, "E004");
        assert!(e.render(&src).contains("多余的"), "{}", e.render(&src));
    }

    #[test]
    fn chinese_column_alignment() {
        // 中文列宽：`嵌入` 占 4 列，插入符号要对齐到第 3 个字符处
        let src = SourceFile::new("t.sparrow", "  嵌入 Spars(1)");
        // `Spars` 的字节区间是 9..14，字符列是 6
        let e = Diag::error("E002", "test", Span::new(9, 14));
        let out = e.render(&src);
        assert!(out.contains("1:6"), "{out}");
        assert!(out.contains("^^^^^"), "{out}");
        // 插入符号前有 14 列 = 7 列行号槽（"   1 | "）+ 7 列显示宽度（"  嵌入 "）
        let caret_line = out.lines().find(|l| l.contains('^')).unwrap();
        assert_eq!(caret_line.find('^').unwrap(), 14, "{out}");
    }
}
