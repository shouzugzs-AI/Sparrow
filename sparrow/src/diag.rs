//! Sparrow V0.1 —— 诊断与源文件定位（对应任务书第十一部分）
//!
//! 报告要求（第十一部分）逐条落地：
//! - 行号、列号（列按**字符**数，不是字节数，中文才准）
//! - 源码上下文行 + 插入符号 `^^^^`（按**显示宽度**对齐，中文占 2 列）
//! - 提示（`是不是想写 "Sparse"？`）
//! - 候选列表
//!
//! 不允许只报"语法错误"：每个 `Diag` 都带稳定的 `code`。

use crate::ast::Span;
use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Severity {
    Error,
    Warning,
}

/// 一条诊断。
#[derive(Clone, Debug)]
pub struct Diag {
    pub severity: Severity,
    /// 稳定错误码，如 `E002`。
    pub code: &'static str,
    pub message: String,
    pub span: Span,
    /// `是不是想写 "Sparse"？`
    pub hint: Option<String>,
    /// `候选：Sparse, Embed, Linear`
    pub candidates: Vec<String>,
    pub notes: Vec<String>,
}

impl Diag {
    pub fn error(code: &'static str, message: impl Into<String>, span: Span) -> Diag {
        Diag {
            severity: Severity::Error,
            code,
            message: message.into(),
            span,
            hint: None,
            candidates: Vec::new(),
            notes: Vec::new(),
        }
    }

    pub fn warning(code: &'static str, message: impl Into<String>, span: Span) -> Diag {
        Diag { severity: Severity::Warning, ..Diag::error(code, message, span) }
    }

    pub fn with_hint(mut self, hint: impl Into<String>) -> Diag {
        self.hint = Some(hint.into());
        self
    }

    pub fn with_candidates<I, S>(mut self, cands: I) -> Diag
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.candidates = cands.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Diag {
        self.notes.push(note.into());
        self
    }

    pub fn severity_cn(&self) -> &'static str {
        match self.severity {
            Severity::Error => "错误",
            Severity::Warning => "警告",
        }
    }

    /// 人类可读渲染（第十一部分的格式）。
    pub fn render(&self, src: &SourceFile) -> String {
        let mut out = String::new();
        let loc = src.loc(self.span.start);
        out.push_str(&format!(
            "{}[{}]：{}\n",
            self.severity_cn(),
            self.code,
            self.message
        ));
        out.push_str(&format!("  位置：{}:{}:{}\n", src.path, loc.line, loc.col));

        let line_text = src.line_text(loc.line);
        let gutter = loc.line.to_string();
        let gw = gutter.len().max(2);
        out.push_str(&format!("  {:>w$} | {}\n", gutter, line_text, w = gw));
        // 插入符号：按显示宽度对齐，中文占 2 列
        let pad = display_width(&line_text.chars().take(loc.col.saturating_sub(1)).collect::<String>());
        let caret_len = {
            let end = self.span.end.min(src.line_end(loc.line));
            let seg: String = src
                .text
                .get(self.span.start as usize..end as usize)
                .unwrap_or("")
                .chars()
                .take_while(|c| *c != '\n')
                .collect();
            display_width(&seg).max(1)
        };
        out.push_str(&format!(
            "  {:>w$} | {}{}\n",
            "",
            " ".repeat(pad),
            "^".repeat(caret_len),
            w = gw
        ));

        if let Some(h) = &self.hint {
            out.push_str(&format!("  提示：{h}\n"));
        }
        if !self.candidates.is_empty() {
            out.push_str(&format!("  候选：{}\n", self.candidates.join(", ")));
        }
        for n in &self.notes {
            out.push_str(&format!("  附注：{n}\n"));
        }
        out.trim_end().to_string()
    }

    /// 机器可读形态。
    pub fn to_json(&self, src: &SourceFile) -> String {
        let loc = src.loc(self.span.start);
        let end = src.loc(self.span.end);
        let mut s = String::from("{");
        s.push_str(&format!("\"severity\":\"{}\"", self.severity_cn()));
        s.push_str(&format!(",\"code\":\"{}\"", escape_json(self.code)));
        s.push_str(&format!(",\"message\":\"{}\"", escape_json(&self.message)));
        s.push_str(&format!(
            ",\"loc\":{{\"file\":\"{}\",\"line\":{},\"col\":{},\"end_line\":{},\"end_col\":{}}}",
            escape_json(&src.path),
            loc.line,
            loc.col,
            end.line,
            end.col
        ));
        if let Some(h) = &self.hint {
            s.push_str(&format!(",\"hint\":\"{}\"", escape_json(h)));
        }
        if !self.candidates.is_empty() {
            let c: Vec<String> = self.candidates.iter().map(|x| format!("\"{}\"", escape_json(x))).collect();
            s.push_str(&format!(",\"candidates\":[{}]", c.join(",")));
        }
        s.push('}');
        s
    }
}

impl fmt::Display for Diag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

impl std::error::Error for Diag {}

pub type Result<T> = std::result::Result<T, Diag>;

// ---------------------------------------------------------------------------
// 源文件 / 位置
// ---------------------------------------------------------------------------

/// 位置：行列均从 1 开始；`col` 是**字符**序号。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Loc {
    pub line: usize,
    pub col: usize,
}

/// 源文件。载入时统一换行符（`\r\n` → `\n`）并剥离 BOM，
/// 因此 `Span` 的字节偏移是**归一化后**文本的偏移（对外只用于渲染与切片，一致即可）。
pub struct SourceFile {
    pub path: String,
    pub text: String,
    starts: Vec<u32>,
}

impl SourceFile {
    pub fn new(path: impl Into<String>, text: impl Into<String>) -> SourceFile {
        let mut text: String = text.into();
        if text.starts_with('\u{feff}') {
            text.remove(0);
        }
        text = text.replace("\r\n", "\n").replace('\r', "\n");
        let mut starts = vec![0u32];
        for (i, b) in text.bytes().enumerate() {
            if b == b'\n' {
                starts.push((i + 1) as u32);
            }
        }
        SourceFile { path: path.into(), text, starts }
    }

    pub fn line_count(&self) -> usize {
        self.starts.len()
    }

    pub fn loc(&self, byte: u32) -> Loc {
        let mut byte = byte.min(self.text.len() as u32);
        // 容错：落到多字节字符中间时退回前一个边界
        while byte > 0 && !self.text.is_char_boundary(byte as usize) {
            byte -= 1;
        }
        let line_idx = match self.starts.binary_search(&byte) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        let start = self.starts[line_idx] as usize;
        let col = self.text[start..byte as usize].chars().count() + 1;
        Loc { line: line_idx + 1, col }
    }

    pub fn line_text(&self, line: usize) -> &str {
        if line == 0 || line > self.starts.len() {
            return "";
        }
        let start = self.starts[line - 1] as usize;
        let end = self
            .starts
            .get(line)
            .map(|e| *e as usize)
            .unwrap_or(self.text.len());
        self.text[start..end].trim_end_matches('\n')
    }

    pub fn line_end(&self, line: usize) -> u32 {
        self.starts
            .get(line)
            .map(|e| *e)
            .unwrap_or(self.text.len() as u32)
    }

    pub fn slice(&self, span: Span) -> &str {
        let a = (span.start as usize).min(self.text.len());
        let b = (span.end as usize).min(self.text.len());
        if a > b {
            return "";
        }
        self.text.get(a..b).unwrap_or("")
    }

    /// 渲染一段 span 的 `行:列`，报错用。
    pub fn loc_str(&self, span: Span) -> String {
        let l = self.loc(span.start);
        format!("{}:{}:{}", self.path, l.line, l.col)
    }
}

// ---------------------------------------------------------------------------
// 显示宽度（中文对齐）
// ---------------------------------------------------------------------------

pub fn char_width(c: char) -> usize {
    let u = c as u32;
    if c == '\t' {
        return 4;
    }
    let wide = (0x1100..=0x115F).contains(&u)
        || (0x2E80..=0x303E).contains(&u)
        || (0x3041..=0x33FF).contains(&u)
        || (0x3400..=0x4DBF).contains(&u)
        || (0x4E00..=0x9FFF).contains(&u)
        || (0xA000..=0xA4CF).contains(&u)
        || (0xAC00..=0xD7A3).contains(&u)
        || (0xF900..=0xFAFF).contains(&u)
        || (0xFE30..=0xFE6F).contains(&u)
        || (0xFF00..=0xFF60).contains(&u)
        || (0xFFE0..=0xFFE6).contains(&u)
        || (0x1F300..=0x1FAFF).contains(&u)
        || (0x20000..=0x3FFFD).contains(&u);
    if wide {
        2
    } else {
        1
    }
}

pub fn display_width(s: &str) -> usize {
    s.chars().map(char_width).sum()
}

pub fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}
