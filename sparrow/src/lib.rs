//! Sparrow V0.1 —— 文本前端库。
//!
//! 管线：`源码 → lexer → parser → sema → codegen → .vml`
//! 另有 `json`（`--emit-ast-json`）与 `sha256`（验收自检，零依赖）。
//!
//! 设计立场（第一、二部分）：Sparrow 是编程语言，AST 是唯一真相；
//! 宽容语法（关键字中英混用、标点归一、缩进随意、四套块机制）全部在
//! lexer/词表层消化，parser 只认 canonical 形式。

pub mod asmdriver;
pub mod ast;
pub mod codegen;
pub mod diag;
pub mod json;
pub mod keywords;
pub mod lexer;
pub mod libgen;
pub mod parser;
pub mod sema;
pub mod sha256;
pub mod vmbmeta;
pub mod vmlgen;

pub use ast::{Ast, AstStats, Expr, Ident, Item, Span, Stmt};
pub use codegen::{emit, emit_from_ast};
pub use diag::{Diag, Loc, Result, SourceFile};
pub use json::ast_to_json;
pub use keywords::{KeywordTable, Kw, KwSet, KEYWORDS, KINDS};
pub use lexer::{lex, Punct, TokKind, Token};
pub use parser::{parse, parse_tokens};
pub use sema::{lower, CItem, Lowered};

/// 一步到位：源码 → (架构级 `.spar` 文本, 规范化 IR)。
pub fn compile_spar(src: &SourceFile) -> Result<(String, Lowered)> {
    let ast = parse(src)?;
    let lowered = lower(&ast)?;
    Ok((emit(&lowered), lowered))
}

/// V0.1 的旧名（当时架构 IR 叫 `.vml`；V0.1.1 起改名 `.spar`）。
pub fn compile_vml(src: &SourceFile) -> Result<(String, Lowered)> {
    compile_spar(src)
}

/// 一步到位：源码 → opcode 级 `.vml`（V0.1.1 后端，默认展开 `重复`）。
pub fn compile_vml_backend(src: &SourceFile, src_path: &str) -> Result<vmlgen::Generated> {
    compile_vml_backend_with(src, src_path, vmlgen::GenOptions::default())
}

/// 一步到位（带后端开关；`loop_repeat` = P0-2 的循环化）。
pub fn compile_vml_backend_with(
    src: &SourceFile,
    src_path: &str,
    opts: vmlgen::GenOptions,
) -> Result<vmlgen::Generated> {
    let ast = parse(src)?;
    vmlgen::generate_from_src_with(&ast, src_path, opts)
}
