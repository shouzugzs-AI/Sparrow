//! Sparrow —— 架构级 IR 输出（后缀 `.spar`）
//!
//! 命名约定（V0.1.1 第一部分）：
//! ```text
//! .sparrow  Sparrow 源码
//! .spar     架构描述 IR   ← 本模块输出（V0.1 里曾叫 .vml）
//! .vml      opcode 序列    ← 汇编器输入（由 vmlgen 后端生成）
//! .vmb      opcode 字节码  ← 汇编器输出 / 引擎输入
//! ```
//!
//! 输出是**纯函数**：同一 canonical IR 永远得到同一串字节。因此所有"写法不同、
//! 语义相同"的输入都会得到同一个 sha256（验收 3 的核心）。
//!
//! 确定性来源：
//! * 属性按 `(规范键的固定序位, 键名)` 排序，与源码书写顺序无关；
//! * 数值用 [`keywords::fmt_f64`]（`2.0` → `2`）；
//! * 隐式实例名换成 canonical 层类型名（`嵌入`/`Embed` → `EMBED`）；
//! * 行尾只写 `\n`，文件末尾一个换行；
//! * 头部注释里没有时间戳/路径（否则 sha256 不可能稳定）。
//!
//! `.spar` 语法（架构级；下游是 vmlgen 后端）：
//!
//! ```text
//! ; sparrow v0.1
//! ; model: small
//! .model small
//!   LAYER   <name> <KIND> k=v ...
//!   REPEAT  <n> LAYER <name> <KIND> k=v ...
//!   CONNECT <a> <b> <c>
//!   FORWARD <in> <out>
//!   COP     <name> opcode=<expr> args(<a>,<b>)
//!   RAW     <opcode> <operand> ...
//! .end
//! ```

use crate::ast::Expr;
use crate::diag::Result;
use crate::keywords::{act_value_canonical, attr_rank, fmt_f64};
use crate::sema::{lower, CItem, Lowered};

/// `.spar` 头部。注意：不含时间戳/路径，否则 sha256 不可能稳定。
pub const SPAR_HEADER: &str = "; sparrow v0.1 — canonical .spar IR (architecture level)";

/// 旧名（V0.1 的 `.vml`）保留为别名，避免调用方大面积改动。
pub const VML_HEADER: &str = SPAR_HEADER;

/// IR → .vml 文本。
pub fn emit(lowered: &Lowered) -> String {
    let mut out = String::new();
    out.push_str(VML_HEADER);
    out.push('\n');
    out.push_str(&format!("; model: {}\n", lowered.model));
    if lowered.all_models.len() > 1 {
        let extra: Vec<&str> = lowered
            .all_models
            .iter()
            .skip(1)
            .map(|s| s.as_str())
            .collect();
        out.push_str(&format!("; 其它模型（本前端只输出第一个）: {}\n", extra.join(", ")));
    }
    out.push_str(&format!(".model {}\n", lowered.model));
    for item in &lowered.items {
        out.push_str(&emit_item(item));
    }
    out.push_str(".end\n");
    out
}

fn emit_item(item: &CItem) -> String {
    match item {
        CItem::Def { kind, name, params, .. } => {
            let ps = if params.is_empty() {
                String::new()
            } else {
                format!(" params({})", params.join(","))
            };
            format!("  DEF     {kind} {name}{ps}\n")
        }
        CItem::Layer { name, kind, attrs, repeat, .. } => {
            let body = format!("{name} {kind}{}", render_attrs(attrs));
            match repeat {
                Some(n) => format!("  REPEAT  {n} LAYER {body}\n"),
                None => format!("  LAYER   {body}\n"),
            }
        }
        CItem::OpCall { name, op, args, repeat, .. } => {
            let a = if args.is_empty() {
                String::new()
            } else {
                render_attrs(args)
            };
            match repeat {
                Some(n) => format!("  REPEAT  {n} OPCALL {name} {op}{a}\n"),
                None => format!("  OPCALL  {name} {op}{a}\n"),
            }
        }
        CItem::Connect { path, .. } => format!("  CONNECT {}\n", path.join(" ")),
        CItem::Forward { input, output, .. } => format!("  FORWARD {input} {output}\n"),
        CItem::CustomOp { name, opcode, args, .. } => {
            let a = if args.is_empty() {
                "-".to_string()
            } else {
                args.join(",")
            };
            format!("  COP     {name} opcode={} args({a})\n", render_expr(opcode))
        }
        CItem::Raw { opcode, operands, .. } => {
            let ops: Vec<String> = operands.iter().map(render_expr).collect();
            if ops.is_empty() {
                format!("  RAW     {opcode}\n")
            } else {
                format!("  RAW     {opcode} {}\n", ops.join(" "))
            }
        }
        // V0.1.6：训练陈述。输出成**两行固定文本**（无时间戳/无路径），
        // 于是"损失/反向写法不同、语义相同"仍然落到同一个 sha256。
        CItem::Loss { kind, .. } => format!("  LOSS    {kind}\n"),
        CItem::Backward { .. } => "  BACKWARD\n".to_string(),
    }
}

/// 属性渲染：固定排序 + 规范键名；`act` 的取值小写化。
pub fn render_attrs(attrs: &[(String, Expr)]) -> String {
    let mut sorted: Vec<&(String, Expr)> = attrs.iter().collect();
    sorted.sort_by(|a, b| {
        attr_rank(&a.0)
            .cmp(&attr_rank(&b.0))
            .then_with(|| a.0.cmp(&b.0))
    });
    let mut s = String::new();
    for (k, v) in sorted {
        let vs = if k == "act" {
            match v {
                Expr::Ident(x) => act_value_canonical(x),
                other => render_expr(other),
            }
        } else {
            render_expr(v)
        };
        s.push(' ');
        s.push_str(k);
        s.push('=');
        s.push_str(&vs);
    }
    s
}

pub fn render_expr(e: &Expr) -> String {
    match e {
        Expr::Int(n) => n.to_string(),
        Expr::Float(x) => fmt_f64(*x),
        Expr::Ident(s) => s.clone(),
        Expr::Named { key, value } => format!("{}={}", key.name, render_expr(value)),
        Expr::Arrow(items) => items.iter().map(render_expr).collect::<Vec<_>>().join("->"),
    }
}

/// 便捷入口：源码已 parse 成 AST 时直接出 .vml。
pub fn emit_from_ast(ast: &crate::ast::Ast) -> Result<String> {
    Ok(emit(&lower(ast)?))
}
