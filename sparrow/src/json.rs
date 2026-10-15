//! Sparrow V0.1 —— AST JSON 序列化（对应验收 12：`--emit-ast-json`）
//!
//! 手写序列化，**零依赖**（任务书第十三部分：只做前端，不引入编译器框架）。
//! 输出确定性：键顺序固定、无哈希表遍历、缩进 2 空格、UTF-8 原文不转义
//! （中文标识符在 JSON 里保持可读）。

use crate::ast::{Ast, Expr, Ident, Item, Span, Stmt};
use crate::diag::escape_json;

struct J {
    out: String,
    indent: usize,
}

impl J {
    fn new() -> J {
        J { out: String::new(), indent: 0 }
    }

    fn line(&mut self, s: &str) {
        for _ in 0..self.indent {
            self.out.push_str("  ");
        }
        self.out.push_str(s);
        self.out.push('\n');
    }

    fn open(&mut self, s: &str) {
        self.line(s);
        self.indent += 1;
    }

    fn close(&mut self, s: &str) {
        self.indent -= 1;
        self.line(s);
    }
}

fn span_json(span: Span) -> String {
    format!("{{\"start\":{},\"end\":{}}}", span.start, span.end)
}

fn ident_json(id: &Ident) -> String {
    format!("\"{}\"", escape_json(&id.name))
}

pub fn ast_to_json(ast: &Ast) -> String {
    let mut j = J::new();
    j.open("{");
    j.line("\"kind\": \"Program\",");
    let items = ast.items();
    j.line("\"items\": [");
    j.indent += 1;
    for (i, item) in items.iter().enumerate() {
        item_json(&mut j, item);
        if i + 1 < items.len() {
            if let Some(last) = j.out.rfind('\n') {
                j.out.insert(last, ',');
            }
        }
    }
    j.indent -= 1;
    j.line("]");
    j.close("}");
    j.out
}

fn item_json(j: &mut J, item: &Item) {
    match item {
        Item::ModelDef { name, body, span } => {
            j.open("{");
            j.line("\"item\": \"ModelDef\",");
            j.line(&format!("\"name\": {},", ident_json(name)));
            j.line(&format!("\"span\": {},", span_json(*span)));
            j.line("\"body\": [");
            j.indent += 1;
            stmt_list_json(j, body);
            j.indent -= 1;
            j.line("]");
            j.close("}");
        }
        Item::LayerDef { name, params, kind, attrs, span } => {
            j.open("{");
            j.line("\"item\": \"LayerDef\",");
            j.line(&format!("\"name\": {},", ident_json(name)));
            j.line(&format!("\"kind\": {},", ident_json(kind)));
            j.line(&format!("\"params\": [{}],", params.iter().map(ident_json).collect::<Vec<_>>().join(", ")));
            j.line("\"attrs\": [");
            j.indent += 1;
            attr_list_json(j, attrs);
            j.indent -= 1;
            j.line("],");
            j.line(&format!("\"span\": {}", span_json(*span)));
            j.close("}");
        }
    }
}

fn attr_list_json(j: &mut J, attrs: &[(Ident, Expr)]) {
    for (i, (k, v)) in attrs.iter().enumerate() {
        let tail = if i + 1 < attrs.len() { "," } else { "" };
        j.line(&format!(
            "{{\"attr\": {}, \"value\": {}}}{}",
            ident_json(k),
            expr_json(v),
            tail
        ));
    }
}

fn stmt_list_json(j: &mut J, stmts: &[Stmt]) {
    for (i, s) in stmts.iter().enumerate() {
        stmt_json(j, s);
        if i + 1 < stmts.len() {
            if let Some(last) = j.out.rfind('\n') {
                j.out.insert(last, ',');
            }
        }
    }
}

fn stmt_json(j: &mut J, s: &Stmt) {
    match s {
        Stmt::Instantiate { name, kind, args, span } => {
            j.open("{");
            j.line("\"stmt\": \"Instantiate\",");
            j.line(&format!("\"name\": {},", ident_json(name)));
            j.line(&format!("\"kind\": {},", ident_json(kind)));
            j.line(&format!("\"span\": {},", span_json(*span)));
            j.line("\"args\": [");
            j.indent += 1;
            expr_list_json(j, args);
            j.indent -= 1;
            j.line("]");
            j.close("}");
        }
        Stmt::Repeat { body, count, span } => {
            j.open("{");
            j.line("\"stmt\": \"Repeat\",");
            j.line(&format!("\"count\": {},", expr_json(count)));
            j.line(&format!("\"span\": {},", span_json(*span)));
            j.line("\"body\":");
            j.indent += 1;
            stmt_json(j, body);
            j.indent -= 1;
            j.close("}");
        }
        Stmt::Connect { path, span } => {
            j.open("{");
            j.line("\"stmt\": \"Connect\",");
            j.line(&format!("\"path\": [{}],", path.iter().map(ident_json).collect::<Vec<_>>().join(", ")));
            j.line(&format!("\"span\": {}", span_json(*span)));
            j.close("}");
        }
        Stmt::Forward { input, output, span } => {
            j.open("{");
            j.line("\"stmt\": \"Forward\",");
            j.line(&format!("\"input\": {},", ident_json(input)));
            j.line(&format!("\"output\": {},", ident_json(output)));
            j.line(&format!("\"span\": {}", span_json(*span)));
            j.close("}");
        }
        Stmt::CustomOpDef { name, opcode, args, span } => {
            j.open("{");
            j.line("\"stmt\": \"CustomOpDef\",");
            j.line(&format!("\"name\": {},", ident_json(name)));
            j.line(&format!("\"opcode\": {},", expr_json(opcode)));
            j.line(&format!("\"args\": [{}],", args.iter().map(ident_json).collect::<Vec<_>>().join(", ")));
            j.line(&format!("\"span\": {}", span_json(*span)));
            j.close("}");
        }
        Stmt::RawOp { opcode, operands, span } => {
            j.open("{");
            j.line("\"stmt\": \"RawOp\",");
            j.line(&format!("\"opcode\": {},", ident_json(opcode)));
            j.line(&format!("\"span\": {},", span_json(*span)));
            j.line("\"operands\": [");
            j.indent += 1;
            expr_list_json(j, operands);
            j.indent -= 1;
            j.line("]");
            j.close("}");
        }
        // V0.1.6：训练陈述（`损失 均方` / `反向`）
        Stmt::Loss { kind, span } => {
            j.open("{");
            j.line("\"stmt\": \"Loss\",");
            j.line(&format!("\"kind\": {},", ident_json(kind)));
            j.line(&format!("\"span\": {}", span_json(*span)));
            j.close("}");
        }
        Stmt::Backward { span } => {
            j.open("{");
            j.line("\"stmt\": \"Backward\",");
            j.line(&format!("\"span\": {}", span_json(*span)));
            j.close("}");
        }
    }
}

fn expr_list_json(j: &mut J, exprs: &[Expr]) {
    for (i, e) in exprs.iter().enumerate() {
        let tail = if i + 1 < exprs.len() { "," } else { "" };
        j.line(&format!("{}{}", expr_json(e), tail));
    }
}

pub fn expr_json(e: &Expr) -> String {
    match e {
        Expr::Int(n) => format!("{{\"expr\": \"Int\", \"value\": {n}}}"),
        Expr::Float(x) => format!("{{\"expr\": \"Float\", \"value\": {}}}", crate::keywords::fmt_f64(*x)),
        Expr::Ident(s) => format!("{{\"expr\": \"Ident\", \"name\": \"{}\"}}", escape_json(s)),
        Expr::Named { key, value } => format!(
            "{{\"expr\": \"Named\", \"key\": {}, \"value\": {}}}",
            ident_json(key),
            expr_json(value)
        ),
        Expr::Arrow(items) => format!(
            "{{\"expr\": \"Arrow\", \"items\": [{}]}}",
            items.iter().map(expr_json).collect::<Vec<_>>().join(", ")
        ),
    }
}
