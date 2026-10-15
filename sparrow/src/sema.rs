//! Sparrow V0.1 —— 语义归一化与校验（AST → 规范化 IR）
//!
//! 这一层做四件事，全部服务于"多套写法 → 一个 canonical 形式"：
//!
//! 1. **层类型解析**：内置层类型（精确匹配）→ 用户 `定义` → 自定义算子 → 报 E002。
//! 2. **参数绑定**：位置参数按 [`keywords::bind_kind_positional`] 或 LayerDef 的形参
//!    顺序绑定成规范属性键；命名参数走 [`keywords::attr_canonical`]。
//! 3. **引用解析**：`连接 嵌入 稀疏 线性` 里的 `嵌入` 要落到具体实例。
//!    规则：先按实例名精确匹配，再按"canonical 层类型"匹配，取**该引用之前最后声明**的那个。
//! 4. **缺参校验**：E003（验收 11 的"参数缺失"）。
//!
//! 名称归一化是"中英对照同 .vml"的关键：
//! 隐式名字（`name == kind` 原文）在输出里一律换成 **canonical 层类型名**
//! （`嵌入` → `EMBED`，`Embed` → `EMBED`），显式名字保留原文。

use std::collections::HashMap;

use crate::ast::{Ast, Expr, Ident, Item, Span, Stmt};
use crate::diag::{Diag, Result};
use crate::keywords::{self, attr_canonical, kind_lookup, KINDS};

/// 规范化 IR 的一条。
#[derive(Clone, Debug)]
pub enum CItem {
    /// `定义` 模板（canonical 化后）。
    Def { kind: String, name: String, params: Vec<String>, span: Span },
    /// 层实例。`repeat` 是结构化重复次数（不是文本替换）。
    Layer { name: String, kind: String, attrs: Vec<(String, Expr)>, repeat: Option<i64>, span: Span },
    /// 自定义算子的实例化（第五部分 `我的激活` 单独一行）。
    OpCall { name: String, op: String, args: Vec<(String, Expr)>, repeat: Option<i64>, span: Span },
    Connect { path: Vec<String>, span: Span },
    Forward { input: String, output: String, span: Span },
    CustomOp { name: String, opcode: Expr, args: Vec<String>, span: Span },
    Raw { opcode: String, operands: Vec<Expr>, span: Span },
    /// `损失 均方`（V0.1.6）：训练目标的损失函数，`kind` 已 canonical 成 `mse`。
    Loss { kind: String, span: Span },
    /// `反向`（V0.1.6）：要求按拓扑反序发射反向遍历。
    Backward { span: Span },
}

impl CItem {
    pub fn span(&self) -> Span {
        match self {
            CItem::Def { span, .. }
            | CItem::Layer { span, .. }
            | CItem::OpCall { span, .. }
            | CItem::Connect { span, .. }
            | CItem::Forward { span, .. }
            | CItem::CustomOp { span, .. }
            | CItem::Raw { span, .. }
            | CItem::Loss { span, .. }
            | CItem::Backward { span, .. } => *span,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Lowered {
    pub model: String,
    pub items: Vec<CItem>,
    /// 所有模型名（多模型文件在报告里提示只输出第一个）。
    pub all_models: Vec<String>,
}

struct Inst {
    /// 输出用的名字。
    emitted: String,
    /// canonical 层类型（内置）或 LayerDef/自定义算子的原文名。
    kind: String,
    /// 声明位置（诊断时可回溯"最近一次声明在哪"）。
    span: Span,
}


/// 层类型的来源，决定缺参校验与输出形态。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum KindClass {
    /// 内置层类型（Embed/Sparse/Linear/…）
    Builtin,
    /// 用户 `定义`
    UserDef,
    /// 自定义算子
    CustomOp,
}

/// 层定义模板的形参 → 规范属性键。
struct DefInfo {
    kind: String,
    /// 形参名，按声明顺序。
    params: Vec<String>,
    /// 形参名 → 模板里的属性键（canonical）。
    param_key: HashMap<String, String>,
}

pub fn lower(ast: &Ast) -> Result<Lowered> {
    // --- 收集 `定义` ---
    let mut defs: HashMap<String, DefInfo> = HashMap::new();
    let mut def_order: Vec<String> = Vec::new();
    for item in ast.items() {
        if let Item::LayerDef { name, params, kind, attrs, .. } = item {
            let mut param_key = HashMap::new();
            for (i, p) in params.iter().enumerate() {
                // 模板里 value 恰好是该形参名 → 绑定它对应的键
                let mut key = p.name.clone();
                for (k, v) in attrs {
                    if let Expr::Ident(s) = v {
                        if keywords::eq_norm(s, &p.name) {
                            key = k.name.clone();
                            break;
                        }
                    }
                }
                let _ = i;
                param_key.insert(p.name.clone(), attr_canonical(&key));
            }
            if !def_order.iter().any(|d| keywords::eq_norm(d, &name.name)) {
                def_order.push(name.name.clone());
            }
            defs.insert(
                name.name.clone(),
                DefInfo {
                    kind: kind.name.clone(),
                    params: params.iter().map(|p| p.name.clone()).collect(),
                    param_key,
                },
            );
        }
    }

    // --- 收集自定义算子 ---
    let mut custom_ops: Vec<String> = Vec::new();
    for item in ast.items() {
        for st in item.body() {
            let mut push = |name: &str| {
                if !custom_ops.iter().any(|c| keywords::eq_norm(c, name)) {
                    custom_ops.push(name.to_string());
                }
            };
            st.walk(&mut |s| {
                if let Stmt::CustomOpDef { name, .. } = s {
                    push(&name.name);
                }
            });
        }
    }

    // --- 主模型 ---
    let models: Vec<&Item> = ast.items().iter().filter(|i| i.is_model()).collect();
    let all_models = models.iter().map(|m| m.name().name.clone()).collect();
    let Some(main) = models.first() else {
        return Err(Diag::error(
            "E013",
            "文件里没有任何模型：既没有 `模型 名:`，也没有顶层语句",
            Span::ZERO,
        )
        .with_hint("最低限度也要写一条 `嵌入 512`"));
    };

    let mut out: Vec<CItem> = Vec::new();

    // 边界判断：`定义` 模板**不进 .vml**。
    //
    // 用户层定义是编译期糖 —— 展开后的 LAYER 行已经携带全部信息。若把模板也
    // 写进 .vml，"定义式写法"与"直写式写法"就会得到不同哈希，直接违背验收 10
    // （语义相同 → 同一个 .vml）。模板信息在 `--emit-ast-json` 里仍可见。
    let _ = def_order;

    let mut instances: Vec<Inst> = Vec::new();
    let mut name_use: HashMap<String, usize> = HashMap::new();

    for st in main.body() {
        lower_stmt(st, &defs, &custom_ops, &mut instances, &mut name_use, &mut out)?;
    }

    if out.is_empty() {
        // Level 6 允许整个模型都是裸指令，所以只看"有没有语句"，不看是不是层实例
        return Err(Diag::error(
            "E014",
            format!("模型 `{}` 是空的", main.name().name),
            main.span(),
        )
        .with_hint("至少写一条 `嵌入 512`，或者一条 `@load ...`"));
    }

    // V0.1.6：`反向` 必须与 `损失` 配对 —— 反向的起点就是损失对输出的梯度。
    // 这条**不依赖后端**（`--emit-spar` 也要拦），所以放在语义层而不是 vmlgen。
    // vmlgen 里还有一份同样的检查：那是给"直接构造 Lowered/CItem"的调用方兜底，
    // 两边都在是刻意的（这一族错误都属于"不报错就会静默算错"）。
    if out.iter().any(|c| matches!(c, CItem::Backward { .. }))
        && !out.iter().any(|c| matches!(c, CItem::Loss { .. }))
    {
        let span = out
            .iter()
            .find_map(|c| match c {
                CItem::Backward { span } => Some(*span),
                _ => None,
            })
            .unwrap_or(Span::ZERO);
        return Err(
            Diag::error("E113", "`反向` 需要一个起点：模型里没有声明损失", span)
                .with_hint("先写一行 `损失 均方`（引擎侧是 `OP_MSE_BWD`），再写 `反向`"),
        );
    }

    Ok(Lowered { model: main.name().name.clone(), items: out, all_models })
}

fn lower_stmt(
    st: &Stmt,
    defs: &HashMap<String, DefInfo>,
    custom_ops: &[String],
    instances: &mut Vec<Inst>,
    name_use: &mut HashMap<String, usize>,
    out: &mut Vec<CItem>,
) -> Result<()> {
    match st {
        Stmt::Instantiate { name, kind, args, span } => {
            let (ckind, attrs, class) = resolve_kind(kind, args, defs, custom_ops)?;
            let emitted = emitted_name(name, kind, &ckind, name_use);
            if class == KindClass::Builtin && attrs.is_empty() {
                // 缺参：内置层类型至少要一个参数（验收 11）
                let params = kind_lookup(&kind.name)
                    .map(|k| k.params.join(", "))
                    .unwrap_or_else(|| "dim".to_string());
                return Err(Diag::error(
                    "E003",
                    format!("层类型 \"{}\" 缺少参数（至少要给 {params}）", kind.name),
                    *span,
                )
                .with_hint(format!("例：`{} 维度=512` 或 `{}(512)`", kind.name, kind.name))
                .with_candidates(params.split(", ").map(|s| s.to_string())));
            }
            match class {
                KindClass::CustomOp => out.push(CItem::OpCall {
                    name: emitted.clone(),
                    op: ckind.clone(),
                    args: attrs,
                    repeat: None,
                    span: *span,
                }),
                _ => out.push(CItem::Layer {
                    name: emitted.clone(),
                    kind: ckind.clone(),
                    attrs,
                    repeat: None,
                    span: *span,
                }),
            }
            instances.push(Inst { emitted, kind: ckind, span: *span });
            Ok(())
        }
        Stmt::Repeat { body, count, span } => {
            let n = match count {
                Expr::Int(n) => *n,
                other => {
                    return Err(Diag::error(
                        "E003",
                        format!("重复次数必须是整数，实际是 `{}`", other.render()),
                        *span,
                    )
                    .with_hint("写成 `重复24次` / `x 24` / `24层`"));
                }
            };
            if n <= 0 {
                return Err(Diag::error("E003", format!("重复次数必须是正数，实际是 {n}"), *span));
            }
            match &**body {
                Stmt::Instantiate { name, kind, args, span: sspan } => {
                    let (ckind, attrs, class) = resolve_kind(kind, args, defs, custom_ops)?;
                    let emitted = emitted_name(name, kind, &ckind, name_use);
                    match class {
                        KindClass::CustomOp => out.push(CItem::OpCall {
                            name: emitted.clone(),
                            op: ckind.clone(),
                            args: attrs,
                            repeat: Some(n),
                            span: *sspan,
                        }),
                        _ => out.push(CItem::Layer {
                            name: emitted.clone(),
                            kind: ckind.clone(),
                            attrs,
                            repeat: Some(n),
                            span: *sspan,
                        }),
                    }
                    instances.push(Inst { emitted, kind: ckind, span: *sspan });
                    Ok(())
                }
                other => Err(Diag::error(
                    "E005",
                    format!("`重复` 只能包住层实例化，实际是{}", other.kind()),
                    other.span(),
                )),
            }
        }
        Stmt::Connect { path, span } => {
            let mut resolved = Vec::new();
            for p in path {
                resolved.push(resolve_ref(p, instances)?);
            }
            out.push(CItem::Connect { path: resolved, span: *span });
            Ok(())
        }
        Stmt::Forward { input, output, span } => {
            let a = resolve_ref(input, instances)?;
            let b = resolve_ref(output, instances)?;
            out.push(CItem::Forward { input: a, output: b, span: *span });
            Ok(())
        }
        Stmt::CustomOpDef { name, opcode, args, span } => {
            out.push(CItem::CustomOp {
                name: name.name.clone(),
                opcode: opcode.clone(),
                args: args.iter().map(|a| a.name.clone()).collect(),
                span: *span,
            });
            Ok(())
        }
        Stmt::RawOp { opcode, operands, span } => {
            out.push(CItem::Raw { opcode: opcode.name.clone(), operands: operands.clone(), span: *span });
            Ok(())
        }
        // V0.1.6：`损失 均方` —— 损失函数名在这里 canonical 化（与层类型同一套做法）。
        Stmt::Loss { kind, span } => {
            let Some(c) = crate::keywords::loss_kind_canonical(&kind.name) else {
                let cands = crate::keywords::LOSS_CANDIDATES.join(" / ");
                return Err(Diag::error(
                    "E112",
                    format!("不认识的损失函数 `{}`", kind.name),
                    kind.span,
                )
                .with_hint(format!("候选：{cands}（现在只有均方误差，引擎侧是 `OP_MSE_BWD`）")));
            };
            out.push(CItem::Loss { kind: c.to_string(), span: *span });
            Ok(())
        }
        // V0.1.6：`反向` —— 语义上要求"先有损失"，这条约束在这里判定（AST 只记录陈述本身）。
        Stmt::Backward { span } => {
            out.push(CItem::Backward { span: *span });
            Ok(())
        }
    }
}

/// 层类型解析 + 参数绑定。
fn resolve_kind(
    kind: &Ident,
    args: &[Expr],
    defs: &HashMap<String, DefInfo>,
    custom_ops: &[String],
) -> Result<(String, Vec<(String, Expr)>, KindClass)> {
    // 1. 内置层类型
    if let Some(k) = kind_lookup(&kind.name) {
        let mut attrs = Vec::new();
        let mut positional = 0usize;
        let bound = keywords::bind_kind_positional(&kind.name, &positional_only(args));
        let mut bound_iter = bound.into_iter();
        for a in args {
            match a {
                Expr::Named { key, value } => {
                    attrs.push((attr_canonical(&key.name), (**value).clone()));
                }
                other => {
                    let _ = other;
                    if let Some((k2, v)) = bound_iter.next() {
                        attrs.push((k2, v));
                    } else {
                        attrs.push((format!("arg{positional}"), a.clone()));
                    }
                    positional += 1;
                }
            }
        }
        return Ok((k.canonical.to_string(), attrs, KindClass::Builtin));
    }
    // 2. 用户 `定义`
    if let Some(d) = defs.get(&kind.name).or_else(|| {
        defs.iter()
            .find(|(n, _)| keywords::eq_norm(n, &kind.name))
            .map(|(_, v)| v)
    }) {
        let mut attrs: Vec<(String, Expr)> = Vec::new();
        let mut positional = 0usize;
        for a in args {
            match a {
                Expr::Named { key, value } => {
                    attrs.push((attr_canonical(&key.name), (**value).clone()));
                }
                other => {
                    let pname = d.params.get(positional);
                    let key = match pname {
                        Some(p) => d
                            .param_key
                            .get(p)
                            .cloned()
                            .unwrap_or_else(|| attr_canonical(p)),
                        None => format!("arg{positional}"),
                    };
                    attrs.push((key, other.clone()));
                    positional += 1;
                }
            }
        }
        if args.len() < d.params.len() {
            return Err(Diag::error(
                "E003",
                format!(
                    "层类型 \"{}\" 需要 {} 个参数（{}），只给了 {} 个",
                    kind.name,
                    d.params.len(),
                    d.params.join(", "),
                    args.len()
                ),
                kind.span,
            )
            .with_hint(format!(
                "例：`{}({})`",
                kind.name,
                d.params
                    .iter()
                    .map(|p| format!("<{p}>"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
            .with_candidates(d.params.clone()));
        }
        // 用户定义要归约到内置层类型
        let ckind = if kind_lookup(&d.kind).is_some() {
            canonical_kind_name(&d.kind)
        } else if custom_ops.iter().any(|c| keywords::eq_norm(c, &d.kind)) {
            d.kind.clone()
        } else {
            return Err(Diag::error(
                "E002",
                format!("层定义 \"{}\" 的模板层类型 \"{}\" 未定义", kind.name, d.kind),
                kind.span,
            )
            .with_candidates(kind_candidates())
            .with_hint("模板第一条语句的层类型必须是内置层类型"));
        };
        return Ok((ckind, attrs, KindClass::UserDef));
    }
    // 3. 自定义算子直接当层用
    if let Some(c) = custom_ops.iter().find(|c| keywords::eq_norm(c, &kind.name)) {
        let attrs = args
            .iter()
            .enumerate()
            .map(|(i, a)| match a {
                Expr::Named { key, value } => (attr_canonical(&key.name), (**value).clone()),
                other => (format!("arg{i}"), other.clone()),
            })
            .collect();
        return Ok((c.clone(), attrs, KindClass::CustomOp));
    }
    // 4. 报错 + 建议（验收 11）
    let mut cands = kind_candidates();
    for n in defs.keys() {
        cands.push(n.clone());
    }
    for c in custom_ops {
        cands.push(c.clone());
    }
    let near = keywords::suggest_from(&kind.name, &cands.iter().map(|s| s.as_str()).collect::<Vec<_>>(), 6);
    let mut d = Diag::error("E002", format!("未定义的层类型 \"{}\"", kind.name), kind.span);
    if let Some(best) = near.first() {
        d = d.with_hint(format!("是不是想写 \"{best}\"？"));
    }
    // 候选 = 最像的排前面，后面补齐全部已知层类型（第十一部分的 `候选：` 行要求）
    let mut listed: Vec<String> = near.iter().map(|s| s.to_string()).collect();
    for c in &cands {
        if !listed.iter().any(|x| x == c) {
            listed.push(c.clone());
        }
    }
    Err(d.with_candidates(listed))
}

fn positional_only(args: &[Expr]) -> Vec<Expr> {
    args.iter().filter(|a| !a.is_named()).cloned().collect()
}

/// 报错候选：内置层类型的显示名（`Sparse, Embed, Linear, …`）。
pub fn kind_candidates() -> Vec<String> {
    KINDS.iter().map(|k| k.display.to_string()).collect()
}

/// 引用解析：实例名精确匹配 → canonical 层类型匹配（取最后声明）。
fn resolve_ref(id: &Ident, instances: &[Inst]) -> Result<String> {
    if let Some(i) = instances.iter().rev().find(|i| keywords::eq_norm(&i.emitted, &id.name)) {
        return Ok(i.emitted.clone());
    }
    if let Some(k) = kind_lookup(&id.name) {
        if let Some(i) = instances.iter().rev().find(|i| i.kind == k.canonical) {
            return Ok(i.emitted.clone());
        }
    }
    let mut names: Vec<String> = instances.iter().map(|i| i.emitted.clone()).collect();
    names.dedup();
    let near = keywords::suggest_from(&id.name, &names.iter().map(|s| s.as_str()).collect::<Vec<_>>(), 6);
    let mut d = Diag::error(
        "E006",
        format!("连接/前向引用了未定义的层 \"{}\"", id.name),
        id.span,
    );
    if let Some(best) = near.first() {
        d = d.with_hint(format!("是不是想写 \"{best}\"？"));
    }
    Err(d.with_candidates(names))
}

/// 输出名：隐式名（`name == kind` 原文）→ canonical 层类型名；显式名保留原文。
fn emitted_name(name: &Ident, kind: &Ident, ckind: &str, used: &mut HashMap<String, usize>) -> String {
    let implicit = keywords::eq_norm(&name.name, &kind.name);
    let base = if implicit { ckind.to_string() } else { name.name.clone() };
    let n = used.entry(base.clone()).or_insert(0);
    *n += 1;
    if *n == 1 {
        base
    } else {
        format!("{base}#{n}")
    }
}

/// 内置层类型 → canonical 名（未知名原样返回）。
pub fn canonical_kind_name(kind: &str) -> String {
    kind_lookup(kind)
        .map(|k| k.canonical.to_string())
        .unwrap_or_else(|| kind.to_string())
}
