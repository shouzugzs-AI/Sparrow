//! Sparrow V0.1.3 验收：注意力块（`注意力 维度=D 头=H [序列=T] [因果=是]`）
//!
//! 汇编器里**没有** `ATTEN`/`HEAD` 这类助记符，所以整块是我们自己用
//! `GEMM/SLICE/TRANSPOSE/SCALE/SOFTMAX/CONCAT` 拼的。这里的判据分两类：
//!
//! * **指令结构**：拆头数、每头一条 TRANSPOSE/SOFTMAX、causal 只出现在 SOFTMAX 上、
//!   拼接宽度、缩放常量的值 = 1/√hd；
//! * **语义边界**：整除性 / 缺头数 / 真假词 / 序列>1 的链首限制。
//!
//! 数值等价（`注意力(T=1)` 恒等于三层线性串联、因果掩码第 0 行等于 T=1）
//! 在 `tools/attn_ab.ps1` 里用真引擎验，且是**逐字符**比较。

use sparrow::vmlgen::{self, DataType, GenOptions};

fn gen_opts(src: &str, opts: GenOptions) -> vmlgen::Generated {
    let f = sparrow::diag::SourceFile::new("t.sparrow", src.to_string());
    sparrow::compile_vml_backend_with(&f, "t.sparrow", opts)
        .unwrap_or_else(|d| panic!("{}", d.render(&f)))
}

fn gen(src: &str) -> vmlgen::Generated {
    gen_opts(src, GenOptions::default())
}

fn gen_err(src: &str) -> (String, String) {
    let f = sparrow::diag::SourceFile::new("t.sparrow", src.to_string());
    match sparrow::compile_vml_backend_with(&f, "t.sparrow", GenOptions::default()) {
        Ok(_) => panic!("本该报错：\n{src}"),
        Err(d) => (d.code.to_string(), d.message.clone()),
    }
}

fn ins(vml: &str) -> Vec<String> {
    vml.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with(';') && !l.starts_with('.') && !l.ends_with(':'))
        .map(|l| l.to_string())
        .collect()
}

fn count(ins: &[String], m: &str) -> usize {
    ins.iter().filter(|l| l.split_whitespace().next() == Some(m)).count()
}

/// 模型：线性 → 注意力(D=8, H=2)
fn attn_src(extra: &str) -> String {
    format!(
        "模型 mdl:\n  线性 输入=8 输出=8\n  注意力 维度=8 头=2{extra}\n  连接 线性 注意力\n  前向 线性 注意力\n"
    )
}

#[test]
fn attn_expands_into_documented_sequence() {
    let g = gen(&attn_src(""));
    let lines = ins(&g.vml);
    // GEMM：链首线性 1 + QKV 3 + 每头 2 条（QK^T 与 ·V）×2 + 输出投影 1 = 9
    assert_eq!(count(&lines, "GEMM"), 1 + 3 + 2 * 2 + 1, "QKV + 每头两条 + 输出投影 + 链首");
    assert_eq!(count(&lines, "BIAS"), 1 + 3 + 1, "链首 + Q/K/V + 输出投影各一条");
    // 每头：3 条 SLICE（q/k/v）+ 1 条 TRANSPOSE + 1 条 SOFTMAX + 1 条 SCALE
    assert_eq!(count(&lines, "SLICE"), 6, "2 头 × 3（q/k/v）");
    assert_eq!(count(&lines, "TRANSPOSE"), 2, "2 头各一次（不能用 RESHAPE：未实现）");
    assert_eq!(count(&lines, "SOFTMAX"), 2);
    assert_eq!(count(&lines, "SCALE"), 2);
    assert_eq!(count(&lines, "CONCAT"), 1, "2 个头拼一次");
    // 三个投影的偏置都是常量（BIAS 第 2 个操作数），不是 flag
    assert!(lines.iter().any(|l| l == "BIAS    ATTN_q, ATTN_bWq"), "{lines:#?}");
    // 头 0/头 1 的切片偏移是 0 与 hd
    assert!(lines.iter().any(|l| l == "SLICE   ATTN_q0, ATTN_q, 0, 4"), "{lines:#?}");
    assert!(lines.iter().any(|l| l == "SLICE   ATTN_q1, ATTN_q, 4, 4"), "{lines:#?}");
    // NN 布局：QK^T 与 ·V 都必须带 `nn`（NT 会把 N 取成 rows(b)，形状就错了）
    assert_eq!(lines.iter().filter(|l| l.ends_with(", nn")).count(), 4, "2 头 × 2 条 GEMM");
    // 拼接后宽度 = D
    assert!(lines.iter().any(|l| l == "SHAPE   ATTN_ctx, 1, 8"), "{lines:#?}");
    // 缩放常量：值 = 1/sqrt(hd) = 1/sqrt(4) = 0.5，且是前端自己产出的 .bin
    let (name, bytes) = g
        .extra_bins
        .iter()
        .find(|(n, _)| n == "ATTN_scale.bin")
        .expect("缩放常量必须由前端产出");
    let v = f32::from_le_bytes(*bytes);
    assert_eq!(name, "ATTN_scale.bin");
    assert!((v - 0.5).abs() < 1e-6, "1/sqrt(4) = 0.5，实际 {v}");
    assert!(g.consts.iter().any(|c| c.name == "ATTN_scale" && c.rows == 1 && c.cols == 1));
    // 权重可以吃 bf16，偏置不行（引擎装载期会拒）
    let wq = g.consts.iter().find(|c| c.name == "ATTN_Wq").unwrap();
    assert_eq!(wq.rows, 8);
    assert_eq!(wq.cols, 8);
    let bq = g.consts.iter().find(|c| c.name == "ATTN_bWq").unwrap();
    assert_eq!(bq.dtype, DataType::F32, "BIAS 的常量必须是 f32");
}

#[test]
fn causal_goes_on_softmax_only() {
    let g = gen(&attn_src(" 因果=是"));
    let lines = ins(&g.vml);
    let soft: Vec<&String> = lines.iter().filter(|l| l.starts_with("SOFTMAX")).collect();
    assert_eq!(soft.len(), 2);
    assert!(soft.iter().all(|l| l.ends_with(", causal")), "{soft:#?}");
    // `causal` 只能出现在 SOFTMAX 上（它是 SOFTMAX 的裸 flag）
    assert_eq!(lines.iter().filter(|l| l.contains("causal")).count(), 2);
}

#[test]
fn causal_accepts_chinese_and_english_truthy_words() {
    for w in ["因果=是", "因果=真", "因果=1", "掩码=是", "causal=yes", "causal=true"] {
        let g = gen(&attn_src(&format!(" {w}")));
        assert!(g.vml.contains("SOFTMAX ATTN_s0, ATTN_s0, causal"), "{w} 应开启掩码");
    }
    for w in ["因果=否", "因果=不", "因果=0", "causal=no"] {
        let g = gen(&attn_src(&format!(" {w}")));
        assert!(g.vml.contains("SOFTMAX ATTN_s0, ATTN_s0\n"), "{w} 应关掉掩码");
    }
}

#[test]
fn seq_declaration_raises_input_rows() {
    let g = gen(
        "模型 mdl:\n  线性 输入=8 输出=8 序列=4\n  注意力 维度=8 头=2 序列=4\n  连接 线性 注意力\n  前向 线性 注意力\n",
    );
    let inp = g.consts.iter().find(|c| c.name == "mdl_in").unwrap();
    assert_eq!((inp.rows, inp.cols), (4, 8), "序列进了架构 IR：输入激活是 [T, D]");
    // 没有 Q 的那一层时也要能看出因果掩码挂在 SOFTMAX 上
    assert!(g.vml.contains("SOFTMAX ATTN_s0, ATTN_s0\n"));
}

#[test]
fn attn_inside_repeat_gets_per_instance_scale_bins() {
    let g = gen_opts(
        "模型 mdl:\n  线性 输入=4 输出=4\n  a = 注意力 维度=4 头=2 重复2次\n  连接 线性 a\n  前向 线性 a\n",
        GenOptions { loop_repeat: true, ..Default::default() },
    );
    // 每份实例一条缩放常量 + 一条 .bin（循环体的 LAYERB 会走到第 2 份）
    for r in 1..=2 {
        assert!(g.consts.iter().any(|c| c.name == format!("a__{r}_scale")));
        assert!(
            g.extra_bins.iter().any(|(n, _)| n == &format!("a__{r}_scale.bin")),
            "第 {r} 份的缩放 .bin 必须也产出：{:?}",
            g.extra_bins.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>()
        );
    }
    // 常量池里第 2 份紧跟第 1 份（步长恒定）
    assert!(g.loop_groups.len() == 1);
    assert!(g.loop_groups[0].stride > 0);
}

#[test]
fn attn_errors_are_named() {
    let (c, m) = gen_err("模型 mdl:\n  线性 输入=6 输出=6\n  注意力 维度=6 头=4\n  连接 线性 注意力\n  前向 线性 注意力\n");
    assert_eq!(c, "E108");
    assert!(m.contains("整除"), "{m}");
    let (c, m) = gen_err("模型 mdl:\n  线性 输入=4 输出=4\n  注意力 维度=4\n  连接 线性 注意力\n  前向 线性 注意力\n");
    assert_eq!(c, "E105");
    assert!(m.contains("头"), "{m}");
    let (c, m) = gen_err("模型 mdl:\n  线性 输入=4 输出=4\n  注意力 维度=4 头=2 因果=或许\n  连接 线性 注意力\n  前向 线性 注意力\n");
    assert_eq!(c, "E108");
    assert!(m.contains("真假词"), "{m}");
    // V0.1.5：**嵌入链首 + 序列>1 现在是合法的**（逐行查表 + 拼成 [T,D]）。
    // 这条曾经是 E107；现在要能生成，且生成物里要有 T 次 `EMB`。
    let g = gen("模型 mdl:\n  嵌入 词表=8 维度=4 序列=4\n  注意力 维度=4 头=2 序列=4\n  连接 嵌入 注意力\n  前向 嵌入 注意力\n");
    assert_eq!(
        g.vml.matches("    EMB     ").count(),
        4,
        "4 个词元 ⇒ 4 次逐行查表"
    );
    assert!(g.vml.contains("SHAPE   EMBED_tok3, 1, 1"), "第 4 个词元也要取到：{}", g.vml);
    assert!(g.vml.contains("TRANSPOSE EMBED_out,"), "拼完列再转成 [T,D]");
    // 但**中间**的 `嵌入`（输入已经是 [T,D] 激活）仍然要拒绝
    let (c, m) = gen_err(
        "模型 mdl:\n  线性 输入=4 输出=4 序列=4\n  嵌入 词表=8 维度=4\n  连接 线性 嵌入\n  前向 线性 嵌入\n",
    );
    assert_eq!(c, "E107");
    assert!(m.contains("链首"), "{m}");
    // `稀疏` 是向量×矩阵，序列>1 时一次只吃一行 ⇒ 仍拒绝（否则静默只算第一行）
    let (c, m) = gen_err(
        "模型 mdl:\n  线性 输入=4 输出=4 序列=4\n  稀疏 维度=4 连接=2\n  连接 线性 稀疏\n  前向 线性 稀疏\n",
    );
    assert_eq!(c, "E107");
    assert!(m.contains("向量×矩阵"), "{m}");
    // 不写 `维度=` 时，宽度沿用上游（宽容：能推就不要求写）
    let g = gen("模型 mdl:\n  线性 输入=4 输出=4\n  注意力 头=2\n  连接 线性 注意力\n  前向 线性 注意力\n");
    assert!(g.vml.contains("SLICE   ATTN_q0, ATTN_q, 0, 2"), "D 从上游推出来 = 4");
    // 但链首的注意力**推不出来**（没有上游）
    let (c, m) = gen_err(
        "模型 mdl:\n  a = 注意力 头=2\n  b = 线性 输出=4\n  连接 a b\n  前向 a b\n",
    );
    assert_eq!(c, "E103");
    assert!(m.contains("维度"), "{m}");
}

/// 注意力块里的**目的槽**集合（只取注意力那一段，不含链首线性层）。
fn attn_dsts(vml: &str) -> std::collections::BTreeSet<String> {
    let mut it = vml.lines().skip_while(|l| !l.contains("注意力 `"));
    it.next(); // 跳过那条 `; --- 注意力 … ---` 注释本身
    let mut set = std::collections::BTreeSet::new();
    for l in it {
        let t = l.trim();
        if t.starts_with(".end") || t.starts_with("PRINT") {
            break;
        }
        if t.starts_with(';') {
            continue;
        }
        let mut parts = t.split_whitespace();
        let Some(op) = parts.next() else { continue };
        if !matches!(
            op,
            "SHAPE" | "SLICE" | "TRANSPOSE" | "GEMM" | "SCALE" | "SOFTMAX" | "CONCAT" | "COPY"
                | "BIAS" | "EWADD" | "ACT" | "NORM"
        ) {
            continue;
        }
        if let Some(d) = parts.next() {
            set.insert(d.trim_end_matches(',').to_string());
        }
    }
    set
}

/// V0.1.6：`--slot-reuse` 下注意力的**每头暂存槽**按角色共用。
///
/// 判据不是"槽少了几个"这种数字（数字变小可能是别的原因），而是**名字本身**：
/// * `q/k/kt/v/s` 五个角色**各只有一份** —— H 个头轮流借同一批（写完立刻读、读完不再读）；
/// * `c`（每头输出）**必须**按头分开 —— H 份要一起活到拼头那一刻；
/// * 拼头目标借 `_q`（最后一次读它在**最后一个头**的 GEMM，形状同为 [T, D]）。
///
/// can-fail：把 `c` 也改成共用（一行改动）⇒ 输出**立刻变**（实测四个注意力模型的
/// off/on 全不同，见 docs/TRAPS.md G10）。所以"输出逐字符相同"这条判据不是恒真的。
#[test]
fn slot_reuse_shares_per_head_scratch_but_not_head_outputs() {
    let opts = |reuse: bool| GenOptions {
        slot_reuse: reuse,
        ..GenOptions::default()
    };
    let src = attn_src(" 序列=4 因果=是"); // D=8, H=2, T=4 ⇒ hd=4
    let off = gen_opts(&src, opts(false));
    let on = gen_opts(&src, opts(true));
    let a = attn_dsts(&off.vml);
    let b = attn_dsts(&on.vml);
    // 不复用：3 个投影 + 2 头 × 6 个角色 + `_ctx` + `_out` = 17
    assert_eq!(a.len(), 17, "{a:#?}");
    // 复用：3 个投影 + 5 个角色 + 2 个 `c` + `_out` = 11（`_ctx` 借 `_q`）
    assert_eq!(b.len(), 11, "{b:#?}");
    for role in ["hq", "hk", "hkt", "hv", "hs"] {
        let name = format!("ATTN_{role}");
        assert!(b.contains(&name), "缺共用角色槽 {name}：{b:#?}");
        assert!(!a.contains(&name), "不复用版不该出现角色名 {name}：{a:#?}");
    }
    assert!(
        b.contains("ATTN_c0") && b.contains("ATTN_c1"),
        "每头输出槽必须分开：{b:#?}"
    );
    assert!(!b.contains("ATTN_ctx"), "拼头目标借 `_q`，不该再有 `_ctx`：{b:#?}");
    // 不复用版与 V0.1.5 逐字节一致（默认路径不变）
    assert!(off.vml.contains("SLICE   ATTN_q0, ATTN_q, 0, 4"), "{}", off.vml);
    assert!(off.vml.contains("SLICE   ATTN_q1, ATTN_q, 4, 4"), "{}", off.vml);
    assert!(on.vml.contains("SLICE   ATTN_hq, ATTN_q, 0, 4"), "{}", on.vml);
}
