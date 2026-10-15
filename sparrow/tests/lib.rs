//! Sparrow V0.1.4 验收：`--backend lib`（`.spar → Rust AOT 源`）
//!
//! 这里测**不需要 cargo 编译**的部分：生成文本的结构、同源约定、明确拒绝。
//! 真正的"与 `.vmb` 后端**逐字符相同**"在 `tools/lib_ab.ps1` 里验
//! （生成 crate → `cargo build` → 直接跑产物 → 与引擎输出逐字符比对）。
//!
//! 判据为什么能成立：生成的 Rust 直接 `use vmlm_ops::kernels::*` 调**同一个函数**。
//! 朴素循环版本实测差 1 ULP，过不了"逐字符"，所以必须同源 —— 这条约束写在
//! `libgen` 的模块注释里，也有测试守着（生成的代码必须出现 `kernels::` 调用）。

use sparrow::vmlgen::{self, GenOptions};

fn gen(src: &str) -> vmlgen::Generated {
    let f = sparrow::diag::SourceFile::new("t.sparrow", src.to_string());
    sparrow::compile_vml_backend_with(&f, "t.sparrow", GenOptions::default())
        .unwrap_or_else(|d| panic!("{}", d.render(&f)))
}

const TINY: &str = "模型 tiny:\n  嵌入 词表=8 维度=4\n  稀疏 维度=4 连接=2\n  线性 输出=4\n  连接 嵌入 稀疏 线性\n  前向 嵌入 线性\n";

#[test]
fn rust_source_calls_the_same_kernels() {
    let g = gen(TINY);
    let rs = sparrow::libgen::generate_rust(&g, r"D:\w").expect("应能生成 Rust 源");
    // 同源约定：必须调 vmlm_ops 的内核，而不是自己写循环
    assert!(rs.contains("use vmlm_ops::kernels;"), "{rs}");
    assert!(rs.contains("kernels::gemm_nt("), "线性层必须调同一个 GEMM");
    assert!(rs.contains("vmlm_ops::interp::print_f32("), "必须用引擎那个截断式打印");
    // 权重走 include_bytes!，且是**绝对路径**（生成的 crate 可能在任何地方）
    assert!(rs.contains("include_bytes!(\"D:/w/"), "{rs}");
    assert!(rs.contains("LINEAR_W.bin"), "{rs}");
    // 一个层一个 struct + impl
    assert!(rs.contains("struct L0 {") && rs.contains("impl L0 {"), "{rs}");
    assert!(rs.contains("struct Model {") && rs.contains("impl Model {"), "{rs}");
    // 稀疏层按 interp.rs 的同一条标量规则（首项赋值、其余累加）
    assert!(rs.contains("if t == o0 { acc = term; } else { acc += term; }"), "{rs}");
    // 嵌入层要校验行号（引擎那三条硬失败要在 Rust 侧同样成立）
    assert!(rs.contains("EMB: row index out of range"), "{rs}");
}

#[test]
fn rust_source_is_deterministic_and_has_a_manifest() {
    let g = gen(TINY);
    let a = sparrow::libgen::generate_rust(&g, r"D:\w").unwrap();
    let b = sparrow::libgen::generate_rust(&g, r"D:\w").unwrap();
    assert_eq!(a, b);
    let toml = sparrow::libgen::cargo_toml("D:/czsai/vmlm", "m");
    assert!(toml.contains("vmlm-ops = { path = \"D:/czsai/vmlm/ops\" }"), "{toml}");
    assert!(toml.contains("[workspace]"), "生成在宿主仓库目录里时要声明独立包：{toml}");
}

#[test]
fn attention_goes_through_the_same_kernels() {
    let g = gen("模型 mdl:\n  线性 输入=4 输出=4\n  注意力 维度=4 头=2 序列=4 因果=是\n  连接 线性 注意力\n  前向 线性 注意力\n");
    let rs = sparrow::libgen::generate_rust(&g, r"D:\w").unwrap();
    // NN 布局用 kernels::gemm（不是 gemm_nt），softmax 带 causal
    assert!(rs.contains("kernels::gemm("), "QK^T 与 ·V 是 NN 布局");
    assert!(rs.contains("kernels::softmax(sc.as_mut_ptr(), rows, rows, true)"), "{rs}");
    // 序列进了 IR：输入 [4,4]，rows = 4
    assert!(rs.contains("let rows = 4usize;"), "{rs}");
    assert!(g.plan.t_seq == 4);
}

#[test]
fn merge_is_expanded_in_the_struct_impl_style() {
    let g = gen("模型 mdl:\n  嵌入 词表=8 维度=4\n  t = 稀疏 维度=4 连接=2\n  mg = 线性 输出=4\n  连接 嵌入 t\n  连接 t mg\n  连接 嵌入 mg\n  前向 嵌入 mg\n");
    let rs = sparrow::libgen::generate_rust(&g, r"D:\w").unwrap();
    assert!(rs.contains("merge_add(&[&slots[\"t_out\"], &slots[\"EMBED_out\"]]"), "{rs}");
}

#[test]
fn loop_groups_become_a_for_loop_with_per_instance_consts() {
    let f = sparrow::diag::SourceFile::new(
        "t.sparrow",
        "模型 mdl:\n  嵌入 词表=8 维度=4\n  t = 稀疏 维度=4 连接=2 重复3次\n  线性 输出=4\n  连接 嵌入 t 线性\n  前向 嵌入 线性\n".to_string(),
    );
    let g = sparrow::compile_vml_backend_with(
        &f,
        "t.sparrow",
        GenOptions { loop_repeat: true, ..Default::default() },
    )
    .unwrap();
    // 循环组在 Plan 里有投影（Rust 后端不自己推图）
    assert_eq!(g.plan.groups.len(), 1, "必须有 1 个循环组");
    let grp = &g.plan.groups[0];
    assert_eq!(grp.insts.len(), 3, "3 份实例");
    let rs = sparrow::libgen::generate_rust(&g, r"D:\w").expect("循环组现在应能生成");
    // **一份代码 + N 份常量**：这正是循环化对 Rust 后端的意义（源码长度与 N 无关）
    assert!(rs.contains("struct G0Inst"), "{rs}");
    assert!(rs.contains("for inst in &self.g0.items"), "{rs}");
    assert!(rs.contains("carry"), "进位槽：语义是「上一圈的输出」");
    for k in 1..=3 {
        assert!(rs.contains(&format!("C_t__{k}_W_RAW")), "第 {k} 份权重必须内嵌：{rs}");
    }
    // 循环控制常量（LAYERB 的实现细节）不该内嵌进 Rust
    assert!(!rs.contains("t__1_lstride"), "循环控制常量不该出现在 Rust 里");
    assert!(!rs.contains("t__1_iters"), "循环控制常量不该出现在 Rust 里");
    // 一份 `forward` 代码（循环组），而不是 3 份；组外的层各自一份
    let non_group = g.plan.layers.len() - grp.insts.len();
    assert_eq!(
        rs.matches("fn forward(&self, x: &[f32], rows: usize)").count(),
        1 + non_group,
        "组内 1 份 + 组外每层 1 份"
    );

    // 展开模式不得出现循环（否则 `--loop-repeat` 这个开关就没意义）
    let g2 = sparrow::compile_vml_backend_with(&f, "t.sparrow", GenOptions::default()).unwrap();
    let rs2 = sparrow::libgen::generate_rust(&g2, r"D:\w").unwrap();
    assert!(!rs2.contains("for inst in"), "展开模式不该有循环");
    assert!(rs2.matches("struct L").count() >= 5, "展开模式每层一个 struct");
}

/// `--bench-iters N`（V0.1.5）：生成的 Rust 里多一段计时循环，而**默认输出一个字节
/// 都不变** —— 后者是本项目"两条后端逐字符一致"那条判据的前提：把插入的那一段整块
/// 删掉，剩下的文本必须与默认版**逐字节相同**。
#[test]
fn bench_iters_adds_a_timer_and_leaves_the_default_untouched() {
    let g = gen(TINY);
    let plain = sparrow::libgen::generate_rust(&g, r"D:\w").unwrap();
    assert!(!plain.contains("avg us/iter"), "默认不得生成计时设施");

    let timed = sparrow::libgen::generate_rust_with(
        &g,
        r"D:\w",
        sparrow::libgen::LibOptions { bench_iters: 7 },
    )
    .unwrap();

    // 同格式：与 vmlm 的 bench harness（runtime/src/bench.rs）打同一行文本
    assert!(timed.contains("avg us/iter = {}"), "{timed}");
    assert!(timed.contains("for _ in 0..7usize"), "{timed}");
    // 预热 1 遍 + 计时在打印**之后**（打印不能进计时区）
    let p_print = timed.find("print_f32(").unwrap();
    let p_warm = timed.find("black_box(model.forward(&input))").unwrap();
    assert!(p_print < p_warm, "计时必须在打印之后");

    // 删掉插入块（基准注释行之前的那一行空行也算插入的一部分）后必须与默认版逐字节相同
    let lines: Vec<&str> = timed.lines().collect();
    let idx = lines
        .iter()
        .position(|l| l.contains("// ---- 基准（--bench-iters 7）"))
        .expect("应能找到基准注释行");
    let mut kept: Vec<&str> = lines[..idx].to_vec();
    while kept.last().map_or(false, |l| l.trim().is_empty()) {
        kept.pop();
    }
    let mut out = kept.join("\n");
    out.push_str("\n}\n");
    assert_eq!(out, plain, "去掉计时段之后必须与默认版逐字节相同");
}

/// int8（V0.1.5 ⑥）：线性层的量化权重走 `gemm_nt_int8`，常量是**两个**文件
/// （`_Wq` i8 + `_scale` 每输出行一个 f32）。真正的一致性判据在 `tools/int8_ab.ps1`。
#[test]
fn int8_weights_use_the_quantised_kernel() {
    let f = sparrow::diag::SourceFile::new("t.sparrow", TINY.to_string());
    let g = sparrow::compile_vml_backend_with(
        &f,
        "t.sparrow",
        GenOptions { weight_dtype: vmlgen::DataType::Int8, ..Default::default() },
    )
    .unwrap();
    // 常量表：线性层没有 `_W`，而是 `_Wq` + `_scale`
    let names: Vec<&str> = g.plan.consts.iter().map(|c| c.name.as_str()).collect();
    assert!(names.contains(&"LINEAR_Wq"), "{names:?}");
    assert!(names.contains(&"LINEAR_scale"), "{names:?}");
    assert!(!names.contains(&"LINEAR_W"), "int8 时不该再有 f32 权重：{names:?}");
    let rs = sparrow::libgen::generate_rust(&g, r"D:\w").unwrap();
    assert!(rs.contains("kernels::gemm_nt_int8("), "{rs}");
    assert!(rs.contains("wq: Vec<i8>") && rs.contains("scale: Vec<f32>"), "{rs}");
    assert!(rs.contains("i8s(C_LINEAR_Wq_RAW)"), "int8 权重要按字节解释成 i8：{rs}");
    // 嵌入表不吃 int8（回落 f32）
    assert!(rs.contains("e: Vec<f32>"), "嵌入表必须是 f32：{rs}");
    // `.vmb` 文本里不再有 GEMM（那条路没有反量化指令，main.rs 会先拒）
    assert!(!g.vml.contains("GEMM    LINEAR_out"), "int8 不该发射 GEMM：{}", g.vml);
}

#[test]
fn plan_matches_the_text_backend() {
    // `Plan` 必须就是文本后端用的那张图（不能是"第二份真相"）
    let g = gen(TINY);
    let p = &g.plan;
    assert_eq!(p.consts.len(), g.consts.len());
    assert_eq!(p.layers.len(), p.slot_base.len());
    assert_eq!(p.t_seq, 1);
    // 拓扑序覆盖所有层
    let mut seen = vec![false; p.layers.len()];
    for &i in &p.order {
        seen[i] = true;
    }
    assert!(seen.iter().all(|x| *x), "拓扑序必须覆盖所有层");
    // 出口层在 plan 里
    assert_eq!(p.slot_base[p.out_idx], "LINEAR");
    // 每层的槽基名与文本后端一致（展开、无复用时 == 层名）
    for (i, l) in p.layers.iter().enumerate() {
        assert_eq!(p.slot_base[i], l.base);
    }
    // bf16 也走同一条路：常量 dtype 进 plan
    let f = sparrow::diag::SourceFile::new("t.sparrow", TINY.to_string());
    let g2 = sparrow::compile_vml_backend_with(
        &f,
        "t.sparrow",
        GenOptions { weight_dtype: vmlgen::DataType::Bf16, ..Default::default() },
    )
    .unwrap();
    assert!(g2.plan.consts.iter().any(|c| c.dtype == vmlgen::DataType::Bf16));
    let rs = sparrow::libgen::generate_rust(&g2, r"D:\w").unwrap();
    assert!(rs.contains("bf16s("), "bf16 权重要走加宽读法");
}
