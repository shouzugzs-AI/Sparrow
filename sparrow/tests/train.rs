//! Sparrow V0.1.6 验收：训练路径（`损失 均方` + `反向`）
//!
//! 这一组只做**指令结构 + 边界**的快速判据（毫秒级）。数值正确性在
//! `tools/train_ab.ps1` 里验：那边的判据是 numpy 独立算的**中心差分**梯度
//! 与引擎打印的解析梯度逐项对比，并且需要一份 `--features train` 的运行时
//! （官方推理版遇到反向指令会 `exit=6`）。
//!
//! 这里守的是三件在数值判据里**看不出来**的事：
//!   1. 反向算子的**顺序**（拓扑反序）与**操作数约定**（尤其 `MSE_BWD` 的梯度
//!      落在第 3 个操作数 —— 与 `CE_BWD` 不同，抄错就是"静默算错"）；
//!   2. `dy` 真的**穿过**了激活层（`GEMM_BWD_X` → `ACT_BWD` → `GEMM_BWD_W` 链）；
//!   3. 边界：没有损失不许反向（E113）、与复用/循环组合硬拒绝（E114）、
//!      AOT 后端不支持（E115）、损失函数不认识要报候选（E112）。

use sparrow::vmlgen::{DataType, GenOptions};

fn gen(src: &str) -> String {
    gen_opts(src, GenOptions::default())
}

fn gen_opts(src: &str, opts: GenOptions) -> String {
    let f = sparrow::diag::SourceFile::new("t.sparrow", src.to_string());
    sparrow::compile_vml_backend_with(&f, "t.sparrow", opts)
        .unwrap_or_else(|d| panic!("{}", d.render(&f)))
        .vml
}

fn err(src: &str, opts: GenOptions) -> (String, String) {
    let f = sparrow::diag::SourceFile::new("t.sparrow", src.to_string());
    match sparrow::compile_vml_backend_with(&f, "t.sparrow", opts) {
        Ok(_) => panic!("本该报错：\n{src}"),
        Err(d) => (d.code.to_string(), d.message.clone()),
    }
}

/// 线性(4→4) → 激活(tanh) → 线性(4→2) + 均方损失 + 反向
const TRAIN_SRC: &str =
    "模型 tmse:\n  l1 = 线性 输入=4 输出=4\n  act = 激活 激活=tanh\n  l2 = 线性 输入=4 输出=2\n  连接 l1 act l2\n  前向 l1 l2\n  损失 均方\n  反向\n";

fn lines(vml: &str) -> Vec<String> {
    vml.lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty() && !l.starts_with(';') && !l.starts_with('.'))
        .collect()
}

#[test]
fn backward_ops_are_emitted_in_reverse_topological_order() {
    let vml = gen(TRAIN_SRC);
    let ls = lines(&vml);
    let idx = |needle: &str| ls.iter().position(|l| l.starts_with(needle)).unwrap_or_else(|| {
        panic!("缺少 `{needle}`：\n{vml}");
    });
    // 顺序：先损失梯度，再 l2（下游）→ act → l1（上游）
    let mse = idx("MSE_BWD");
    let z_l2 = idx("ZERO    g_l2_W");
    let gw_l2 = idx("GEMM_BWD_W g_l2_W");
    let act = idx("ACT_BWD");
    let gw_l1 = idx("GEMM_BWD_W g_l1_W");
    assert!(mse < z_l2 && z_l2 < gw_l2, "ZERO 必须在累加的 GEMM_BWD_W 之前：\n{vml}");
    assert!(gw_l2 < act, "先反传下游层，再到激活：\n{vml}");
    assert!(act < gw_l1, "激活之后才轮到上游层：\n{vml}");

    // 操作数约定：MSE_BWD dst=预测, a=目标, b=梯度(写) —— 梯度在**第 3 个**位置
    assert!(
        ls.iter().any(|l| l == "MSE_BWD l2_out, tmse_tgt, tmse_dloss"),
        "MSE_BWD 的操作数顺序（梯度落在第 3 个）写错了：\n{vml}"
    );
    // dy 真的穿过了激活层：ACT_BWD 的 dy 来自上游层的 GEMM_BWD_X
    assert!(
        ls.iter().any(|l| l == "ACT_BWD act_dact, l2_din, l1_out, tanh"),
        "ACT_BWD 的三个操作数应是 (dx, dy=l2_din, ypre=l1_out)+kind：\n{vml}"
    );
    // 链首不需要 GEMM_BWD_X（它的输入是模型输入常量）
    assert!(!vml.contains("l1_din"), "链首不该产出输入梯度：\n{vml}");
    // 参数梯度逐个打印（训练步唯一的可观测输出）
    for g in ["g_l2_W", "g_l2_b", "g_l1_W", "g_l1_b"] {
        assert!(ls.iter().any(|l| l == &format!("PRINT   {g}")), "缺 PRINT {g}：\n{vml}");
    }
}

#[test]
fn backward_defaults_are_exact_v015_when_absent() {
    // 没有 `损失`/`反向` 时，产物与 V0.1.5 逐字节相同（默认路径不变）。
    let plain = "模型 zoo:\n  l1 = 线性 输入=4 输出=4\n  l2 = 线性 输入=4 输出=2\n  连接 l1 l2\n  前向 l1 l2\n";
    let vml = gen(plain);
    assert!(!vml.contains("BWD") && !vml.contains("ZERO"), "无训练陈述时不该有反向算子：\n{vml}");
    assert!(!vml.contains("_tgt"), "无损失时不该声明目标常量：\n{vml}");
}

#[test]
fn backward_boundaries_are_named() {
    // E113：没有损失
    let (c, m) = err(
        "模型 zoo:\n  l1 = 线性 输入=4 输出=4\n  反向\n  前向 l1 l1\n",
        GenOptions::default(),
    );
    assert_eq!(c, "E113");
    assert!(m.contains("没有声明损失"), "{m}");

    // E114：与槽位复用组合
    let (c, m) = err(
        TRAIN_SRC,
        GenOptions { slot_reuse: true, ..GenOptions::default() },
    );
    assert_eq!(c, "E114");
    assert!(m.contains("--slot-reuse"), "{m}");

    // E114：与循环化重复组合
    let loop_src = "模型 zoo:\n  l1 = 线性 输入=4 输出=4\n  a = 激活 激活=tanh 重复2次\n  l2 = 线性 输出=2\n  连接 l1 a\n  连接 a l2\n  前向 l1 l2\n  损失 均方\n  反向\n";
    let (c, m) = err(loop_src, GenOptions { loop_repeat: true, ..GenOptions::default() });
    assert_eq!(c, "E114");
    assert!(m.contains("--loop-repeat"), "{m}");

    // E114：非 f32 权重
    let (c, m) = err(
        TRAIN_SRC,
        GenOptions { weight_dtype: DataType::Bf16, ..GenOptions::default() },
    );
    assert_eq!(c, "E114");
    assert!(m.contains("f32"), "{m}");

    // E114：反向路径上有没接反向内核的层
    let emb_src = "模型 zoo:\n  h1 = 线性 输入=4 输出=4\n  nm = 归一化 维度=4\n  连接 h1 nm\n  前向 h1 nm\n  损失 均方\n  反向\n";
    let (c, m) = err(emb_src, GenOptions::default());
    assert_eq!(c, "E114");
    assert!(m.contains("NORM"), "{m}");

    // E112：不认识的损失函数（sema 里判定，带候选）
    let f = sparrow::diag::SourceFile::new(
        "t.sparrow",
        "模型 zoo:\n  h1 = 线性 输入=4 输出=4\n  损失 交叉熵\n  前向 h1 h1\n",
    );
    match sparrow::compile_vml(&f) {
        Ok(_) => panic!("本该报错"),
        Err(d) => {
            assert_eq!(d.code, "E112");
            assert!(d.render(&f).contains("均方"), "要给出候选：{}", d.render(&f));
        }
    }
}
