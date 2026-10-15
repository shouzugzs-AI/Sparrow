//! Sparrow V0.1.3 验收：汇入（一个层有多个前驱 → EWADD / CONCAT）
//!
//! | 验收 | 在哪 |
//! |---|---|
//! | 相加（等宽）/ 拼接（不等宽）/ 3 前驱折叠 | 这里（指令文本）+ `tools/residual_ab.ps1`（数值） |
//! | 与独立 numpy 参考**逐字符**相同 | `tools/residual_ab.ps1` |
//! | 相加可交换 / 拼接顺序敏感（can-fail） | `tools/residual_ab.ps1` |
//! | 不等宽相加 / 声明宽度冲突 / 汇入+循环 / 成环 | 这里 |

use sparrow::vmlgen::{self, GenOptions, MergeMode};

fn gen(src: &str) -> vmlgen::Generated {
    let f = sparrow::diag::SourceFile::new("t.sparrow", src.to_string());
    sparrow::compile_vml_backend_with(&f, "t.sparrow", GenOptions::default())
        .unwrap_or_else(|d| panic!("{}", d.render(&f)))
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

/// 相加：等宽 → `SHAPE x_in, 1, D` + `EWADD x_in, p0, p1`，然后才是这一层自己的 op。
#[test]
fn equal_width_merges_with_ewadd() {
    let g = gen(
        "模型 mdl:\n  嵌入 词表=8 维度=4\n  t = 稀疏 维度=4 连接=2\n  mg = 线性 输出=4\n  连接 嵌入 t\n  连接 t mg\n  连接 嵌入 mg\n  前向 嵌入 mg\n",
    );
    let body = ins(&g.vml).join("\n");
    assert!(body.contains("SHAPE   mg_in, 1, 4"), "{body}");
    assert!(body.contains("EWADD   mg_in, t_out, EMBED_out"), "{body}");
    assert!(body.contains("GEMM    mg_out, mg_in, mg_W"), "{body}");
    assert!(!body.contains("CONCAT"), "等宽不该拼接");
    // 汇入**先算**（在任何写到 mg_out 的 op 之前）
    let ew = body.find("EWADD   mg_in").unwrap();
    let gemm = body.find("GEMM    mg_out").unwrap();
    assert!(ew < gemm);
}

/// 不等宽（或显式 `合并=拼接`）→ `CONCAT`，宽度是各前驱之和。
#[test]
fn widening_merges_with_concat_and_infers_total_width() {
    let g = gen(
        "模型 mdl:\n  嵌入 词表=8 维度=4\n  t = 稀疏 维度=4 连接=2\n  mg = 线性 输出=4 合并=拼接\n  连接 嵌入 t\n  连接 t mg\n  连接 嵌入 mg\n  前向 嵌入 mg\n",
    );
    let body = ins(&g.vml).join("\n");
    assert!(body.contains("SHAPE   mg_in, 1, 8"), "拼接后宽度 4+4=8：\n{body}");
    assert!(body.contains("CONCAT  mg_in, t_out, EMBED_out"), "{body}");
    // 权重形状跟着汇入宽度走：mg_W 是 [4, 8]
    assert!(
        g.consts.iter().any(|c| c.name == "mg_W" && c.rows == 4 && c.cols == 8),
        "{:?}",
        g.consts
    );
}

/// 前驱顺序按 `连接` 语句**出现顺序**（可预期、确定）。
#[test]
fn predecessor_order_follows_connect_order() {
    let a = gen(
        "模型 mdl:\n  嵌入 词表=8 维度=4\n  t = 稀疏 维度=4 连接=2\n  mg = 线性 输出=4\n  连接 嵌入 t\n  连接 t mg\n  连接 嵌入 mg\n  前向 嵌入 mg\n",
    );
    let b = gen(
        "模型 mdl:\n  嵌入 词表=8 维度=4\n  t = 稀疏 维度=4 连接=2\n  mg = 线性 输出=4\n  连接 嵌入 t\n  连接 嵌入 mg\n  连接 t mg\n  前向 嵌入 mg\n",
    );
    assert!(ins(&a.vml).join("\n").contains("EWADD   mg_in, t_out, EMBED_out"));
    // 提醒：相加可交换，所以 B 里谁在前只影响**槽序**，不影响数值
    assert!(ins(&b.vml).join("\n").contains("EWADD   mg_in, EMBED_out, t_out"));
    // 拼接则顺序进结果 ⇒ 前端必须原样保留顺序（工具脚本里用引擎验数值）
    let c = gen(
        "模型 mdl:\n  嵌入 词表=8 维度=4\n  t = 稀疏 维度=4 连接=2\n  mg = 线性 输出=4 合并=拼接\n  连接 嵌入 t\n  连接 嵌入 mg\n  连接 t mg\n  前向 嵌入 mg\n",
    );
    assert!(ins(&c.vml).join("\n").contains("CONCAT  mg_in, EMBED_out, t_out"));
}

/// 3 个前驱：左折叠（2 个 2 个地合），相加可以就地折叠。
#[test]
fn three_predecessors_fold_left() {
    let g = gen(
        "模型 mdl:\n  嵌入 词表=8 维度=4\n  t = 稀疏 维度=4 连接=2\n  lb = 线性 输出=4\n  ld = 线性 输出=4\n  连接 嵌入 t\n  连接 t lb\n  连接 嵌入 ld\n  连接 lb ld\n  连接 t ld\n  前向 嵌入 ld\n",
    );
    let body = ins(&g.vml).join("\n");
    assert!(body.contains("EWADD   ld_in, EMBED_out, lb_out"), "{body}");
    assert!(body.contains("EWADD   ld_in, ld_in, t_out"), "就地折叠：\n{body}");
    // 折叠用**同一个** ld_in（引擎逐元素先读两个再写同下标 ⇒ dst==a 安全）
    assert_eq!(body.matches("EWADD   ld_in").count(), 2);
    assert_eq!(g.loop_groups.len(), 0);
}

/// 拼接 3 个前驱：宽度在长大，所以每次折叠都要临时槽 + COPY 搬回来。
#[test]
fn three_way_concat_uses_temp_slots() {
    let g = gen(
        "模型 mdl:\n  嵌入 词表=8 维度=4\n  t = 稀疏 维度=4 连接=2\n  lb = 线性 输出=4\n  ld = 线性 输出=4 合并=拼接\n  连接 嵌入 t\n  连接 t lb\n  连接 嵌入 ld\n  连接 lb ld\n  连接 t ld\n  前向 嵌入 ld\n",
    );
    let body = ins(&g.vml).join("\n");
    assert!(body.contains("SHAPE   ld_in, 1, 12"), "4+4+4=12：\n{body}");
    assert!(body.contains("CONCAT  ld_in, EMBED_out, lb_out"), "{body}");
    assert!(body.contains("SHAPE   ld_in_f1, 1, 12"), "{body}");
    assert!(body.contains("CONCAT  ld_in_f1, ld_in, t_out"), "{body}");
    assert!(body.contains("COPY    ld_in, ld_in_f1"), "{body}");
}

/// 图与维度的四种硬失败（都要有名字）。
#[test]
fn merge_errors_are_named() {
    // 不等宽 + 显式相加
    let (c, m) = gen_err(
        "模型 mdl:\n  嵌入 词表=8 维度=4\n  t = 稀疏 维度=4 连接=2\n  w8 = 线性 输出=8\n  mg = 线性 输出=4 合并=相加\n  连接 嵌入 t\n  连接 t w8\n  连接 嵌入 mg\n  连接 w8 mg\n  前向 嵌入 mg\n",
    );
    assert_eq!(c, "E100");
    assert!(m.contains("相加"), "{m}");
    // 这一层自己声明的 `输入=` 与汇入宽度冲突
    let (c, m) = gen_err(
        "模型 mdl:\n  嵌入 词表=8 维度=4\n  t = 稀疏 维度=4 连接=2\n  mg = 线性 输入=4 输出=4 合并=拼接\n  连接 嵌入 t\n  连接 嵌入 mg\n  连接 t mg\n  前向 嵌入 mg\n",
    );
    assert_eq!(c, "E100");
    assert!(m.contains("汇入后是 8"), "{m}");
    // `合并=` 的值不认识
    let (c, m) = gen_err(
        "模型 mdl:\n  嵌入 词表=8 维度=4\n  t = 稀疏 维度=4 连接=2\n  mg = 线性 输出=4 合并=混合\n  连接 嵌入 t\n  连接 嵌入 mg\n  连接 t mg\n  前向 嵌入 mg\n",
    );
    assert_eq!(c, "E100");
    assert!(m.contains("混合"), "{m}");
    // 成环
    let (c, m) = gen_err(
        "模型 mdl:\n  嵌入 词表=8 维度=4\n  t = 稀疏 维度=4 连接=2\n  连接 嵌入 t\n  连接 t 嵌入\n  前向 嵌入 t\n",
    );
    assert_eq!(c, "E102");
    assert!(m.contains("环") || m.contains("链首"), "{m}");
    // 汇入 + 循环化：V0.1.3 起**支持**（组归一化让组内每份的前驱结构一致），
    // 循环体里先汇入再算这一层。数值等价由 tools/residual_ab.ps1 §7 用引擎验。
    let f = sparrow::diag::SourceFile::new(
        "t.sparrow",
        "模型 mdl:\n  嵌入 词表=8 维度=4\n  aa = 线性 输出=4\n  t = 线性 输出=4 重复3次\n  连接 嵌入 aa\n  连接 aa t\n  连接 嵌入 t\n  前向 嵌入 t\n"
            .to_string(),
    );
    let g = sparrow::compile_vml_backend_with(&f, "t.sparrow", GenOptions { loop_repeat: true, ..Default::default() })
        .expect("组内汇入 + 循环化应可编译");
    assert_eq!(g.loop_groups.len(), 1);
    let body = ins(&g.vml).join("\n");
    // 循环体里两条：汇入（进位槽 + 残差）与这一层自己
    assert!(body.contains("EWADD   t__3__m, t__3__in, EMBED_out"), "{body}");
    assert!(body.contains("GEMM    t__3_out, t__3__m, t__1_W"), "{body}");
    // 播种进位槽的来源必须是**链前驱的输出槽**（不是展开版的汇入槽 `<name>_in`）
    assert!(body.contains("COPY    t__3__in, aa_out"), "{body}");
    assert!(!body.contains("t__1_in"), "循环版不该引用展开版的汇入槽：\n{body}");
    // 标签必须在 LOOP_CNT **指令**之前（向后跳）—— 注意别在注释里找 `LOOP_CNT`：
    // `.vml` 头部的常量说明里也写了这个词
    assert!(g.vml.find("t__3__l1:").unwrap() < g.vml.find("LOOP_CNT t__3__ctr").unwrap());
}

/// 汇入**不影响**没有多前驱的模型：默认路径字节不变。
#[test]
fn no_merge_means_no_change() {
    let g = gen("模型 mdl:\n  嵌入 词表=8 维度=4\n  t = 稀疏 维度=4 连接=2\n  连接 嵌入 t\n  前向 嵌入 t\n");
    assert!(!g.vml.contains("EWADD") && !g.vml.contains("CONCAT"));
    assert_eq!(ins(&g.vml).len(), 7, "SHAPE+EMB+SHAPE+SPARSE+TRANSPOSE+PRINT+HALT");
    assert_eq!(MergeMode::Add.name(), "相加(EWADD)");
    assert_eq!(MergeMode::Concat.name(), "拼接(CONCAT)");
}
