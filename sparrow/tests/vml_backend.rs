//! Sparrow V0.1.1 —— opcode 后端回归测试（`.spar` → `.vml`）
//!
//! 这里断言的是**文本契约**（助记符、操作数顺序、槽位名、常量清单形状），
//! 因为引擎侧的逐字节验收在 `tools/e2e_tiny.ps1`（要调真汇编器与真引擎）。
//! 两侧分工：本文件保证"发出去的每一个字符都符合汇编器的契约"，
//! 那个脚本保证"这些字符在真引擎上算出正确的数"。

use sparrow::diag::SourceFile;

fn gen(src: &str) -> sparrow::vmlgen::Generated {
    let sf = SourceFile::new("t.sparrow", src);
    match sparrow::compile_vml_backend(&sf, "t.sparrow") {
        Ok(g) => g,
        Err(d) => panic!("编译失败：\n{}", d.render(&sf)),
    }
}

fn gen_err(src: &str) -> (String, String) {
    let sf = SourceFile::new("t.sparrow", src);
    match sparrow::compile_vml_backend(&sf, "t.sparrow") {
        Ok(_) => panic!("本应报错却成功了：{src}"),
        Err(d) => (d.code.to_string(), d.render(&sf)),
    }
}

fn example(name: &str) -> String {
    std::fs::read_to_string(format!("{}/examples/{name}", env!("CARGO_MANIFEST_DIR")))
        .unwrap_or_else(|e| panic!("读不到 examples/{name}：{e}"))
}

// ---------------------------------------------------------------------------
// 1. e2e_tiny 的完整 opcode 序列（端到端脚本用的就是这份文本）
// ---------------------------------------------------------------------------

#[test]
fn backend_e2e_tiny_sequence() {
    let g = gen(&example("e2e_tiny.sparrow"));
    for line in [
        "SHAPE   EMBED_out, 1, 4",
        "EMB     EMBED_out, EMBED_E, tiny_in",
        // SPARSE_MATVEC 的 dst 必须是列向量（引擎硬契约）⇒ 后面跟 TRANSPOSE
        "SHAPE   SPARSE_col, 4, 1",
        "SPARSE_MATVEC SPARSE_col, SPARSE_off, SPARSE_src, SPARSE_W, EMBED_out",
        "TRANSPOSE SPARSE_out, SPARSE_col",
        "GEMM    LINEAR_out, SPARSE_out, LINEAR_W",
        "BIAS    LINEAR_out, LINEAR_b",
        "PRINT   LINEAR_out",
        "HALT",
    ] {
        assert!(g.vml.contains(line), "生成的 .vml 缺少 `{line}`：\n{}", g.vml);
    }
    // 不能出现未实现的指令（汇编器 UNIMPLEMENTED_OPS 会硬失败）
    for bad in ["RESHAPE", "LOADK", "MEAN", "GATHER", "SCATTER", "BRANCH", "OP_CALL", "NORM "] {
        assert!(!g.vml.contains(bad), "用了未实现的指令 {bad}：\n{}", g.vml);
    }
    // 常量清单形状（决定 .bin 大小，写错就是静默错）
    for want in [
        "tiny_in:tiny_in.bin:1:1",
        "EMBED_E:EMBED_E.bin:8:4",
        "SPARSE_off:SPARSE_off.bin:5:1",
        "SPARSE_src:SPARSE_src.bin:8:1",
        "SPARSE_W:SPARSE_W.bin:8:1",
        "LINEAR_W:LINEAR_W.bin:4:4",
        "LINEAR_b:LINEAR_b.bin:1:4",
    ] {
        assert!(g.args_text.contains(want), "权重清单缺 `{want}`：\n{}", g.args_text);
    }
    assert_eq!(g.consts.len(), 7);
}

#[test]
fn backend_is_deterministic() {
    let a = gen(&example("e2e_tiny.sparrow"));
    let b = gen(&example("e2e_tiny.sparrow"));
    assert_eq!(a.vml, b.vml);
    assert_eq!(a.args_text, b.args_text);
}

// ---------------------------------------------------------------------------
// 2. 槽位分配：朴素、不复用、每个实例独占一个输出槽
// ---------------------------------------------------------------------------

#[test]
fn slots_are_naive_and_disjoint() {
    let g = gen(&example("e2e_tiny.sparrow"));
    assert_eq!(g.slots, vec!["EMBED_out", "SPARSE_out", "LINEAR_out"]);
    // 不复用：每条写指令的 dst 都不同（除 BIAS 就地累加）
    let writes: Vec<&str> = g
        .vml
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            t.starts_with("EMB ") || t.starts_with("SPARSE_MATVEC") || t.starts_with("GEMM ")
                || t.starts_with("TRANSPOSE") || t.starts_with("LN ") || t.starts_with("ACT ")
        })
        .collect();
    assert_eq!(writes.len(), 4, "{:?}", writes);
}

#[test]
fn repeat_is_unrolled_with_distinct_slots() {
    let src = "模型 mdl:\n  嵌入 词表=8 维度=4\n  t = 稀疏 维度=4 连接=2 重复3次\n  线性 输出=4\n  连接 嵌入 t 线性\n  前向 嵌入 线性\n";
    let g = gen(src);
    assert_eq!(g.vml.matches("SPARSE_MATVEC").count(), 3, "{}", g.vml);
    assert!(g.vml.contains("t__1") && g.vml.contains("t__2") && g.vml.contains("t__3"), "{}", g.vml);
    // 24 层也不产生循环指令（V0.1.1 是展开；LOOP_CNT 排在 V0.1.2）
    let src24 = "模型 mdl:\n  嵌入 词表=8 维度=4\n  t = 稀疏 维度=4 连接=2 重复24次\n  线性 输出=4\n  连接 嵌入 t 线性\n  前向 嵌入 线性\n";
    let g24 = gen(src24);
    assert_eq!(g24.vml.matches("SPARSE_MATVEC").count(), 24);
    assert!(!g24.vml.contains("LOOP_CNT"));
    // 模型输入 1 + 嵌入表 1 + 24 份各自的 off/src/W + 线性 W/b 2
    assert_eq!(g24.consts.len(), 1 + 1 + 3 * 24 + 2, "每份重复各有自己的 off/src/W");
}

// ---------------------------------------------------------------------------
// 3. 名字安全：中文实例名换成 ASCII，避开 st_ / g_ 前缀
// ---------------------------------------------------------------------------

#[test]
fn chinese_instance_names_become_ascii() {
    let g = gen(&example("e2e_tiny.sparrow"));
    assert!(g.vml.contains("EMBED_out"), "{}", g.vml);
    // 指令行必须全 ASCII（符号表/命令行按字节处理）；`;` 注释行可以留中文
    // —— 汇编器在 parse 之前就把 `;` 之后的内容剥掉了。
    for line in g.vml.lines().map(|l| l.split(';').next().unwrap_or("")) {
        assert!(
            line.is_ascii(),
            "指令行/槽位名必须全 ASCII，实际：`{line}`\n{}",
            g.vml
        );
    }
    // `st_` / `g_` 是汇编器的保留前缀（持久状态 / 梯度槽），必须避开
    let src = "模型 mdl:\n  st_emb = 嵌入 词表=8 维度=4\n  g_out = 线性 输出=4\n  连接 st_emb g_out\n  前向 st_emb g_out\n";
    let g2 = gen(src);
    assert!(!g2.vml.contains("st_emb_out"), "{}", g2.vml);
    assert!(!g2.vml.contains("g_out_out"), "{}", g2.vml);
}

// ---------------------------------------------------------------------------
// 4. 不支持的构造要报清楚（任务书第六部分：不碰引擎/汇编器，能力边界要显式）
// ---------------------------------------------------------------------------

#[test]
fn unsupported_constructs_are_named() {
    // 裸指令（Level 6）
    let (code, r) = gen_err(&example("level6.sparrow"));
    assert_eq!(code, "E101");
    assert!(r.contains("裸指令"), "{r}");
    assert!(r.contains("V0.2"), "{r}");
    // 自定义算子（Level 5）
    let (code, _) = gen_err(&example("level5.sparrow"));
    assert_eq!(code, "E101");
    // 注意力现在能编了（V0.1.3）；剩下的"不支持"是自定义算子/裸指令
    let g = gen("模型 mdl:\n  线性 输入=8 输出=8\n  注意力 维度=8 头=2\n  连接 线性 注意力\n  前向 线性 注意力\n");
    assert!(g.vml.contains("SOFTMAX"), "注意力应展开成 SOFTMAX 等一整套");
    // 但注意力自己的参数错必须报出来
    let (code, r) = gen_err("模型 mdl:\n  线性 输入=6 输出=6\n  注意力 维度=6 头=4\n  连接 线性 注意力\n  前向 线性 注意力\n");
    assert_eq!(code, "E108");
    assert!(r.contains("整除"), "{r}");
}

#[test]
fn graph_errors_are_named() {
    // 不等宽还想相加（V0.1.3 起"两个前驱"本身是合法的**汇入**，不再是错误）
    let (code, r) = gen_err(
        "模型 mdl:\n  嵌入 词表=8 维度=4\n  t = 稀疏 维度=4 连接=2\n  w8 = 线性 输出=8\n  mg = 线性 输出=4 合并=相加\n  连接 嵌入 t\n  连接 t w8\n  连接 嵌入 mg\n  连接 w8 mg\n  前向 嵌入 mg\n",
    );
    assert_eq!(code, "E100");
    assert!(r.contains("相加"), "{r}");
    // 有连接但没覆盖全部层 ⇒ 两个链首
    let (code, _) = gen_err(
        "模型 mdl:\n  嵌入 词表=8 维度=4\n  稀疏 维度=4 连接=2\n  线性 输出=4\n  连接 嵌入 稀疏\n",
    );
    assert_eq!(code, "E102");
    // 线性层的输入维度推不出来（链首且没写 in）
    let (code, r) = gen_err("模型 mdl:\n  线性 输出=4\n");
    assert_eq!(code, "E103");
    assert!(r.contains("输入维度"), "{r}");
}

// ---------------------------------------------------------------------------
// 5. 无 `连接` 时按声明序成链（Level 0/1 没有连接语句）
// ---------------------------------------------------------------------------

#[test]
fn implicit_chain_by_declaration_order() {
    let g = gen("模型 mdl:\n  嵌入 词表=8 维度=4\n  稀疏 维度=4 连接=2\n  线性 输出=4\n");
    assert!(g.vml.contains("SPARSE_MATVEC"), "{}", g.vml);
    assert!(g.vml.contains("GEMM    LINEAR_out, SPARSE_out, LINEAR_W"), "{}", g.vml);
    // 链首是嵌入时，输入常量是 [1,1] 的词元 id
    assert!(g.args_text.contains("mdl_in:mdl_in.bin:1:1"), "{}", g.args_text);
}

// ---------------------------------------------------------------------------
// 6. 归一化 / 激活 的展开
// ---------------------------------------------------------------------------

#[test]
fn norm_and_act_expansion() {
    let g = gen(
        "模型 mdl:\n  嵌入 词表=8 维度=4\n  归一化 维度=4\n  激活 act=relu\n  线性 输出=4\n  连接 嵌入 归一化 激活 线性\n  前向 嵌入 线性\n",
    );
    assert!(g.vml.contains("LN      NORM_out, EMBED_out, NORM_g, NORM_b"), "{}", g.vml);
    assert!(g.vml.contains("ACT     ACT_out, NORM_out, relu"), "{}", g.vml);
    assert!(g.args_text.contains("NORM_g:NORM_g.bin:1:4"));
    assert!(g.args_text.contains("NORM_b:NORM_b.bin:1:4"));
}
