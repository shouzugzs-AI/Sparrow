//! Sparrow V0.1.2 P0-2 验收：`重复` 的循环化（`LOOP_CNT` + `LAYERB`）
//!
//! 这里只测**不需要引擎/汇编器**的性质（默认 `cargo test` 就能跑）；
//! 需要真引擎/真汇编器的部分在 `tools/loop_ab.ps1`：
//!
//! | 验收 | 在哪 |
//! |---|---|
//! | 1. O(1) 指令流：24 层与 6 层指令数相同 | `instruction_stream_is_independent_of_n` |
//! | 2. 常量池不变（循环版 .args 以展开版为前缀） | `args_prefix_matches_unrolled` |
//! | 3. 引擎输出逐字节相同（6 层，24 层展开版装不出来） | `tools/loop_ab.ps1` §3 |
//! | 4. can-fail：步长 ±64/±192 输出必须变 | `tools/loop_ab.ps1` §4 |
//! | 5. can-fail：池不连续必须报 E210 | `verify_rejects_inconsistent_pool` + `tools/loop_ab.ps1` §5 |
//! | 6. 默认（不加 `--loop-repeat`）字节不变 | `default_stays_unrolled` |

use sparrow::vmlgen::{self, GenOptions};
use sparrow::vmbmeta::{align_up, read_const_table, CONST_ALIGN};

fn gen_with(src: &str, loop_repeat: bool) -> vmlgen::Generated {
    let f = sparrow::diag::SourceFile::new("t.sparrow", src.to_string());
    sparrow::compile_vml_backend_with(&f, "t.sparrow", GenOptions { loop_repeat, ..Default::default() })
        .unwrap_or_else(|d| panic!("{}", d.render(&f)))
}

/// 期待一个**命名诊断**（不是 panic）。
fn err_of(src: &str, loop_repeat: bool) -> (String, String) {
    let f = sparrow::diag::SourceFile::new("t.sparrow", src.to_string());
    match sparrow::compile_vml_backend_with(&f, "t.sparrow", GenOptions { loop_repeat, ..Default::default() }) {
        Ok(_) => panic!("应当报错"),
        Err(d) => (d.code.to_string(), d.message.clone()),
    }
}

fn model(n: usize) -> String {
    // 注意 `模型 m:` 会撞 E007（`m` 是 model 的保留别名）—— 用 `mdl`
    format!(
        "模型 mdl:\n  嵌入 词表=8 维度=4\n  t = 稀疏 维度=4 连接=2 重复{n}次\n  线性 输出=4\n  连接 嵌入 t 线性\n  前向 嵌入 线性\n"
    )
}

/// 只看指令行（去掉注释、标签、空行、`.section`/`.end`）。
fn ins_lines(vml: &str) -> Vec<String> {
    vml.lines()
        .map(|l| l.trim())
        .filter(|l| {
            !l.is_empty()
                && !l.starts_with(';')
                && !l.starts_with('.')
                && !l.ends_with(':')
        })
        .map(|l| l.to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// 验收 1：指令流与 N 无关
// ---------------------------------------------------------------------------

#[test]
fn instruction_stream_is_independent_of_n() {
    let g6 = gen_with(&model(6), true);
    let g24 = gen_with(&model(24), true);
    let g1 = gen_with(&model(1), true);

    let n6 = ins_lines(&g6.vml).len();
    let n24 = ins_lines(&g24.vml).len();
    assert_eq!(
        n6, n24,
        "24 层与 6 层的指令数必须相同（循环发射与 N 无关）；6 层={n6} 24 层={n24}\n{}",
        g24.vml
    );
    // 模板只有一份：SPARSE_MATVEC 只出现一次，而不是 24 次
    // （只数**指令行**：注释里的 `LOOP_CNT`/`LAYERB` 字样不算）
    let ins = ins_lines(&g24.vml);
    let count = |m: &str| {
        ins.iter()
            .filter(|l| l.split_whitespace().next() == Some(m))
            .count()
    };
    assert_eq!(count("SPARSE_MATVEC"), 1, "模板只有一份");
    assert_eq!(count("LOOP_CNT"), 1);
    assert_eq!(count("LAYERB"), 2, "推进一次 + 还原一次");
    assert_eq!(count("SHAPE"), 6, "嵌入1 + 进位1 + 计数器1 + 步长2 + 模板内1");
    assert_eq!(count("COPY"), 5, "进位播种 + 计数器/步长×2 + 循环内进位");

    // N=1 不发射循环（没有收益，且会让 1 层模型白白多 9 条）
    assert!(!g1.vml.contains("LOOP_CNT"), "1 层不该走循环");
    assert_eq!(ins_lines(&g1.vml).len(), 9, "1 层 = 嵌入2 + 稀疏3 + 线性2 + PRINT/HALT");
}

/// 展开版是 O(N)：每层 3 条。这条与上一条互为对照（否则"指令数相同"可能是都错）。
#[test]
fn unrolled_grows_with_n() {
    let u6 = ins_lines(&gen_with(&model(6), false).vml).len();
    let u24 = ins_lines(&gen_with(&model(24), false).vml).len();
    assert_eq!(u24 - u6, 18 * 3, "展开版每层多 3 条（24-6=18 层）");
    assert!(u6 > 0);
}

// ---------------------------------------------------------------------------
// 验收 2：常量池不变
// ---------------------------------------------------------------------------

#[test]
fn args_prefix_matches_unrolled() {
    let l = gen_with(&model(24), true);
    let u = gen_with(&model(24), false);
    let la: Vec<&str> = l.args_text.lines().collect();
    let ua: Vec<&str> = u.args_text.lines().collect();
    assert_eq!(la.len(), ua.len() + 3, "多出的 3 条是循环控制常量");
    assert_eq!(&la[..ua.len()], &ua[..], "前 {len} 行必须逐字节相同", len = ua.len());
    // 循环控制常量必须在**池尾**：中间插一个就破坏了"相邻两份偏移差 = 步长"
    assert!(la[ua.len()].starts_with("t__1_iters:"));
    assert!(la[ua.len() + 1].starts_with("t__1_lstride:"));
    assert!(la[ua.len() + 2].starts_with("t__1_nstride:"));
    // 池里 24 份常量齐全（LAYERB 要走到第 24 份）
    for r in 1..=24 {
        assert!(l.consts.iter().any(|c| c.name == format!("t__{r}_W")));
    }
}

// ---------------------------------------------------------------------------
// 验收 6（反面）：默认仍是展开，且与 V0.1.1 语义一致
// ---------------------------------------------------------------------------

#[test]
fn default_stays_unrolled() {
    let g = gen_with(&model(3), false);
    assert!(!g.vml.contains("LOOP_CNT") && !g.vml.contains("LAYERB"));
    assert_eq!(g.vml.matches("SPARSE_MATVEC").count(), 3, "展开成 3 份");
    assert!(g.loop_groups.is_empty() && g.extra_bins.is_empty());
    assert_eq!(
        sparrow::vmlgen::generate_from_src_with(
            &sparrow::parse(&sparrow::diag::SourceFile::new("t", model(3))).unwrap(),
            "t",
            GenOptions { loop_repeat: false, ..Default::default() },
        )
        .unwrap()
        .vml,
        g.vml,
        "默认路径必须与显式 loop_repeat=false 一致"
    );
}

// ---------------------------------------------------------------------------
// 循环体的结构约束（每条都是踩过/差点踩的坑）
// ---------------------------------------------------------------------------

#[test]
fn loop_body_only_touches_first_set_and_act_slots() {
    let g = gen_with(&model(24), true);
    let body: Vec<String> = g
        .vml
        .lines()
        .skip_while(|l| !l.starts_with("t__24__l1:"))
        .map(|l| l.trim().to_string())
        .take_while(|l| !l.starts_with("GEMM"))
        .collect();
    let body = body.join("\n");
    // 模板里的常量引用一律是第 1 份
    for c in ["t__1_off", "t__1_src", "t__1_W"] {
        assert!(body.contains(c), "模板缺常量 `{c}`：\n{body}");
    }
    // 除第 1 份以外，**任何**其它份的常量块都不能出现在循环体里
    //（第 N 份的槽位名可以出现 —— 那是下游接线用的输出槽）
    for r in 2..=24 {
        for sfx in ["off", "src", "W"] {
            assert!(
                !body.contains(&format!("t__{r}_{sfx}")),
                "循环体里出现了第 {r} 份的常量 `t__{r}_{sfx}` —— 那样 LAYERB 就没有意义了"
            );
        }
    }
    // 槽位用**最后一份**的名字：下游接线与展开版一字不改
    assert!(body.contains("t__24_out") && body.contains("t__24__in"));
    // 常量的 `@` 前缀只允许出现在 LOOP_CNT 的目标上（汇编器把 `@` 当跳转标签，
    // `COPY x, @c` 会 panic："指令 COPY 不支持跳转标签"）
    for line in g.vml.lines() {
        if let Some(pos) = line.find('@') {
            assert!(
                line.trim_start().starts_with("LOOP_CNT"),
                "`@` 只能出现在 LOOP_CNT 的目标上：{line}"
            );
            assert!(pos > 0);
        }
    }
    // 计数器与步长槽必须是**激活槽**（先 SHAPE 再用）：常量槽会被 layer_base 平移
    for s in ["t__24__ctr", "t__24__ls", "t__24__ns"] {
        assert!(g.vml.contains(&format!("SHAPE   {s}, 1, 1")), "`{s}` 必须先 SHAPE");
    }
    // LOOP_CNT 的目标必须**向后**（标签在 LOOP_CNT 之前），否则汇编器硬报错
    let lbl_at = g.vml.find("t__24__l1:").expect("有标签");
    let loop_at = g.vml.find("LOOP_CNT t__24__ctr, @t__24__l1").expect("有 LOOP_CNT");
    assert!(lbl_at < loop_at, "标签必须在 LOOP_CNT 之前（向后跳）");
}

#[test]
fn loop_mode_is_deterministic() {
    let a = gen_with(&model(24), true);
    let b = gen_with(&model(24), true);
    assert_eq!(a.vml, b.vml);
    assert_eq!(a.args_text, b.args_text);
    assert_eq!(a.extra_bins, b.extra_bins);
    assert_eq!(a.loop_groups[0].stride, 192, "off(64)+src(64)+W(64)");
    assert_eq!(a.loop_groups[0].iters, 24);
}

// ---------------------------------------------------------------------------
// 验收 5：步长对账（E210 的判据本体，带 can-fail）
// ---------------------------------------------------------------------------

/// 按汇编器的真实规则（每块补齐到 64）伪造一份 `.vmb`（头 + 常量池 + META）。
fn fake_table(g: &vmlgen::Generated, perturb: Option<(usize, i64)>) -> Vec<u8> {
    let n = g.consts.len();
    let mut offs = Vec::new();
    let mut cur = 0u64;
    for c in &g.consts {
        offs.push(cur);
        cur = align_up(cur + (c.rows as u64 * c.cols as u64 * 4), CONST_ALIGN as u64);
    }
    if let Some((i, d)) = perturb {
        offs[i] = (offs[i] as i64 + d) as u64;
    }
    let meta_len = 28 + n * 24;
    let const_len = cur as usize;
    let meta_off = 32 + const_len;
    let mut b: Vec<u8> = Vec::new();
    b.extend_from_slice(&0x4D4C_4D56u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&0u16.to_le_bytes());
    b.extend_from_slice(&32u32.to_le_bytes()); // text_off（这里没有 TEXT 段）
    b.extend_from_slice(&0u32.to_le_bytes()); // text_len
    b.extend_from_slice(&32u32.to_le_bytes()); // const_off
    b.extend_from_slice(&(const_len as u32).to_le_bytes());
    b.extend_from_slice(&(meta_off as u32).to_le_bytes());
    b.extend_from_slice(&(meta_len as u32).to_le_bytes());
    b.extend(std::iter::repeat(0u8).take(const_len)); // 常量池（内容与对账无关）
    // META
    b.extend_from_slice(&(n as u32).to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes()); // n_slot
    b.extend_from_slice(&0u32.to_le_bytes()); // slots_off
    b.extend_from_slice(&[0u8; 16]); // act/state/grad/names_off
    assert_eq!(b.len(), meta_off + 28);
    for (i, c) in g.consts.iter().enumerate() {
        b.extend_from_slice(&(offs[i] as u32).to_le_bytes());
        b.extend_from_slice(&(c.rows * c.cols * 4).to_le_bytes());
        b.extend_from_slice(&c.rows.to_le_bytes());
        b.extend_from_slice(&c.cols.to_le_bytes());
        b.push(0);
        b.extend_from_slice(&[0u8; 3]);
        b.extend_from_slice(&0u32.to_le_bytes());
    }
    b
}

#[test]
fn verify_accepts_consistent_pool_and_rejects_perturbations() {
    let g = gen_with(&model(24), true);

    // 正：与真实汇编器同一规则 → 通过，且步长 == 前端预测值
    let good = fake_table(&g, None);
    let ents = read_const_table(&good).expect("头/表结构合法");
    let n = vmlgen::verify_loop_groups(&g, &ents).expect("一致的布局必须通过");
    assert_eq!(n, 1, "对账了 1 个循环组");
    assert_eq!(g.loop_groups[0].stride, 192);

    // can-fail 1：把第 2 份的 off 块 +64（下标 = 第 1 份 3 个块 + 模型输入/嵌入表之前）
    let idx_t2_off = g
        .consts
        .iter()
        .position(|c| c.name == "t__2_off")
        .expect("有 t__2_off");
    let bad = fake_table(&g, Some((idx_t2_off, 64)));
    let ents = read_const_table(&bad).unwrap();
    let e = vmlgen::verify_loop_groups(&g, &ents).expect_err("步长被破坏必须报错");
    assert!(e.contains("不恒定"), "{e}");

    // can-fail 2：把**最后一份**的 off 块 -64 ⇒ 该处步长 128 ≠ 192（不恒定）
    let idx_t24_off = g
        .consts
        .iter()
        .position(|c| c.name == "t__24_off")
        .expect("有 t__24_off");
    let bad2 = fake_table(&g, Some((idx_t24_off, -64)));
    let ents = read_const_table(&bad2).unwrap();
    let e2 = vmlgen::verify_loop_groups(&g, &ents).expect_err("步长被破坏必须报错");
    assert!(e2.contains("不恒定"), "{e2}");

    // can-fail 3：条数不符
    let mut bad3 = fake_table(&g, None);
    let meta_off = u32::from_le_bytes([bad3[24], bad3[25], bad3[26], bad3[27]]) as usize;
    bad3[meta_off..meta_off + 4].copy_from_slice(&(g.consts.len() as u32 - 1).to_le_bytes());
    let ents = read_const_table(&bad3).unwrap();
    let e3 = vmlgen::verify_loop_groups(&g, &ents).expect_err("条数不符必须报错");
    assert!(e3.contains("条数不符"), "{e3}");

    // can-fail 4：某条常量的长度对不上（"池序 = 清单序"被打破）
    // 挑第 1 条（嵌入表 [8,4] = 128 B），把 len 谎报成 4
    let mut bad4 = fake_table(&g, None);
    let meta_off = u32::from_le_bytes([bad4[24], bad4[25], bad4[26], bad4[27]]) as usize;
    let ent1 = meta_off + 28 + 24;
    bad4[ent1 + 4..ent1 + 8].copy_from_slice(&4u32.to_le_bytes());
    let ents = read_const_table(&bad4).unwrap();
    let e4 = vmlgen::verify_loop_groups(&g, &ents).expect_err("长度不符必须报错");
    assert!(e4.contains("对不上"), "{e4}");
}

/// 循环化**源层**（没有前驱的 `重复` 组）必须报**命名诊断**，不能 panic。
///
/// 实测背景：V0.1.4 及以前这里是 `l.preds[0]` 越界 panic
/// （`index out of bounds: the len is 0 but the index is 0`），
/// 而本项目纪律是"非法输入硬失败并说清楚理由"。
#[test]
fn loop_group_on_a_source_layer_is_a_named_error() {
    let src = "模型 mdl:\n  嵌入 词表=8 维度=4 重复3次\n  线性 输出=4\n  连接 嵌入 线性\n  前向 嵌入 线性\n";
    let (code, msg) = err_of(src, true);
    assert_eq!(code, "E110");
    assert!(msg.contains("没有前驱"), "{msg}");
    // 展开模式（同一条链）**不该**受影响 —— 它每个实例读各自的输入槽
    let g = gen_with(src, false);
    assert!(g.loop_groups.is_empty());
    assert!(g.vml.contains("LINEAR_out"), "展开模式要能正常产出");
}

/// `.vmb` META 读取本身必须是硬的（magic/越界/放不下）。
#[test]
fn vmb_meta_reader_is_strict() {
    let g = gen_with(&model(6), true);
    let good = fake_table(&g, None);
    assert_eq!(read_const_table(&good).unwrap().len(), g.consts.len());
    let mut bad = good.clone();
    bad[0] = 0; // magic 错
    assert!(read_const_table(&bad).is_err());
    let mut short = good.clone();
    short.truncate(40);
    assert!(read_const_table(&short).is_err());
}
