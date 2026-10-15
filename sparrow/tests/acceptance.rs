//! Sparrow V0.1 —— 验收测试（对应任务书第十二部分 1–12 条）
//!
//! 每一项都对应一个 `#[test]`，断言的是**可复现的字节级事实**（sha256），
//! 不是"看起来对"。

use sparrow::diag::SourceFile;
use sparrow::sha256::sha256_hex;

// ---------------------------------------------------------------------------
// 工具
// ---------------------------------------------------------------------------

fn vml(src: &str) -> String {
    let sf = SourceFile::new("t.sparrow", src);
    match sparrow::compile_vml(&sf) {
        Ok((v, _)) => v,
        Err(d) => panic!("编译失败：\n{}", d.render(&sf)),
    }
}

fn try_vml(src: &str) -> Result<String, String> {
    let sf = SourceFile::new("t.sparrow", src);
    match sparrow::compile_vml(&sf) {
        Ok((v, _)) => Ok(v),
        Err(d) => Err(d.render(&sf)),
    }
}

fn ast(src: &str) -> sparrow::Ast {
    let sf = SourceFile::new("t.sparrow", src);
    match sparrow::parse(&sf) {
        Ok(a) => a,
        Err(d) => panic!("parse 失败：\n{}", d.render(&sf)),
    }
}

fn hash(s: &str) -> String {
    sha256_hex(s.as_bytes())
}

/// 去掉 span 的规范化文本形态。
/// 跨文件比较 AST 必须忽略字节偏移（中文标点是 3 字节，全角/半角写法会shift偏移）。
fn norm_ast(a: &sparrow::Ast) -> String {
    let mut out = String::new();
    for item in a.items() {
        match item {
            sparrow::Item::ModelDef { name, body, .. } => {
                out.push_str(&format!("model {}\n", name.name));
                for st in body {
                    norm_stmt(st, &mut out, 1);
                }
            }
            sparrow::Item::LayerDef { name, params, kind, attrs, .. } => {
                out.push_str(&format!(
                    "def {} {} params({}) attrs({})\n",
                    name.name,
                    kind.name,
                    params.iter().map(|p| p.name.clone()).collect::<Vec<_>>().join(","),
                    attrs
                        .iter()
                        .map(|(k, v)| format!("{}={}", k.name, v.render()))
                        .collect::<Vec<_>>()
                        .join(",")
                ));
            }
        }
    }
    out
}

fn norm_stmt(s: &sparrow::Stmt, out: &mut String, depth: usize) {
    let pad = "  ".repeat(depth);
    match s {
        sparrow::Stmt::Instantiate { name, kind, args, .. } => out.push_str(&format!(
            "{pad}inst {name} {kind} [{}]\n",
            args.iter().map(|a| a.render()).collect::<Vec<_>>().join(", ")
        )),
        sparrow::Stmt::Repeat { body, count, .. } => {
            out.push_str(&format!("{pad}repeat {} of\n", count.render()));
            norm_stmt(body, out, depth + 1);
        }
        sparrow::Stmt::Connect { path, .. } => out.push_str(&format!(
            "{pad}connect {}\n",
            path.iter().map(|p| p.name.clone()).collect::<Vec<_>>().join(" ")
        )),
        sparrow::Stmt::Forward { input, output, .. } => {
            out.push_str(&format!("{pad}forward {input} {output}\n"))
        }
        sparrow::Stmt::CustomOpDef { name, opcode, args, .. } => out.push_str(&format!(
            "{pad}cop {name} {} ({})\n",
            opcode.render(),
            args.iter().map(|a| a.name.clone()).collect::<Vec<_>>().join(",")
        )),
        sparrow::Stmt::RawOp { opcode, operands, .. } => out.push_str(&format!(
            "{pad}raw {opcode} {}\n",
            operands.iter().map(|o| o.render()).collect::<Vec<_>>().join(" ")
        )),
        // V0.1.6：训练陈述（"写法不同、语义相同"的归一化也要覆盖它们）
        sparrow::Stmt::Loss { kind, .. } => out.push_str(&format!("{pad}loss {kind}\n")),
        sparrow::Stmt::Backward { .. } => out.push_str(&format!("{pad}backward\n")),
    }
}

fn example(name: &str) -> String {
    let p = format!("{}/examples/{name}.sparrow", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("读不到 {p}：{e}"))
}

fn example_vml(name: &str) -> String {
    vml(&example(name))
}

// ---------------------------------------------------------------------------
// 验收 1：Level 0–6 全部示例解析通过
// ---------------------------------------------------------------------------

#[test]
fn acceptance_1_levels_0_to_6() {
    for n in 0..=6 {
        let name = format!("level{n}");
        let out = example_vml(&name);
        assert!(out.contains(".model"), "{name} 没产出 .model");
        assert!(out.contains(".end"), "{name} 没产出 .end");
    }
    // Level 0 进隐式模型 main
    assert!(example_vml("level0").contains(".model main"));
    // Level 4 的重复是结构化节点（REPEAT 24），不是文本展开
    let l4 = example_vml("level4");
    assert!(l4.contains("REPEAT  24 LAYER t SPARSE dim=1536 k0=8"), "{l4}");
    // Level 5 自定义算子
    assert!(example_vml("level5").contains("COP     我的激活 opcode=200 args(arg0,arg1)"));
    // Level 6 裸指令
    let l6 = example_vml("level6");
    assert!(l6.contains("RAW     load r0 cpool+0"), "{l6}");
    assert!(l6.contains("RAW     gemm r0 x r0 h"), "{l6}");
}

// ---------------------------------------------------------------------------
// 验收 2：纯中文 7 个示例全部解析通过
// ---------------------------------------------------------------------------

#[test]
fn acceptance_2_seven_chinese_examples() {
    // A–F 与 G 的自洽版（cn_g2）完整编译通过
    for name in ["cn_a", "cn_b", "cn_c", "cn_d", "cn_e", "cn_f", "cn_g2"] {
        let out = example_vml(name);
        assert!(out.contains(".model"), "{name} 没产出模型");
    }
    // 示例 G 逐字原文：**能解析**（AST 正常），但 sema 报未定义层类型 ——
    // 任务书的 G 单独不成立，它隐含依赖 F 的 `定义 稀疏块`。这是发现的边界，不是 bug。
    let g = ast(&example("cn_g"));
    assert_eq!(g.items().len(), 1);
    let sf = SourceFile::new("cn_g", example("cn_g"));
    let err = sparrow::compile_vml(&sf).unwrap_err();
    assert_eq!(err.code, "E002");
    assert!(err.message.contains("稀疏块"), "{}", err.message);
}

/// 边界发现（写进报告）：任务书说"7 个示例解析成同一个 AST"，但逐字看它们
/// **内容不同**（A 用隐式名 + `线性 输出=4054`；E/F/G 用显式名 e/t/o；G 少了
/// `重复24次` 且 `线性 4054`）。所以这里断言的是能成立的等价类：
/// A/B/C 三种写法同一个 AST；D 与 A 的差别只有 `线性 1536 4054` 多给了 in。
#[test]
fn acceptance_2b_same_ast_equivalence_classes() {
    let a = norm_ast(&ast(&example("cn_a")));
    let b = norm_ast(&ast(&example("cn_b")));
    let c = norm_ast(&ast(&example("cn_c")));
    assert_eq!(a, b, "A（英文标点）与 B（中文标点）必须同 AST");
    assert_eq!(a, c, "A 与 C（无缩进 + 块标记）必须同 AST");
    // D 的内容确实多一个 in=1536
    let d = example_vml("cn_d");
    assert!(d.contains("LINEAR LINEAR in=1536 out=4054"), "{d}");
    let a_vml = example_vml("cn_a");
    assert!(a_vml.contains("LINEAR LINEAR out=4054"), "{a_vml}");
}

// ---------------------------------------------------------------------------
// 验收 3：tiny_cn / tiny_en / tiny_mixed 三次输出 sha256 相同
// ---------------------------------------------------------------------------

#[test]
fn acceptance_3_cn_en_mixed_same_sha256() {
    let cn = hash(&example_vml("tiny_cn"));
    let en = hash(&example_vml("tiny_en"));
    let mixed = hash(&example_vml("tiny_mixed"));
    assert_eq!(cn, en, "中文版与英文版 .vml 不同");
    assert_eq!(cn, mixed, "混写版 .vml 与中文版不同");
    // 期望的规范化形态（人工核对过）
    assert!(example_vml("tiny_cn").contains("LAYER   EMBED EMBED vocab=4054 dim=1536"));
}

// ---------------------------------------------------------------------------
// 验收 4：缩进宽容（2/4/8 空格、Tab、宽度不统一）
// ---------------------------------------------------------------------------

#[test]
fn acceptance_4_indent_tolerance() {
    let body = |ind: &str| {
        format!(
            "模型 小:\n{ind}嵌入 512\n{ind}稀疏 512 k0=8\n{ind}线性 4054\n{ind}连接 嵌入 稀疏 线性\n"
        )
    };
    let two = vml(&body("  "));
    for (name, ind) in [("4 空格", "    "), ("8 空格", "        "), ("Tab", "\t")] {
        assert_eq!(two, vml(&body(ind)), "{name} 缩进产出不同 .vml");
    }
    // 同级宽度不统一：不报错，且仍然一样
    let ragged = "模型 小:\n  嵌入 512\n     稀疏 512 k0=8\n  线性 4054\n  连接 嵌入 稀疏 线性\n";
    assert_eq!(two, vml(ragged), "同级宽度不统一时产出不同 .vml");
}

// ---------------------------------------------------------------------------
// 验收 5：标点宽容（中文标点 / 英文标点 / 无标点）
// ---------------------------------------------------------------------------

#[test]
fn acceptance_5_punctuation_tolerance() {
    let cn = "模型 小：嵌入 512；稀疏 512 k0=8；线性 4054；连接 嵌入 稀疏 线性。";
    let en = "模型 小: 嵌入 512; 稀疏 512 k0=8; 线性 4054; 连接 嵌入 稀疏 线性.";
    // 无标点版本：靠换行 + "已知层类型处自动断句"推断层级
    let none = "模型 小 嵌入 512 稀疏 512 k0=8 线性 4054 连接 嵌入 稀疏 线性\n";
    let a = vml(cn);
    assert_eq!(a, vml(en), "中文标点与英文标点产出不同");
    // 无标点版本：靠换行/缩进断句，`稀疏 512 k0=8 线性 4054` 会自动在已知层类型处断开
    assert_eq!(a, vml(none), "无标点版本产出不同");
}

// ---------------------------------------------------------------------------
// 验收 6：块机制宽容（冒号+缩进 / 开始结束 / 花括号 / 一行分号）
// ---------------------------------------------------------------------------

#[test]
fn acceptance_6_block_mechanisms() {
    let base = "  嵌入 512\n  稀疏 512 k0=8\n  线性 4054\n  连接 嵌入 稀疏 线性\n";
    let colon = format!("模型 小:\n{base}");
    let marker = "模型 小 开始\n嵌入 512\n稀疏 512 k0=8\n线性 4054\n连接 嵌入 稀疏 线性\n模型 小 结束\n";
    let brace = "模型 小 {\n  嵌入 512\n  稀疏 512 k0=8\n  线性 4054\n  连接 嵌入 稀疏 线性\n}\n";
    let bracket = "模型 小 【\n  嵌入 512\n  稀疏 512 k0=8\n  线性 4054\n  连接 嵌入 稀疏 线性\n】\n";
    let oneline = "模型 小：嵌入 512；稀疏 512 k0=8；线性 4054；连接 嵌入 稀疏 线性\n";
    let a = vml(&colon);
    for (name, src) in [
        ("开始/结束", marker),
        ("花括号", brace),
        ("方括号", bracket),
        ("一行分号", oneline),
    ] {
        assert_eq!(a, vml(src), "{name} 块产出不同 .vml");
    }
}

// ---------------------------------------------------------------------------
// 验收 7：中文标识符（嵌入层/主干/x1/_tmp 等）
// ---------------------------------------------------------------------------

#[test]
fn acceptance_7_chinese_identifiers() {
    // 注意：第七部分的示例写成 `嵌入层 词表=4054`（层类型位置放的是名字），
    // V0.1 要求显式层类型，所以这里用 `名字 = 类型 …`（见报告"边界判断"）。
    let src = "模型 小:\n  嵌入层 = 嵌入 词表=4054 维度=1536\n  主干 = 稀疏 维度=1536 连接=8\n  输出层 = 线性 输出=4054\n  x1 = 归一化 1536\n  _tmp = 激活 act=relu\n  连接 嵌入层 主干 输出层\n  前向 嵌入层 输出层\n";
    let out = vml(src);
    for id in ["嵌入层", "主干", "输出层", "x1", "_tmp"] {
        assert!(out.contains(id), "标识符 {id} 没进 .vml：\n{out}");
    }
    // 中英混用标识符
    let mixed = "模型 小:\n  embed_layer = 嵌入 词表=4054 维度=1536\n  trunk = 稀疏 维度=1536 连接=8\n  output = 线性 输出=4054\n  连接 embed_layer trunk output\n";
    let out = vml(mixed);
    assert!(out.contains("embed_layer"));
    assert!(out.contains("trunk") && out.contains("output"));
    // 纯中文标识符也能直接当名字用（`名字 类型 参数` 顺序）
    let cn_names = "模型 小:\n  嵌入层 = 嵌入 512\n  主干 = 稀疏 512 8\n  连接 嵌入层 主干\n";
    let out = vml(cn_names);
    assert!(out.contains("LAYER   嵌入层 EMBED"), "{out}");
    assert!(out.contains("LAYER   主干 SPARSE"), "{out}");
}

// ---------------------------------------------------------------------------
// 验收 8：中文标点作分隔符 → 3 个标识符的列表
// ---------------------------------------------------------------------------

#[test]
fn acceptance_8_punctuation_as_separator() {
    for (name, sep) in [
        ("中文逗号", "，"),
        ("顿号", "、"),
        ("英文分号", ";"),
        ("中文分号", "；"),
        ("英文逗号", ","),
    ] {
        let src = format!("模型 小:\n  嵌入 512\n  稀疏 512 k0=8\n  线性 4054\n  连接 嵌入{sep}稀疏{sep}线性\n");
        let out = vml(&src);
        assert!(
            out.contains("CONNECT EMBED SPARSE LINEAR"),
            "{name} 没解析成 3 元素连接：\n{out}"
        );
    }
}

// ---------------------------------------------------------------------------
// 验收 9：关键字冲突场景报错清晰
// ---------------------------------------------------------------------------

#[test]
fn acceptance_9_keyword_conflict() {
    // `m` 是 model 的关键字别名，不能当实例名
    let sf = SourceFile::new("t.sparrow", "模型 小:\n  m = 嵌入 512\n");
    let d = sparrow::compile_vml(&sf).unwrap_err();
    assert_eq!(d.code, "E007");
    let r = d.render(&sf);
    assert!(r.contains("是关键字"), "{r}");
    assert!(r.contains("转义"), "{r}");
    assert!(r.contains("t.sparrow:2:3"), "{r}");
    // 转义后合法
    let ok = vml("模型 小:\n  @m = 嵌入 512\n  线性 4054\n  连接 @m 线性\n");
    assert!(ok.contains("LAYER   m EMBED dim=512"), "{ok}");
    assert!(ok.contains("CONNECT m LINEAR"), "{ok}");
    // 引用处忘记转义：报错要指出转义方法
    let sf2 = SourceFile::new("t.sparrow", "模型 小:\n  嵌入 512\n  线性 4054\n  连接 m 线性\n");
    let r2 = sparrow::compile_vml(&sf2).unwrap_err().render(&sf2);
    assert!(r2.contains("@m"), "{r2}");
    // 双引号转义也可以
    let ok2 = vml("模型 小:\n  \"x\" = 嵌入 512\n");
    assert!(ok2.contains("LAYER   x EMBED dim=512"), "{ok2}");
}

// ---------------------------------------------------------------------------
// 验收 10：≥20 组"语义相同、写法不同"→ 相同 .vml
// ---------------------------------------------------------------------------

const GROUPS: &[(&str, &[&str])] = &[
    (
        "01 隐式/显式模型头",
        &[
            "嵌入 512\n",
            "模型 main:\n  嵌入 512\n",
            "网络 main 开始\n嵌入 512\n网络 main 结束\n",
        ],
    ),
    (
        "02 位置/命名参数",
        &[
            "模型 小:\n  嵌入 512\n",
            "模型 小:\n  嵌入 维度=512\n",
            "模型 小:\n  Embed(dim=512)\n",
        ],
    ),
    (
        "03 两个位置参数 = 两个命名参数（顺序任意）",
        &[
            "模型 小:\n  嵌入 4054 1536\n",
            "模型 小:\n  嵌入 词表=4054 维度=1536\n",
            "模型 小:\n  嵌入 维度=1536 词表=4054\n",
            "模型 小:\n  Embed(vocab=4054, dim=1536)\n",
        ],
    ),
    (
        "04 层类型缩写/大小写",
        &[
            "模型 小:\n  嵌 512\n",
            "模型 小:\n  Embed(512)\n",
            "模型 小:\n  EMBED 512\n",
            "模型 小:\n  emb 512\n",
        ],
    ),
    (
        "05 稀疏层的四种写法",
        &[
            "模型 小:\n  稀疏 512 k0=8\n",
            "模型 小:\n  Sparse(512, 8)\n",
            "模型 小:\n  稀疏 512 8\n",
            "模型 小:\n  稀疏 维度=512 连接=8\n",
            "模型 小:\n  sparse d=512 k=8\n",
        ],
    ),
    (
        "06 线性层输出维度",
        &[
            "模型 小:\n  线性 4054\n",
            "模型 小:\n  线性 输出=4054\n",
            "模型 小:\n  Linear(out=4054)\n",
            "模型 小:\n  线 出=4054\n",
            "模型 小:\n  dense(out_dim=4054)\n",
        ],
    ),
    (
        "07 连接写法：关键字/箭头/逗号/分号/函数调用",
        &[
            "模型 小:\n  嵌入 512\n  线性 4054\n  连接 嵌入 线性\n",
            "模型 小:\n  嵌入 512\n  线性 4054\n  嵌入 -> 线性\n",
            "模型 小:\n  嵌入 512\n  线性 4054\n  嵌入，线性\n",
            "模型 小:\n  嵌入 512\n  线性 4054\n  嵌入;线性\n",
            "模型 小:\n  嵌入 512\n  线性 4054\n  connect(嵌入, 线性)\n",
        ],
    ),
    (
        "08 自然语序连接（三个层）",
        &[
            "模型 小:\n  嵌入 512\n  稀疏 512 8\n  线性 4054\n  把 嵌入 连到 稀疏 再连到 线性\n",
            "模型 小:\n  嵌入 512\n  稀疏 512 8\n  线性 4054\n  连接 嵌入 稀疏 线性\n",
            "模型 小:\n  嵌入 512\n  稀疏 512 8\n  线性 4054\n  嵌入 -> 稀疏 -> 线性\n",
        ],
    ),
    (
        "09 中文复合连接词 连到",
        &[
            "模型 小:\n  嵌入 512\n  线性 4054\n  把 嵌入 连到 线性\n",
            "模型 小:\n  嵌入 512\n  线性 4054\n  嵌入 连到 线性\n",
            "模型 小:\n  嵌入 512\n  线性 4054\n  嵌入 -> 线性\n",
        ],
    ),
    (
        "10 前向的四种写法",
        &[
            "模型 小:\n  嵌入 512\n  线性 4054\n  前向 嵌入 线性\n",
            "模型 小:\n  嵌入 512\n  线性 4054\n  forward(嵌入, 线性)\n",
            "模型 小:\n  嵌入 512\n  线性 4054\n  从 嵌入 前向到 线性\n",
            "模型 小:\n  嵌入 512\n  线性 4054\n  从 嵌入 前向 线性\n",
            "模型 小:\n  嵌入 512\n  线性 4054\n  forward 嵌入 线性\n",
        ],
    ),
    (
        "11 重复的六种写法",
        &[
            "模型 小:\n  嵌入 512\n  t = 稀疏(512, 8) 重复24次\n  线性 4054\n",
            "模型 小:\n  嵌入 512\n  t = 稀疏(512, 8) x 24\n  线性 4054\n",
            "模型 小:\n  嵌入 512\n  t = 稀疏(512, 8) * 24\n  线性 4054\n",
            "模型 小:\n  嵌入 512\n  t = 稀疏(512, 8) 24层\n  线性 4054\n",
            "模型 小:\n  嵌入 512\n  t = 稀疏(512, 8) 重复 二十四 次\n  线性 4054\n",
            "模型 小:\n  嵌入 512\n  t = 稀疏(512, 8) 叠 24 个\n  线性 4054\n",
        ],
    ),
    (
        "12 层定义展开 = 直写",
        &[
            "定义 稀疏块 用 维 连 为:\n  稀疏 维度=维 连接=连\n\n模型 小:\n  嵌入 512\n  t = 稀疏块(512, 8)\n  线性 4054\n  连接 嵌入 t 线性\n",
            "模型 小:\n  嵌入 512\n  t = 稀疏 512 8\n  线性 4054\n  连接 嵌入 t 线性\n",
            "定义 稀疏块(维, 连):\n  稀疏 维度=维 连接=连\n模型 小:\n  嵌入 512\n  t = 稀疏块(512, 8)\n  线性 4054\n  连接 嵌入 t 线性\n",
        ],
    ),
    (
        "13 注释与空行不影响",
        &[
            "模型 小:\n  嵌入 512\n  线性 4054\n",
            "# 头注释\n模型 小:   // 尾注释\n  嵌入 512\n\n   /* 块注释 */ 线性 4054\n",
            "模型 小:\n  嵌入 512。;\n  线性 4054\n",
        ],
    ),
    (
        "14 模型头：冒号/无冒号/花括号/标记词",
        &[
            "模型 小:\n  嵌入 512\n",
            "模型 小\n  嵌入 512\n",
            "模型 小 {\n  嵌入 512\n}\n",
            "模型 小 开始\n嵌入 512\n模型 小 结束\n",
            "模型 小【嵌入 512】\n",
        ],
    ),
    (
        "15 裸指令：逗号/空白分隔",
        &[
            "模型 裸:\n  @load r0 cpool+0\n  @gemm r0 x r0 -> h\n  @store h -> y\n",
            "模型 裸:\n  @load r0, cpool+0\n  @gemm(r0, x, r0) -> h\n  @store h, y\n",
            "模型 裸 开始\n  @load r0 cpool+0\n  @gemm r0 x r0 h\n  @store h y\n模型 裸 结束\n",
        ],
    ),
    (
        "16 自定义算子的两种声明",
        &[
            "模型 自研:\n  嵌入 512\n  算子 f = opcode(200) args(2)\n  f\n",
            "模型 自研:\n  嵌入 512\n  算子 f 操作码 200 参数 arg0 arg1\n  f\n",
            "模型 自研:\n  嵌入 512\n  算子 f : 操作码(200) 形参(arg0, arg1)\n  f\n",
        ],
    ),
    (
        "17 标点体系：中文/英文/无",
        &[
            "模型 小：嵌入 512；线性 4054；连接 嵌入 线性。",
            "模型 小: 嵌入 512; 线性 4054; 连接 嵌入 线性.",
            "模型 小 嵌入 512 线性 4054 连接 嵌入 线性\n",
        ],
    ),
    (
        "18 数值写法（1e-5 = 0.00001）",
        &[
            "模型 小:\n  嵌入 512\n  归一化 512 eps=1e-5\n",
            "模型 小:\n  嵌入 512\n  归一化 512 eps=0.00001\n",
            "模型 小:\n  嵌入 512\n  norm(512, eps=1E-5)\n",
        ],
    ),
    (
        "19 属性键语言/缩写",
        &[
            "模型 小:\n  嵌入 维度=512\n",
            "模型 小:\n  嵌入 dim=512\n",
            "模型 小:\n  嵌入 维=512\n",
            "模型 小:\n  嵌入 D=512\n",
        ],
    ),
    (
        "20 关键字大小写混写",
        &[
            "模型 小:\n  嵌入 512\n  线性 4054\n  连接 嵌入 线性\n",
            "MoDeL 小:\n  EmBeD 512\n  LiNeAr 4054\n  CoNnEcT 嵌入 线性\n",
            "MODEL 小 开始\nEMBED 512\nLINEAR 4054\nCONNECT 嵌入 线性\nMODEL 小 结束\n",
        ],
    ),
];

#[test]
fn acceptance_10_variant_groups() {
    assert!(GROUPS.len() >= 20, "变体组不足 20：{}", GROUPS.len());
    let mut checked = 0usize;
    for (name, variants) in GROUPS {
        assert!(variants.len() >= 2, "{name} 少于 2 种写法");
        let first = vml(variants[0]);
        for (i, v) in variants.iter().enumerate().skip(1) {
            let got = vml(v);
            assert_eq!(
                first, got,
                "\n变体组 [{name}] 第 {} 种写法产出不同 .vml\n--- 基准 ---\n{first}\n--- 实际 ---\n{got}",
                i + 1
            );
            checked += 1;
        }
    }
    println!("变体组 {} 组，比较 {} 次，全部同 .vml", GROUPS.len(), checked);
}

// ---------------------------------------------------------------------------
// 验收 11：错误报告（未定义层 / 参数缺失 / 括号不匹配）
// ---------------------------------------------------------------------------

fn err_of(src: &str) -> (String, String) {
    let sf = SourceFile::new("t.sparrow", src);
    match sparrow::compile_vml(&sf) {
        Ok(_) => panic!("本应报错却成功了：{src}"),
        Err(d) => (d.code.to_string(), d.render(&sf)),
    }
}

/// ⑦ 的**可判定边界**（V0.1.6 收窄）：`训练`/`优化` 一类仍报 `E111`（说清"语言还没有
/// 训练循环/参数更新，而引擎侧的算子已经存在"），而不是 E002「层类型未定义」（暗示用户打错）。
/// V0.1.6 起 `损失`/`反向` 已经是**真语法**（见 tests/train.rs），不再被这条拦。
/// 同时保证不误伤：真打错的层类型仍报 E002，把这些词当**实例名**也要照常工作。
#[test]
fn training_constructs_are_a_named_boundary() {
    for word in ["训练", "优化", "train", "optimizer"] {
        let src = format!("模型 t:\n  线性 输入=4 输出=4\n  {word} 步数=10\n  前向 线性 线性\n");
        let (code, r) = err_of(&src);
        assert_eq!(code, "E111", "{word} 应报 E111：{r}");
        assert!(r.contains("训练循环"), "{r}");
        assert!(r.contains("OP_ADAM"), "提示里要点出参数更新算子：{r}");
    }
    // 不再是边界：`损失 均方` / `反向`（V0.1.6）
    assert!(
        try_vml("模型 t:\n  线性 输入=4 输出=4\n  损失 均方\n  反向\n  前向 线性 线性\n").is_ok(),
        "`损失 均方` + `反向` 应该能编译"
    );
    // can-fail 1：真打错仍然是 E002（不能被 E111 吞掉）
    let (code, _) = err_of("模型 t:\n  线形 输入=4 输出=4\n  前向 线形 线形\n");
    assert_eq!(code, "E002");
    // can-fail 2：把这些词当**实例名**用不该触发任何训练诊断
    assert!(
        try_vml("模型 t:\n  训练 = 线性 输入=4 输出=4\n  前向 训练 训练\n").is_ok(),
        "`训练 = …` 是实例名，E111 不该拦"
    );
    assert!(
        try_vml("模型 t:\n  损失 = 线性 输入=4 输出=4\n  前向 损失 损失\n").is_ok(),
        "`损失 = …` 是实例名，不该被当成损失声明"
    );
}

#[test]
fn acceptance_11_error_reports() {
    // (1) 未定义的层类型：行号、列号、上下文、建议、候选
    let (code, r) = err_of("模型 小:\n  嵌入 512\n  t = Spars(1536, 8)\n");
    assert_eq!(code, "E002");
    assert!(r.contains("错误[E002]：未定义的层类型 \"Spars\""), "{r}");
    assert!(r.contains("位置：t.sparrow:3:7"), "{r}");
    assert!(r.contains("3 |   t = Spars(1536, 8)"), "{r}");
    assert!(r.contains("^^^^^"), "{r}");
    assert!(r.contains("提示：是不是想写 \"Sparse\"？"), "{r}");
    assert!(r.contains("候选：") && r.contains("Sparse") && r.contains("Embed"), "{r}");

    // (2) 参数缺失
    let (code, r) = err_of(
        "定义 稀疏块 用 维 连 为:\n  稀疏 维度=维 连接=连\n\n模型 小:\n  t = 稀疏块(1536)\n",
    );
    assert_eq!(code, "E003");
    assert!(r.contains("需要 2 个参数（维, 连），只给了 1 个"), "{r}");
    assert!(r.contains("位置：t.sparrow:5:7"), "{r}");
    assert!(r.contains("提示：例：`稀疏块(<维>, <连>)`"), "{r}");

    // (3) 括号不匹配：指向开括号
    let (code, r) = err_of("模型 小:\n  嵌入 512\n  t = 稀疏块(1536, 8\n");
    assert_eq!(code, "E004");
    assert!(r.contains("括号不匹配"), "{r}");
    assert!(r.contains("位置：t.sparrow:3:10"), "{r}");
    assert!(r.contains("提示：在文件末尾补一个 `)`"), "{r}");

    // (4) 多余的右括号
    let (code, r) = err_of("模型 小:\n  嵌入 512)\n");
    assert_eq!(code, "E004");
    assert!(r.contains("多余的"), "{r}");

    // (5) 没有"语法错误"这种模糊报错：每个错误都带稳定错误码
    for src in [
        "模型 小:\n  @@@\n",
        "模型 小:\n  嵌入\n",
        "模型 小:\n  嵌入 512\n  连接 嵌入 注意力\n",
        "",
    ] {
        let sf = SourceFile::new("t.sparrow", src);
        if let Err(d) = sparrow::compile_vml(&sf) {
            assert!(d.code.starts_with('E'), "错误码不规范：{}", d.code);
            assert!(!d.message.contains("语法错误"), "报错太模糊：{}", d.message);
        } else {
            panic!("本应报错：{src:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// 验收 12：AST JSON
// ---------------------------------------------------------------------------

#[test]
fn acceptance_12_ast_json() {
    let json = sparrow::ast_to_json(&ast(&example("level4")));
    // 命名参数（`Expr::Named`）在 tiny_cn 里
    let json_named = sparrow::ast_to_json(&ast(&example("tiny_cn")));
    for key in [
        "\"kind\": \"Program\"",
        "\"item\": \"ModelDef\"",
        "\"item\": \"LayerDef\"",
        "\"stmt\": \"Instantiate\"",
        "\"stmt\": \"Repeat\"",
        "\"stmt\": \"Connect\"",
        "\"stmt\": \"Forward\"",
        "\"expr\": \"Ident\"",
        "\"expr\": \"Int\"",
        "\"count\": {\"expr\": \"Int\", \"value\": 24}",
        "\"params\": [\"维\", \"连\"]",
    ] {
        assert!(json.contains(key), "AST JSON 缺少 {key}：\n{json}");
    }
    for key in ["\"expr\": \"Named\"", "\"key\": \"词表\"", "\"value\": {\"expr\": \"Int\", \"value\": 4054}"] {
        assert!(json_named.contains(key), "AST JSON 缺少 {key}：\n{json_named}");
    }
    // 括号配平（跳过字符串内的括号）
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for c in json.chars() {
        if in_str {
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '{' | '[' => depth += 1,
            '}' | ']' => depth -= 1,
            _ => {}
        }
        assert!(depth >= 0, "JSON 括号不配平");
    }
    assert_eq!(depth, 0, "JSON 括号不配平");
    // 中文不转义，保持可读
    assert!(json.contains("\"name\": \"大\"") || json.contains("\"name\": \"小\""), "{json}");
}

// ---------------------------------------------------------------------------
// 额外：宽容性反向测试（不该被吞成关键字的标识符）
// ---------------------------------------------------------------------------

#[test]
fn identifiers_not_swallowed_by_keywords() {
    let src = "模型 小:\n  嵌入层 = 嵌入 512\n  主干 = 稀疏 512\n  x1 = 线性 512\n  _tmp = 归一化 512\n  入 = 激活 act=relu\n  干 = 归一化 512\n  出 = 线性 512\n";
    let out = vml(src);
    for id in ["嵌入层", "主干", "x1", "_tmp", "入", "干", "出"] {
        assert!(out.contains(id), "{id} 被关键字吞掉了：\n{out}");
    }
}

#[test]
fn ast_stats_sane() {
    let a = ast(&example("level2"));
    let s = a.stats();
    assert_eq!(s.models, 1);
    assert_eq!(s.instantiates, 3);
    assert_eq!(s.connects, 1);
}
