//! Sparrow V0.1.4 —— 路径 2 后端：`.spar → Rust AOT 源`（`--backend lib --lang rust`）
//!
//! # 为什么这个后端能"与 `.vmb` 后端逐字符一致"
//!
//! 朴素做法（生成 `for` 循环算 GEMM）**做不到**逐位一致：引擎的 `gemm_nt` 是
//! AVX2 + FMA + 8×8 分块，bf16 又是另一套双累加器内核 —— 实测纯 numpy 参考
//! （四舍五入**和** longdouble-FMA 模拟）都差 1 ULP。
//!
//! 所以这里走**同源**路线：生成的 Rust 直接 `use vmlm_ops::kernels::*`，
//! 调**同一个函数**；其余"逐元素 / 搬运"类算子在 `interp.rs` 里本来就是
//! 平铺的标量循环（`BIAS` / `SCALE` / `TRANSPOSE` / `SLICE` / `CONCAT` /
//! `EWADD` / 稀疏 CSR 累加），按同一规则手写即可逐位一致。
//!
//! | Sparrow 层 | 生成的 Rust 调什么 |
//! |---|---|
//! | 线性 | `gemm_nt` / `gemm_nt_bf16`（权重的 dtype 决定） + `BIAS` 标量循环 |
//! | 归一化 | `layernorm`（eps 与引擎默认一致 1e-5） |
//! | 激活 | `act_avx2`（引擎用的就是它） |
//! | 稀疏 | 标量 CSR 累加（`acc = term` 首项 / `acc += term`，与 `interp.rs` 同序） |
//! | 嵌入 | 行拷贝（f32）或逐元素加宽（bf16） |
//! | 注意力 | QKV/输出投影走 `gemm_nt`，`gemm`(NN) 做 QK^T 与 ·V，`softmax` 带 causal |
//! | 汇入 | `EWADD` ⇒ 逐元素加；`CONCAT` ⇒ 列拼接 |
//! | 打印 | `vmlm_ops::interp::print_f32`（引擎那个**截断**式打印） |
//!
//! 循环化的 `重复` 组（V0.1.5 起支持）：`.vmb` 那条路把**常量池基址**按 `LAYERB`
//! 平移、用 `LOOP_CNT` 转 N 圈；Rust 不能平移指针，于是生成
//! `for inst in &self.g_items { carry = inst.forward(&carry, rows); }` ——
//! **每份实例的常量各自 `include_bytes!`**（`t__1_W` … `t__N_W`）。
//! 两侧语义逐条对齐：进位槽 = 上一圈输出；外部前驱每圈参与一次汇入，且
//! 汇入的**操作数顺序**与文本后端一致（`[进位, 外部前驱…]`，因为拼接不可交换）。
//!
//! 不适用的情形**明确拒绝**（`E109`），不静默降级：
//! * 非 f32/bf16 的权重（int8 GEMM 的 scale 语义还没接）。

use std::fmt::Write as _;

use crate::diag::{Diag, Result};
use crate::vmlgen::{DataType, Generated, MergeMode, Plan, PlanLayer};

/// 路径 2 后端的生成选项（V0.1.5）。
///
/// 目前只有一处：**基准迭代次数**。为什么要它 —— 「AOT 比解释器快多少」必须
/// 用**同一个口径**测，而进程启动（CreateProcess + 页错误 + 装载）会把它盖掉：
/// 引擎自带一套进程内基准（`vmlm-runtime-bench.exe <model.vmb> <reps>`，
/// 见 `runtime/src/bench.rs`：预热 1 遍 + QueryPerformanceCounter 计时 K 遍，
/// 打印 `avg us/iter = N`）。本选项让生成的 Rust 有**同格式、同口径**的计时：
/// 预热 1 遍 + `std::time::Instant` 计时 K 遍，打印同一行文本。
///
/// `bench_iters = 0`（默认）= **不生成任何计时代码**，输出与 V0.1.4 逐字节相同。
#[derive(Clone, Copy, Debug, Default)]
pub struct LibOptions {
    pub bench_iters: u32,
}

/// 生成的 crate 骨架（`Cargo.toml`）—— 依赖 `vmlm-ops` 是为了**同源算术**。
pub fn cargo_toml(vmlm_root: &str, name: &str) -> String {
    format!(
        r#"# 由 sparrow 生成（--backend lib --lang rust）。不要手改：重跑会覆盖。
[package]
name = "{name}"
version = "0.1.0"
edition = "2021"

# 生成的 crate 常常落在宿主仓库目录里（比如 `out/`）—— 空 `[workspace]`
# 明确声明"我是独立包"，否则 cargo 会以为它属于上层 workspace 并报错。
[workspace]

[[bin]]
name = "{name}"
path = "src/main.rs"

[dependencies]
# 关键：**同一个**内核实现 ⇒ 与 .vmb 后端逐位一致
vmlm-ops = {{ path = "{vmlm_root}/ops" }}

[profile.release]
opt-level = 3
"#
    )
}

/// 常量名 → 结构体字段名（`t__1_W` → `w`，`a__2_bWq` → `bwq`）。
fn lower(full: &str) -> String {
    full.rsplit('_').next().unwrap_or(full).to_ascii_lowercase()
}

fn rust_str(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// 常量文件名 → 生成代码里的标识符（`t__1_W` → `C_t__1_W`）。
fn ident(name: &str) -> String {
    let mut out = String::from("C_");
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            out.push(c);
        } else {
            let _ = write!(out, "u{:04x}", c as u32);
        }
    }
    out
}

/// 逐层类型需要的常量名（与 `vmlgen::emit_layer_body` 的命名规则一致）。
///
/// `int8` 的线性层是**两个**常量：量化权重 `_Wq`（i8，1 B/元素）与每输出行一个
/// `_scale`（f32）。判据是"常量表里有没有 `_Wq`"—— 这样前端一份 `Generated`
/// 就能同时服务三条路，不需要在 Plan 之外再记一份精度状态。
fn consts_of(l: &PlanLayer, consts: &[crate::vmlgen::ConstDecl]) -> Vec<(String, String, &'static str)> {
    let n = &l.base;
    let c = |suffix: &str, what: &'static str| {
        (format!("{n}_{suffix}"), format!("{n}_{suffix}.bin"), what)
    };
    let is_int8 = consts
        .iter()
        .any(|x| x.name == format!("{n}_Wq") && x.dtype == DataType::Int8);
    match l.kind.as_str() {
        "EMBED" => vec![c("E", "嵌入表")],
        "SPARSE" => vec![c("off", "CSR 行偏移"), c("src", "CSR 列索引"), c("W", "CSR 权重")],
        "LINEAR" if is_int8 => vec![
            c("Wq", "int8 量化权重 [out, in]"),
            c("scale", "每输出行的反量化 scale"),
            c("b", "线性偏置"),
        ],
        "LINEAR" => vec![c("W", "线性权重"), c("b", "线性偏置")],
        "NORM" => vec![c("g", "gamma"), c("b", "beta")],
        "ACT" => vec![],
        // V0.1.6：注意力的**投影权重**也能吃 int8（它们就是四组 `gemm_nt`）。
        // 命名的坑：`{层}_scale` 已经被"softmax 的 1/sqrt(hd)"占了，所以反量化 scale
        // 必须叫 `{层}_Wq_scale` 这种带投影名的形式 —— 否则两个 scale 会撞成同一个字段。
        "ATTN" if is_int8 => vec![
            c("Wq", "Q 权重（int8）"),
            c("sWq", "Q 每输出行反量化 scale"),
            c("bWq", "Q 偏置"),
            c("Wk", "K 权重（int8）"),
            c("sWk", "K 每输出行反量化 scale"),
            c("bWk", "K 偏置"),
            c("Wv", "V 权重（int8）"),
            c("sWv", "V 每输出行反量化 scale"),
            c("bWv", "V 偏置"),
            c("Wo", "输出投影权重（int8）"),
            c("sWo", "输出投影每输出行反量化 scale"),
            c("bWo", "输出投影偏置"),
            c("scale", "1/sqrt(hd)"),
        ],
        "ATTN" => vec![
            c("Wq", "Q 权重"),
            c("bWq", "Q 偏置"),
            c("Wk", "K 权重"),
            c("bWk", "K 偏置"),
            c("Wv", "V 权重"),
            c("bWv", "V 偏置"),
            c("Wo", "输出投影权重"),
            c("bWo", "输出投影偏置"),
            c("scale", "1/sqrt(hd)"),
        ],
        _ => vec![],
    }
}

/// 生成整个 crate：`src/main.rs` 的正文（默认选项：不计时）。
pub fn generate_rust(gen: &Generated, consts_dir: &str) -> Result<String> {
    generate_rust_with(gen, consts_dir, LibOptions::default())
}

/// 生成整个 crate：`src/main.rs` 的正文，带选项（见 [`LibOptions`]）。
pub fn generate_rust_with(gen: &Generated, consts_dir: &str, opts: LibOptions) -> Result<String> {
    let plan = &gen.plan;
    for l in &plan.layers {
        if !matches!(l.kind.as_str(), "EMBED" | "SPARSE" | "LINEAR" | "NORM" | "ACT" | "ATTN") {
            return Err(Diag::error(
                "E109",
                format!("`--backend lib` 还不支持层类型 `{}`", l.kind),
                crate::ast::Span::ZERO,
            ));
        }
    }
    // V0.1.6：反向（训练）只在 `.vmb` 那条路上落地。Rust 后端要接反向，得先有
    // **自动微分式的代码生成**（每个层类型都要一份 backward 函数 + 梯度槽管理），
    // 那是另一个量级的工作 —— 这里**硬拒绝**，而不是生成一份"看起来能训练"的代码。
    if plan.backward {
        return Err(Diag::error(
            "E115",
            "`--backend lib`（Rust AOT）还不支持 `反向`",
            crate::ast::Span::ZERO,
        )
        .with_hint(
            "反向现在只有 `.vmb` 那条路（用 `--asm` 汇编 + 训练版运行时跑，见 \
             tools/train_ab.ps1 与 REPORT_V015.md §11）。AOT 侧要接反向，\
             得给每个层类型生成一份 backward 函数。",
        ));
    }

    let mut s = String::new();
    s.push_str(&header(plan, consts_dir));

    // ---- 1. 常量：include_bytes! + 载入函数 ----
    // 循环控制常量（`<前缀>_iters` / `_lstride` / `_nstride`）是 `LOOP_CNT`+`LAYERB`
    // 的**实现细节**：Rust 版直接遍历实例，不需要它们。跳过以免生成未使用的 static。
    let ctl: std::collections::HashSet<String> = gen
        .loop_groups
        .iter()
        .flat_map(|g| {
            [
                format!("{}_iters", g.cprefix),
                format!("{}_lstride", g.cprefix),
                format!("{}_nstride", g.cprefix),
            ]
        })
        .collect();
    for c in &plan.consts {
        if ctl.contains(&c.name) {
            continue;
        }
        let path = std::path::Path::new(consts_dir).join(&c.file);
        let path = path.to_string_lossy().replace('\\', "/");
        s.push_str(&format!(
            "static {}_RAW: &[u8] = include_bytes!(\"{}\");\n",
            ident(&c.name),
            rust_str(&path)
        ));
    }
    s.push('\n');
    s.push_str(LOADERS);

    // ---- 2. 每个层一个 struct + impl；循环组的 N 份合成一个「实例表」 ----
    let mut meta: Vec<String> = Vec::new();
    for (i, l) in plan.layers.iter().enumerate() {
        if group_of(plan, i).is_some() {
            continue; // 组内实例由 G{g} 统一承载
        }
        s.push_str(&layer_struct(i, l, &plan.consts)?);
        meta.push(format!("    l{i}: L{i},"));
    }
    for (g, grp) in plan.groups.iter().enumerate() {
        s.push_str(&group_struct(g, grp, plan)?);
        meta.push(format!("    g{g}: G{g},"));
    }

    // ---- 3. Model：按拓扑序跑 ----
    s.push_str("\nstruct Model {\n");
    for m in &meta {
        s.push_str(m);
        s.push('\n');
    }
    s.push_str("}\n\nimpl Model {\n");
    s.push_str("    fn forward(&self, input: &[f32]) -> Vec<f32> {\n");
    let rows = plan.t_seq.max(1) as usize;
    s.push_str(&format!("        let rows = {rows}usize;\n"));
    s.push_str("        let mut slots: std::collections::HashMap<&str, Vec<f32>> = std::collections::HashMap::new();\n");
    s.push_str(&format!(
        "        slots.insert(\"{}\", input.to_vec());\n",
        rust_str(&input_const_name(plan))
    ));
    for &i in &plan.order {
        // 循环组的实例：只在**第 1 份**处发射一次 `for` 循环（与文本后端同一触发点）
        if let Some(g) = group_of(plan, i) {
            if plan.groups[g].insts[0] != i {
                continue;
            }
            s.push_str(&group_loop(g, plan)?);
            continue;
        }
        let l = &plan.layers[i];
        // 汇入：先把前驱合成一个输入槽
        let in_expr = if l.preds.len() >= 2 {
            let srcs: Vec<String> = l
                .preds
                .iter()
                .map(|&p| format!("&slots[\"{}\"]", slot_of(plan, p)))
                .collect();
            let dst = format!("{}__merge", plan.slot_base[i]);
            let mode = match l.merge {
                MergeMode::Add => "add",
                MergeMode::Concat => "concat",
            };
            s.push_str(&format!(
                "        let {dst} = merge_{mode}(&[{}], {rows}, {});\n",
                srcs.join(", "),
                l.in_dim
            ));
            s.push_str(&format!("        slots.insert(\"{dst}\", {dst});\n"));
            format!("slots[\"{dst}\"]")
        } else if l.preds.is_empty() {
            format!("slots[\"{}\"]", rust_str(&input_const_name(plan)))
        } else {
            format!("slots[\"{}\"]", slot_of(plan, l.preds[0]))
        };
        s.push_str(&format!(
            "        {{ let out = self.l{i}.forward(&{in_expr}, rows);\n          slots.insert(\"{}\", out); }}\n",
            slot_of(plan, i)
        ));
    }
    let out_slot = slot_of(plan, plan.out_idx);
    s.push_str(&format!(
        "        slots.remove(\"{out_slot}\").expect(\"出口槽\")\n    }}\n}}\n"
    ));

    // ---- 4. main：跑一遍并**用引擎那个打印函数**输出 ----
    s.push_str(&main_fn(plan, consts_dir, opts.bench_iters));
    Ok(s)
}

fn input_const_name(plan: &Plan) -> String {
    // `input_name` 与 vmlgen 同规则：模型名的 ASCII 安全形式 + `_in`
    let n = &plan.model;
    if !n.is_empty()
        && n.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !n.starts_with("st_")
        && !n.starts_with("g_")
    {
        format!("{n}_in")
    } else {
        "model0_in".to_string()
    }
}

fn slot_of(plan: &Plan, i: usize) -> String {
    format!("{}_out", plan.slot_base[i])
}

fn header(plan: &Plan, dir: &str) -> String {
    format!(
        "//! 由 sparrow 生成：`{model}` 的 Rust AOT 后端（--backend lib --lang rust）。\n\
         //!\n\
         //! 算术与 `.vmb` 后端**同源**：GEMM/归一化/激活/SOFTMAX 直接调用\n\
         //! `vmlm_ops::kernels::*` 里**同一个函数**，其余逐元素/搬运算子按 `interp.rs`\n\
         //! 的同一条标量规则手写 ⇒ 两条后端输出**逐字符一致**。\n\
         //! 权重目录：`{dir}`（`include_bytes!` 用的是绝对路径）。\n\
         #![allow(clippy::needless_range_loop)]\n         #![allow(non_upper_case_globals)]\n\n\
         use vmlm_ops::kernels;\n\n",
        model = plan.model,
        dir = dir
    )
}

const LOADERS: &str = r#"/// 小端字节 → f32（逐元素读，不做对齐假设）。
fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// bf16 → f32：**精确加宽**（高 16 位左移 16），与引擎一致。
fn bf16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

/// int8 量化权重（LLM.int8 布局：`q: i8` + 每**输出行**一个 scale）。
/// 反量化在 GEMM 核里做（`kernels::gemm_nt_int8`），这里只做字节解释。
fn i8s(b: &[u8]) -> Vec<i8> {
    b.iter().map(|&x| x as i8).collect()
}

/// 汇入：逐元素相加（引擎 `OP_EWADD` 的标量规则）。
/// 多前驱**左折叠**，与 `.vml` 里生成的 `EWADD acc, acc, p` 序列同序。
fn merge_add(srcs: &[&Vec<f32>], _rows: usize, dim: u32) -> Vec<f32> {
    let _ = dim;
    let mut acc = srcs[0].clone();
    for s in &srcs[1..] {
        assert_eq!(acc.len(), s.len(), "相加要求等宽");
        for (x, y) in acc.iter_mut().zip(s.iter()) {
            *x += *y;
        }
    }
    acc
}

/// 汇入：列拼接（引擎 `OP_CONCAT`：行跨步 = ac+bc；多前驱左折叠）。
fn merge_concat(srcs: &[&Vec<f32>], rows: usize, dim: u32) -> Vec<f32> {
    let mut acc = srcs[0].clone();
    for s in &srcs[1..] {
        let ac = acc.len() / rows;
        let bc = s.len() / rows;
        let mut out = vec![0f32; rows * (ac + bc)];
        for i in 0..rows {
            out[i * (ac + bc)..i * (ac + bc) + ac].copy_from_slice(&acc[i * ac..i * ac + ac]);
            out[i * (ac + bc) + ac..(i + 1) * (ac + bc)].copy_from_slice(&s[i * bc..i * bc + bc]);
        }
        acc = out;
    }
    assert_eq!(acc.len() / rows, dim as usize, "拼接宽度与声明的输入宽度不符");
    acc
}

/// 列切片（引擎 `OP_SLICE`：`dst[i][j] = src[i][off+j]`，行跨步取自 src）。
fn slice_cols(src: &[f32], rows: usize, sc: usize, off: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * w];
    for i in 0..rows {
        for j in 0..w {
            out[i * w + j] = src[i * sc + off + j];
        }
    }
    out
}

/// 转置（引擎 `OP_TRANSPOSE`：`dst[j][i] = src[i][j]`，dst 行跨步 = m）。
fn transpose(src: &[f32], m: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * c];
    for i in 0..m {
        for j in 0..c {
            out[j * m + i] = src[i * c + j];
        }
    }
    out
}
"#;

fn layer_struct(i: usize, l: &PlanLayer, consts: &[crate::vmlgen::ConstDecl]) -> Result<String> {
    let (decls, inits) = fields_of(l, consts);
    let mut s = format!("\n/// 层 `{}`（{}）。\nstruct L{i} {{\n{decls}}}\n\n", l.orig, l.kind);
    s.push_str(&format!("impl L{i} {{\n"));
    s.push_str("    fn new() -> Self {\n        Self {\n");
    for f in &inits {
        s.push_str(&format!("            {f},\n"));
    }
    s.push_str("        }\n    }\n\n");
    s.push_str(&forward_fn(l, consts)?);
    s.push_str("}\n");
    Ok(s)
}

/// 循环组的 Rust 承载：**一份代码 + N 份常量**。
///
/// 为什么不是"N 个 struct"：那正是展开模式（`--loop-repeat` 关掉时的行为），
/// 而循环化的意义就是源码长度与 N 无关 —— N=1000 时展开版会生成上千个 struct。
/// 这里生成 `G{g}Inst`（代码一份）+ `G{g} { items: Vec<G{g}Inst> }`（常量 N 份）。
fn group_struct(
    g: usize,
    grp: &crate::vmlgen::PlanGroup,
    plan: &Plan,
) -> Result<String> {
    let consts = &plan.consts;
    let first = &plan.layers[grp.insts[0]];
    let n = grp.insts.len();
    let (decls, _) = fields_of(first, consts);
    let mut s = format!(
        "\n/// 循环组 `{}`：**{n} 份**模板（源自第 {} 层那个 `重复`）。\n\
         /// 与 `.vmb` 的 `LOOP_CNT`+`LAYERB` 同一语义：进位槽 = 上一圈的输出，\n\
         /// 每份只差常量（各自 `include_bytes!`）。\n\
         struct G{g}Inst {{\n{decls}}}\n\n",
        first.orig, grp.insts[0]
    );
    s.push_str(&format!("impl G{g}Inst {{\n"));
    s.push_str(&forward_fn(first, consts)?);
    s.push_str("}\n\n");
    s.push_str(&format!(
        "/// {n} 份实例的常量表（第 1 份是模板，其余只是常量不同）。\nstruct G{g} {{\n    items: Vec<G{g}Inst>,\n}}\n\n"
    ));
    s.push_str(&format!("impl G{g} {{\n    fn new() -> Self {{\n        Self {{ items: vec![\n"));
    for &m in &grp.insts {
        let (_, inits) = fields_of(&plan.layers[m], consts);
        s.push_str(&format!("            G{g}Inst {{ {} }},\n", inits.join(", ")));
    }
    s.push_str("        ] }\n    }\n}\n");
    Ok(s)
}

/// 循环组在 `Model::forward` 里的循环体。
///
/// 逐条对齐文本后端（`vmlgen::emit_loop_group`）：
/// * 进位槽 = 上一圈输出；`for` 的每一圈读**本圈实例**的常量；
/// * 有外部前驱时，每圈都汇入一次，且**操作数顺序必须是 `[进位, 外部前驱…]`**
///   —— 文本后端就是这么排的（拼接不可交换，顺序反了就与 `.vmb` 不一致）；
/// * 汇入结果落在循环内局部变量里，循环结束后写进组的**最后一个实例**的输出槽
///   （文本后端用 `<最后一份>_out` 当循环体槽）。
fn group_loop(g: usize, plan: &Plan) -> Result<String> {
    let grp = &plan.groups[g];
    let first = &plan.layers[grp.insts[0]];
    let seed_slot = slot_of(plan, grp.seed);
    let out_slot = slot_of(plan, *grp.insts.last().expect("组至少有 1 份"));
    let n = grp.insts.len();
    let mut s = String::new();
    s.push_str(&format!(
        "        // ---- 重复组 `{}`：{n} 份模板（循环版；与 .vmb 的 LOOP_CNT+LAYERB 同一语义）----\n",
        first.orig
    ));
    s.push_str(&format!(
        "        let mut carry: Vec<f32> = slots[\"{seed_slot}\"].to_vec();\n"
    ));
    s.push_str(&format!("        for inst in &self.g{g}.items {{\n"));
    if grp.ext.is_empty() {
        s.push_str("            carry = inst.forward(&carry, rows);\n");
    } else {
        let srcs: Vec<String> = std::iter::once("&carry".to_string())
            .chain(grp.ext.iter().map(|&p| format!("&slots[\"{}\"]", slot_of(plan, p))))
            .collect();
        let mode = match grp.merge {
            MergeMode::Add => "add",
            MergeMode::Concat => "concat",
        };
        s.push_str(&format!(
            "            let merged = merge_{mode}(&[{}], rows, {});\n",
            srcs.join(", "),
            grp.in_dim
        ));
        s.push_str("            carry = inst.forward(&merged, rows);\n");
    }
    s.push_str("        }\n");
    s.push_str(&format!("        slots.insert(\"{out_slot}\", carry);\n"));
    Ok(s)
}

/// 层 `i` 属于哪个循环组（不属于则 `None`）。
///
/// 判据是「在某个 `PlanGroup::insts` 里」，**不是** `PlanLayer::group < groups.len()`
/// —— 后者对非 `重复` 的层也成立（`Layer::group` 是每层自增的计数器，见 vmlgen 的注释）。
fn group_of(plan: &Plan, i: usize) -> Option<usize> {
    plan.groups.iter().position(|g| g.insts.contains(&i))
}

/// 层的字段**声明文本**与**初始化表达式**（`self.w: f32s(C_x_W_RAW)`）。
///
/// 循环组的每份实例用同一套字段名（`lower()` 取常量名最后一段），所以这两样
/// 必须能脱离 struct 名字单独取出：同一份 `forward` 代码配上 N 份初始化。
fn fields_of(l: &PlanLayer, consts: &[crate::vmlgen::ConstDecl]) -> (String, Vec<String>) {
    let dtype_of = |name: &str| -> DataType {
        consts
            .iter()
            .find(|c| c.name == name)
            .map(|c| c.dtype)
            .unwrap_or(DataType::F32)
    };
    let mut decls = String::new();
    let mut inits: Vec<String> = Vec::new();
    for (name, _file, what) in &consts_of(l, consts) {
        let id = ident(name);
        match dtype_of(name) {
            DataType::Bf16 => {
                decls.push_str(&format!("    /// {what}（bf16）\n    {}: Vec<u16>,\n", lower(name)));
                inits.push(format!("{}: bf16s({id}_RAW)", lower(name)));
            }
            DataType::Int8 => {
                decls.push_str(&format!("    /// {what}（int8）\n    {}: Vec<i8>,\n", lower(name)));
                inits.push(format!("{}: i8s({id}_RAW)", lower(name)));
            }
            DataType::F32 => {
                decls.push_str(&format!("    /// {what}（f32）\n    {}: Vec<f32>,\n", lower(name)));
                inits.push(format!("{}: f32s({id}_RAW)", lower(name)));
            }
        }
    }
    (decls, inits)
}

/// `fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> { … }`（含签名与 doc）。
fn forward_fn(l: &PlanLayer, consts: &[crate::vmlgen::ConstDecl]) -> Result<String> {
    let dtype_of = |name: &str| -> DataType {
        consts
            .iter()
            .find(|c| c.name == name)
            .map(|c| c.dtype)
            .unwrap_or(DataType::F32)
    };
    let mut s = String::new();
    let rows = l.out_dim as usize;
    // EMBED 的"输入"是词元 id（vmlgen 给的 in_dim = 0），实际只有 1 个数
    let ind = if l.kind == "EMBED" { 1usize } else { l.in_dim as usize };

    // ---- forward ----
    s.push_str(&format!(
        "    /// 输入 `[{rows_or_t}, {ind}]` → 输出 `[rows, {rows}]`。\n",
        rows_or_t = "T"
    ));
    s.push_str("    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {\n");
    s.push_str(&format!("        let (ind, outd) = ({ind}usize, {rows}usize);\n"));
    s.push_str("        assert_eq!(x.len(), rows * ind, \"输入形状\");\n");
    match l.kind.as_str() {
        "EMBED" => {
            // 逐行查表（V0.1.5）：`序列=T` 时链首是嵌入，输入就是 T 个词元 id（`[1,T]`）。
            // rows == 1 时这段与 V0.1.4 的单行版本逐字节等价（同一份表行、同样的校验）。
            match dtype_of(&format!("{}_E", l.base)) {
                // 嵌入表不吃 int8：前端在 `--weight-dtype int8` 下把它回落成 f32
                //（`EMB` 的表必须是 f32/bf16）。真出现就是生成端 bug ⇒ 让**生成的 crate
                // 编译失败**并打出这句话，而不是悄悄按 f32 读一份 int8 字节。
                DataType::Int8 => s.push_str(
                    "        compile_error!(\"嵌入表不支持 int8（请用 f32/bf16）\");\n",
                ),
                DataType::Bf16 => s.push_str(
                    "        let mut out = vec![0f32; rows * outd];\n\
                     \x20       let e = &self.e;\n\
                     \x20       let c = e.len() / (e.len() / outd).max(1);\n\
                     \x20       let nrow = e.len() / c;\n\
                     \x20       for k in 0..rows {\n\
                     \x20           let rowf = x[k];\n\
                     \x20           assert!(rowf >= 0.0, \"EMB: row index out of range\");\n\
                     \x20           let row = rowf as usize;\n\
                     \x20           assert!(row as f32 == rowf, \"EMB: row index is not an integer\");\n\
                     \x20           assert!(row < nrow, \"EMB: row index out of range\");\n\
                     \x20           for j in 0..c { out[k * c + j] = f32::from_bits((e[row * c + j] as u32) << 16); }\n\
                     \x20       }\n\
                     \x20       out\n",
                ),
                DataType::F32 => s.push_str(
                    "        let mut out = vec![0f32; rows * outd];\n\
                     \x20       let e = &self.e;\n\
                     \x20       let c = outd;\n\
                     \x20       let nrow = e.len() / c;\n\
                     \x20       for k in 0..rows {\n\
                     \x20           // 行号是 f32（引擎会校验：非整数 / 越界都失败）\n\
                     \x20           let rowf = x[k];\n\
                     \x20           assert!(rowf >= 0.0, \"EMB: row index out of range\");\n\
                     \x20           let row = rowf as usize;\n\
                     \x20           assert!(row as f32 == rowf, \"EMB: row index is not an integer\");\n\
                     \x20           assert!(row < nrow, \"EMB: row index out of range\");\n\
                     \x20           out[k * c..k * c + c].copy_from_slice(&e[row * c..row * c + c]);\n\
                     \x20       }\n\
                     \x20       out\n",
                ),
            }
        }
        "SPARSE" => {
            s.push_str(
                "        // 与 `interp.rs` 的 OP_SPARSE_MATVEC 同一套标量规则：\n\
                 \x20       // 首项 `acc = term`，其后 `acc += term`；空 span 写 0.0。\n\
                 \x20       let (off, src, w) = (&self.off, &self.src, &self.w);\n\
                 \x20       let n = off.len() - 1;\n\
                 \x20       assert_eq!(n, outd, \"offsets 必须是 n_neurons+1 条\");\n\
                 \x20       let nnz = src.len();\n\
                 \x20       assert_eq!(nnz, w.len());\n\
                 \x20       let mut col = vec![0f32; n];\n\
                 \x20       for j in 0..n {\n\
                 \x20           let (o0, o1) = (off[j] as usize, off[j + 1] as usize);\n\
                 \x20           assert!(o1 <= nnz && o0 <= o1, \"offsets out of range\");\n\
                 \x20           let mut acc = 0.0f32;\n\
                 \x20           for t in o0..o1 {\n\
                 \x20               let sid = src[t] as usize;\n\
                 \x20               assert!(sid < ind, \"src_id out of range\");\n\
                 \x20               let term = w[t] * x[sid];\n\
                 \x20               if t == o0 { acc = term; } else { acc += term; }\n\
                 \x20           }\n\
                 \x20           col[j] = acc;\n\
                 \x20       }\n\
                 \x20       // 引擎接下来一条 TRANSPOSE 把它变成行向量\n\
                 \x20       col\n",
            );
        }
        "LINEAR" => {
            let is_int8 = consts.iter().any(|c| c.name == format!("{}_Wq", l.base));
            let bf = dtype_of(&format!("{}_W", l.base)) == DataType::Bf16;
            s.push_str("        let mut out = vec![0f32; rows * outd];\n");
            if is_int8 {
                // int8：**反量化在核内**（`q: i8` + 每输出行一个 `scale`）。
                // 与 f32 路径的差别只有两处：权重字节数（1 vs 4 B/元素）与
                // **累加顺序**（不同内核 ⇒ 即使权重可被 int8 精确表示，和也可能差 1 ULP）。
                // 所以判据不是"逐字符相同"，而是"差值在实测 ULP 界内 + can-fail 必须超出它"，
                // 见 REPORT_V015 §6。
                s.push_str(
                    "        unsafe { kernels::gemm_nt_int8(x.as_ptr(), self.wq.as_ptr(), self.scale.as_ptr(), out.as_mut_ptr(), rows, outd, ind) };\n",
                );
            } else if bf {
                s.push_str(
                    "        unsafe { kernels::gemm_nt_bf16(x.as_ptr(), self.w.as_ptr(), out.as_mut_ptr(), rows, outd, ind) };\n",
                );
            } else {
                s.push_str(
                    "        unsafe { kernels::gemm_nt(x.as_ptr(), self.w.as_ptr(), out.as_mut_ptr(), rows, outd, ind, false) };\n",
                );
            }
            s.push_str(
                "        // BIAS（引擎 `OP_BIAS`：`dst[i][j] += bias[j]`）\n\
                 \x20       for i in 0..rows { for j in 0..outd { out[i * outd + j] += self.b[j]; } }\n\
                 \x20       out\n",
            );
        }
        "NORM" => {
            s.push_str(
                "        let mut out = x.to_vec();\n\
                 \x20       // 引擎的 LN：copy a→d 后 `layernorm(d, g, bias, rows, cols, ln_eps)`；\n\
                 \x20       // eps 默认 1e-5（没有 LN_EPS 指令时）\n\
                 \x20       unsafe { kernels::layernorm(out.as_mut_ptr(), self.g.as_ptr(), self.b.as_ptr(), rows, outd, 1e-5) };\n\
                 \x20       out\n",
            );
        }
        "ACT" => {
            let kind = act_kind(&l.attrs);
            s.push_str(&format!(
                "        let mut out = x.to_vec();\n\
                 \x20       unsafe {{ kernels::act_avx2(out.as_mut_ptr(), rows * outd, {kind}) }};\n\
                 \x20       out\n"
            ));
        }
        "ATTN" => {
            let is_int8 = consts
                .iter()
                .any(|c| c.name == format!("{}_Wq", l.base) && c.dtype == DataType::Int8);
            s.push_str(&attn_body(l, is_int8)?);
        }
        other => {
            return Err(Diag::error(
                "E109",
                format!("`--backend lang rust` 还不支持层类型 `{other}`"),
                crate::ast::Span::ZERO,
            ))
        }
    }
    // 只收 `forward` 本身；`impl` 的闭合 `}` 由调用方（layer_struct / group_struct）加，
    // 因为循环组是"一份代码 + N 份常量"，同一个 forward 要能挂到两种承载上。
    s.push_str("    }\n");
    Ok(s)
}

fn act_kind(attrs: &[(String, crate::ast::Expr)]) -> u16 {
    for (k, v) in attrs {
        if k == "act" {
            if let crate::ast::Expr::Ident(s) = v {
                return match crate::keywords::norm(s).as_str() {
                    "gelu" => 1,
                    "silu" | "swish" => 2,
                    "sigmoid" => 3,
                    "tanh" => 4,
                    "relu2" => 5,
                    _ => 0, // relu（含未识别 → 引擎的默认也是 relu）
                };
            }
        }
    }
    0
}

fn attn_body(l: &PlanLayer, int8: bool) -> Result<String> {
    let heads = l
        .attrs
        .iter()
        .find(|(k, _)| k == "heads")
        .and_then(|(_, v)| match v {
            crate::ast::Expr::Int(n) => Some(*n as u32),
            _ => None,
        })
        .unwrap_or(1);
    let causal = l.attrs.iter().any(|(k, v)| {
        k == "causal"
            && matches!(v, crate::ast::Expr::Ident(s) if !matches!(crate::keywords::norm(s).as_str(), "否"|"不"|"假"|"无"|"关"|"0"|"false"|"no"|"off"|"n"))
            || matches!(v, crate::ast::Expr::Int(n) if *n != 0)
    });
    let d = l.out_dim as usize;
    if d % heads as usize != 0 {
        return Err(Diag::error(
            "E108",
            format!("注意力层 `{}`：维度 {d} 不能被头数 {heads} 整除", l.orig),
            crate::ast::Span::ZERO,
        ));
    }
    let hd = d / heads as usize;
    let mut s = String::new();
    s.push_str(&format!(
        "        // 注意力：D={d} 头={heads} 每头={hd}{}\n",
        if causal { " 因果掩码" } else { "" }
    ));
    s.push_str("        let mut q = vec![0f32; rows * outd];\n");
    s.push_str("        let mut k = vec![0f32; rows * outd];\n");
    s.push_str("        let mut v = vec![0f32; rows * outd];\n");
    for (dst, w, b) in [("q", "wq", "bwq"), ("k", "wk", "bwk"), ("v", "wv", "bwv")] {
        let bf = "";
        let _ = bf;
        if int8 {
            // 与 LINEAR 的 int8 同一条路：反量化在核内（`q: i8` + 每输出行一个 scale）。
            // 四个投影都是 `gemm_nt` 形态，所以共用同一个内核。
            //
            // 字段名为什么是 `s{w}`（`swq`）而不是 `{w}_scale`：`lower()` 取常量名**最后一段**
            // （`a__2_bWq` → `bwq`），所以区分用的后缀必须在最后 —— `ATTN_Wq_scale` 会被压成
            // `scale`，与 softmax 的 `ATTN_scale` 撞成同一个字段（实测 E0124/E0609）。
            s.push_str(&format!(
                "        unsafe {{ kernels::gemm_nt_int8(x.as_ptr(), self.{w}.as_ptr(), self.s{w}.as_ptr(), {dst}.as_mut_ptr(), rows, outd, ind) }};\n\
                 \x20       for i in 0..rows {{ for j in 0..outd {{ {dst}[i * outd + j] += self.{b}[j]; }} }}\n"
            ));
        } else {
            s.push_str(&format!(
                "        unsafe {{ kernels::gemm_nt(x.as_ptr(), self.{w}.as_ptr(), {dst}.as_mut_ptr(), rows, outd, ind, false) }};\n\
                 \x20       for i in 0..rows {{ for j in 0..outd {{ {dst}[i * outd + j] += self.{b}[j]; }} }}\n"
            ));
        }
    }
    s.push_str("        let scale = self.scale[0];\n");
    s.push_str("        let mut heads_out: Vec<Vec<f32>> = Vec::with_capacity(0);\n");
    for h in 0..heads as usize {
        let off = h * hd;
        s.push_str(&format!(
            "        let qh = slice_cols(&q, rows, outd, {off}, {hd});\n\
             \x20       let kh = slice_cols(&k, rows, outd, {off}, {hd});\n\
             \x20       let vh = slice_cols(&v, rows, outd, {off}, {hd});\n\
             \x20       let kt = transpose(&kh, rows, {hd});\n\
             \x20       let mut sc = vec![0f32; rows * rows];\n\
             \x20       unsafe {{ kernels::gemm(qh.as_ptr(), kt.as_ptr(), sc.as_mut_ptr(), rows, rows, {hd}, false) }};\n\
             \x20       for p in sc.iter_mut() {{ *p *= scale; }}\n\
             \x20       unsafe {{ kernels::softmax(sc.as_mut_ptr(), rows, rows, {causal}) }};\n\
             \x20       let mut ch = vec![0f32; rows * {hd}];\n\
             \x20       unsafe {{ kernels::gemm(sc.as_ptr(), vh.as_ptr(), ch.as_mut_ptr(), rows, {hd}, rows, false) }};\n\
             \x20       heads_out.push(ch);\n"
        ));
    }
    s.push_str(&format!(
        "        let mut ctx = vec![0f32; rows * outd];\n\
         \x20       for i in 0..rows {{\n\
         \x20           for (hi, ch) in heads_out.iter().enumerate() {{\n\
         \x20               ctx[i * outd + hi * {hd}..i * outd + (hi + 1) * {hd}]\n\
         \x20                   .copy_from_slice(&ch[i * {hd}..(i + 1) * {hd}]);\n\
         \x20           }}\n\
         \x20       }}\n\
         \x20       let mut out = vec![0f32; rows * outd];\n"
    ));
    if int8 {
        s.push_str(
            "        unsafe { kernels::gemm_nt_int8(ctx.as_ptr(), self.wo.as_ptr(), self.swo.as_ptr(), out.as_mut_ptr(), rows, outd, outd) };\n",
        );
    } else {
        s.push_str(
            "        unsafe { kernels::gemm_nt(ctx.as_ptr(), self.wo.as_ptr(), out.as_mut_ptr(), rows, outd, outd, false) };\n",
        );
    }
    s.push_str(
        "        for i in 0..rows { for j in 0..outd { out[i * outd + j] += self.bwo[j]; } }\n\
         \x20       out\n",
    );
    Ok(s)
}

fn main_fn(plan: &Plan, _dir: &str, bench_iters: u32) -> String {
    let inp = ident(&input_const_name(plan));
    let n = plan.t_seq.max(1) as usize;
    let rows = plan.layers[plan.out_idx].out_dim as usize * n;
    // `--bench-iters 0`（默认）**不生成任何计时代码** ⇒ 输出与 V0.1.4 逐字节相同。
    let bench = if bench_iters == 0 {
        String::new()
    } else {
        format!(
            r#"
    // ---- 基准（--bench-iters {k}）：与 `vmlm-runtime-bench.exe` 同口径 ----
    // 预热一遍（排除首次分配 / 页错误），再计时 K 遍；装载与打印都在计时之外。
    // 只做**前向**，不做任何 I/O —— 否则打印会把计算盖掉（引擎那条路必须先把
    // 程序里的 PRINT 去掉，见 tools/aot_bench.ps1 的说明）。
    std::hint::black_box(model.forward(&input));
    let t0 = std::time::Instant::now();
    for _ in 0..{k}usize {{
        std::hint::black_box(model.forward(&input));
    }}
    let us = t0.elapsed().as_secs_f64() * 1e6 / {k} as f64;
    println!("avg us/iter = {{}}", us as u64);
"#,
            k = bench_iters
        )
    };
    format!(
        r#"
fn main() {{
    let model = Model {{
{inits}    }};
    let input = f32s({inp}_RAW);
    let out = model.forward(&input);
    assert_eq!(out.len(), {rows}, "输出元素个数");
    // 用**引擎那个**打印函数（它的 6 位小数是**截断**，不是四舍五入）
    unsafe {{ vmlm_ops::interp::print_f32(out.as_ptr(), out.len()) }};
{bench}}}
"#,
        inits = struct_inits(plan)
    )
}

/// `Model { … }` 的初始化表：非组层各一个 `l{i}`，每个循环组一个 `g{g}`。
/// 组内实例不单独建 struct（它们由 `G{g}Inst` 的常量表承载）。
fn struct_inits(plan: &Plan) -> String {
    let mut s = String::new();
    for i in 0..plan.layers.len() {
        if group_of(plan, i).is_none() {
            s.push_str(&format!("        l{i}: L{i}::new(),\n"));
        }
    }
    for g in 0..plan.groups.len() {
        s.push_str(&format!("        g{g}: G{g}::new(),\n"));
    }
    s
}
