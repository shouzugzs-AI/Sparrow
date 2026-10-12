// 汇编器：.vml 文本 → .vmb 二进制
// 这是「自由导入模型逻辑」的工具层——新架构 = 新 .vml 脚本，引擎不变。
// 体积不受 30KB 约束（决策 5），可以随便用 std。
use std::collections::HashMap;
use vmlm_core::format::*;
use vmlm_core::interp::Ins;

/// 「格式里有、引擎里没有」的 opcode 清单（P10 普查结果）。
///
/// 这些名字必须**在汇编期就拒绝**，否则会产出「能装载、跑到一半 exit(5)」
/// 的 .vmb —— 使用者按白皮书 §3.4 的表写代码，却在运行时收到一个
/// 不指名道姓的 `unimpl op`。这是最典型的静默降级形态。
///
/// 维护方式：`python tools/p10_opcode_gaps.py` 会重新扫描 format.rs 与
/// interp.rs 的差集。**实现了某条指令后，记得把它从这里删掉**，
/// 否则汇编器会拒绝一条其实已经能用的指令（这个方向是安全失败）。
///
/// RMSNORM 原本在名单里，P10 补上了实现（MiniCPM 必需），故已移除。
const UNIMPLEMENTED_OPS: &[(&str, &str)] = &[
    ("LOADK", "常量搬运未实现；用 COPY 或直接引用常量槽"),
    ("RESHAPE", "形状重解释未实现；用 SHAPE 声明 + COPY"),
    ("MEAN", "行均值未实现；用 SUM 再 SCALE"),
    // MAX 与 TOPK_IDX 在 O6-A 里补上了引擎实现，故从这里移除。
    // 留着会让汇编器拒绝一条**已经能用**的指令 —— 方向上是安全失败，
    // 但它是 O6（自回归生成）的唯一阻塞点：decode 的 argmax 要喂回 st_gen。
    // ❗ 移除前必须先确认 interp.rs 里真有分支；清单由
    //    `python tools/p10_opcode_gaps.py` 复现（它现在应不再列出这两条）。
    ("NORM", "通用范数未实现；用 SQRT(SUM(x^2)) 展开"),
    ("GATHER", "按索引取值未实现"),
    ("SCATTER", "按索引写值未实现"),
    ("BRANCH", "条件跳转未实现；用 CMPLE（它已有实现）"),
    ("OP_CALL", "子程序调用未实现；用内联展开或 JMP/LOOP_CNT"),
];

struct Const {
    name: String,
    rows: u32,
    cols: u32,
    dtype: u8,
    data: Vec<u8>,
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Const,
    Act,
    State,
    Grad,
}

struct SlotDef {
    name: String,
    kind: Kind,
    rows: u32,
    cols: u32,
}

struct Asm {
    consts: Vec<Const>,
    text: Vec<Ins>,
    labels: HashMap<String, u32>,
    fixups: Vec<(usize, String)>,
    slots: Vec<SlotDef>,
    by_name: HashMap<String, u32>,
    slice_w: Vec<(u32, u32)>,
    pinned: std::collections::HashSet<u32>,
    // P10: LOOP_CNT 指令的索引，用于 resolve 后校验「目标必须向后」
    loop_targets: Vec<usize>,
}

impl Asm {
    fn new() -> Asm {
        Asm {
            consts: Vec::new(),
            text: Vec::new(),
            labels: HashMap::new(),
            fixups: Vec::new(),
            slots: Vec::new(),
            by_name: HashMap::new(),
            slice_w: Vec::new(),
            pinned: std::collections::HashSet::new(),
            loop_targets: Vec::new(),
        }
    }

    fn const_idx(&self, name: &str) -> Option<u32> {
        self.consts.iter().position(|c| c.name == name).map(|i| i as u32)
    }

    /// True when the slot at index `slot` is a constant of dtype `want`.
    ///
    /// Used to validate that a GEMM's 4th operand (the int8 scale slot) is only
    /// supplied when the weight really is int8 -- otherwise `dt` would carry an
    /// arbitrary slot number into the kernel, which the engine would then use as
    /// a scale pointer. SlotDef does not carry the const index, so the lookup
    /// goes through the slot NAME (which is how consts are keyed anyway).
    fn slot_is_const_dtype(&self, slot: u32, want: u8) -> bool {
        let sd = match self.slots.get(slot as usize) {
            Some(s) => s,
            None => return false,
        };
        if sd.kind != Kind::Const {
            return false;
        }
        match self.const_idx(&sd.name) {
            Some(ci) => self.consts[ci as usize].dtype == want,
            None => false,
        }
    }

    // 取槽位号：常量名优先匹配常量表；st_ 前缀为持久状态槽；其余是激活槽
    fn slot(&mut self, name: &str) -> u32 {
        if let Some(&s) = self.by_name.get(name) {
            return s;
        }
        let (kind, rows, cols) = if let Some(ci) = self.const_idx(name) {
            let c = &self.consts[ci as usize];
            (Kind::Const, c.rows, c.cols)
        } else if let Some(gname) = name.strip_prefix("g_") {
            // g_<name>：<name> 的梯度槽，形状跟随对应常量。
            //
            // 这里**必须硬报错**而不是静默给 (0,0)：
            // 之前静默的行为会让 `g_r`（本意是激活）变成一个 0x0 的梯度槽，
            // 后面所有写它的指令都变成零长度操作 —— 模型静默错掉，
            // 只在端到端数值对比时才发现。成员 A2 踩过这个坑。
            let (r, cc) = match self.const_idx(gname) {
                Some(ci) => (self.consts[ci as usize].rows, self.consts[ci as usize].cols),
                None => panic!(
                    "`{}` 被当作梯度槽，但找不到对应常量 `{}`。\
                     如果你本想用它作普通激活槽，请改名（不要用 g_ 前缀）。",
                    name, gname
                ),
            };
            (Kind::Grad, r, cc)
        } else if let Some(rest) = name.strip_prefix("st_") {
            let _ = rest;
            (Kind::State, 0, 0)
        } else {
            (Kind::Act, 0, 0)
        };
        let s = self.slots.len() as u32;
        self.slots.push(SlotDef {
            name: name.to_string(),
            kind,
            rows,
            cols,
        });
        self.by_name.insert(name.to_string(), s);
        s
    }

    fn set_shape(&mut self, slot: u32, rows: u32, cols: u32) {
        // pinned 槽位的形状来自显式 SLICE 宽度或 SHAPE 声明，是权威值，
        // 自动传播不得覆盖（否则 qc=8x16 会被推回 8x48）。
        if self.pinned.contains(&slot) {
            return;
        }
        let sd = &mut self.slots[slot as usize];
        if sd.kind != Kind::Const {
            if rows > 0 {
                sd.rows = rows;
            }
            if cols > 0 {
                sd.cols = cols;
            }
        }
    }

    // 无条件写形状（用于显式声明）
    fn force_shape(&mut self, slot: u32, rows: u32, cols: u32) {
        let sd = &mut self.slots[slot as usize];
        if sd.kind != Kind::Const {
            sd.rows = rows;
            sd.cols = cols;
            self.pinned.insert(slot);
        }
    }

    fn opcode(name: &str) -> Option<u8> {
        Some(match name {
            "HALT" => OP_HALT,
            "LOADK" => OP_LOADK,
            "COPY" => OP_COPY,
            "RESHAPE" => OP_RESHAPE,
            "TRANSPOSE" => OP_TRANSPOSE,
            "CONCAT" => OP_CONCAT,
            "SLICE" => OP_SLICE,
            "GEMM" => OP_GEMM,
            "BMM" => OP_BMM,
            "OUTER" => OP_OUTER,
            "GEMV" => OP_GEMV,
            "EMB" => OP_EMB,
            // P13 task E: advance the runtime constant base (layer loop).
            "LAYERB" => OP_LAYERB,
            "EWADD" => OP_EWADD,
            "EWMUL" => OP_EWMUL,
            "EWSUB" => OP_EWSUB,
            "EWDIV" => OP_EWDIV,
            "SCALE" => OP_SCALE,
            "ACT" => OP_ACT,
            "EXP" => OP_EXP,
            "LOG" => OP_LOG,
            "CLAMP" => OP_CLAMP,
            "NEG" => OP_NEG,
            "LN" => OP_LN,
            "RMSNORM" => OP_RMSNORM,
            "SUM" => OP_SUM,
            "MEAN" => OP_MEAN,
            "MAX" => OP_MAX,
            "NORM" => OP_NORM,
            "SOFTMAX" => OP_SOFTMAX,
            "MASK_MUL" => OP_MASK_MUL,
            "TOPK_THRESH" => OP_TOPK_THRESH,
            "TOPK_IDX" => OP_TOPK_IDX,
            "GATHER" => OP_GATHER,
            "SCATTER" => OP_SCATTER,
            "BRANCH" => OP_BRANCH,
            "JMP" => OP_JMP,
            "CMPLE" => OP_CMPLE,
            "OP_CALL" => OP_OP_CALL,
            "SAVE" => OP_SAVE,
            "LOAD" => OP_LOAD,
            "PRINT" => OP_PRINT,
            "SHAPE" => OP_SHAPE,
            "BIAS" => OP_BIAS,
            // 反向指令（P3）
            "GEMM_BWD_W" => OP_GEMM_BWD_W,
            "GEMM_BWD_X" => OP_GEMM_BWD_X,
            "BIAS_BWD" => OP_BIAS_BWD,
            "ACT_BWD" => OP_ACT_BWD,
            "CE_BWD" => OP_CE_BWD,
            "MSE_BWD" => OP_MSE_BWD,
            "ZERO" => OP_ZERO,
            "ADAM" => OP_ADAM,
            "SGD" => OP_SGD,
            "LOSS_ADD" => OP_LOSS_ADD,
            "CLIP_GRAD" => OP_CLIP_GRAD,
            "WRITECOL" => OP_WRITECOL,
            "EWMAX" => OP_EWMAX,
            "GEMASK" => OP_GEMASK,
            // P7 反向算子扩展（编号已冻结，见 format.rs P7 段落）
            "SOFTMAX_BWD" => OP_SOFTMAX_BWD,
            "LN_BWD" => OP_LN_BWD,
            "OUTER_BWD" => OP_OUTER_BWD,
            "EMB_BWD" => OP_EMB_BWD,            "ADAMW" => OP_ADAMW,
            "LN_EPS" => OP_LN_EPS,
            "PUSH" => OP_PUSH,
            "POP" => OP_POP,
            "STACK_INIT" => OP_STACK_INIT,
            // P10 运行时循环 + RoPE
            "LOOP_CNT" => OP_LOOP_CNT,
            "ROPE" => OP_ROPE,
            "SIN" => OP_SIN,
            "COS" => OP_COS,
            // P16 连接稀疏：top-k 选连接 + 只算存活连接的稀疏矩阵向量乘。
            //
            // ⚠️ 这两条**必须**同时出现在 opcode()、UNIMPLEMENTED_OPS 的补集
            // 与 infer_shapes() 三处。本项目被「定义了名字、引擎却没实现」
            // 咬过（OP_MAX / OP_TOPK_IDX 定义了很多阶段都没有引擎分支），
            // 也被反过来的「引了名字、形状没推断 ⇒ 输出槽停在 0x0 ⇒
            // 结果被静默丢弃」咬过。两端的堵法见 interp.rs 与 infer_shapes。
            "TOPK_BY_STRENGTH" => OP_TOPK_BY_STRENGTH,
            "SPARSE_MATVEC" => OP_SPARSE_MATVEC,
            _ => return None,
        })
    }

    // 该指令的「裸 flag 名」如何解释（不占操作数位）。
    // causal 只对前向 SOFTMAX 和反向 SOFTMAX_BWD 有意义。
    fn takes_flag(op: u8, name: &str) -> bool {
        match name {
            "causal" => op == OP_SOFTMAX || op == OP_SOFTMAX_BWD,
            "acc" | "tb" | "nn" | "bias" | "bc" | "neg" => {
                op == OP_GEMM || op == OP_GEMV || op == OP_BMM || op == OP_OUTER
            }
            "relu" | "relu2" | "gelu" | "silu" | "sigmoid" | "tanh" => {
                op == OP_ACT || op == OP_ACT_BWD
            }
            // P16: `nnz` 只对 SPARSE_MATVEC 有意义 —— 让引擎把实际执行的
            // 乘加次数写进 dst[n_neurons]，作为「零连接被跳过」的可测证据。
            "nnz" => op == OP_SPARSE_MATVEC,
            // P16-5: `selgate` 只对 SPARSE_MATVEC 有意义 —— 打开它之后
            // 选择表 sel（src 的紧邻槽）才真正参与求和，k 于是影响 y。
            "selgate" => op == OP_SPARSE_MATVEC,
            // P16-5: TOPK_BY_STRENGTH 上同一个 bit7 表示**分段模式**（逐行
            // top-k）。两位同号不同义是安全的：一个 bit 只被拥有它的 opcode
            // 读取，而这两条指令的 fl 语义本来就各自独立。
            "seg" => op == OP_TOPK_BY_STRENGTH,
            _ => false,
        }
    }

    fn flag(name: &str) -> Option<u16> {
        Some(match name {
            "acc" => FL_ACC,
            "tb" => FL_TRANS_B,
            // GEMM 的 B 布局：默认 NT（B 是 [N,K] 权重，内部转置）；
            // `nn` 切到 NN，把 B 按 [K,N] 解释（注意力 att @ v 用这条）。
            "nn" => FL_TRANS_B,
            "bias" => FL_BIAS,
            "bc" => FL_BROADCAST,
            "causal" => FL_CAUSAL,
            "neg" => FL_NEG,
            // P16: 把「实际乘加次数」写进 dst[n_neurons]。bit6 —— 与
            // SPARSE_MATVEC 编码权重槽号的低 6 位不重叠（见 format.rs）。
            "nnz" => FL_NNZ as u16,
            // P16-5: 打开选择表门控。bit7 —— 三处占用（低 6 位槽号 /
            // bit6 NNZ / bit7 SELGATE）互不重叠，见 format.rs FL_SELGATE。
            "selgate" => FL_SELGATE as u16,
            // P16-5: TOPK_BY_STRENGTH 的分段（逐行）模式，同一个 bit7。
            "seg" => FL_SELGATE as u16,
            "relu" => ACT_RELU,
            "gelu" => ACT_GELU,
            "silu" => ACT_SILU,
            "sigmoid" => ACT_SIGMOID,
            "tanh" => ACT_TANH,
            "relu2" => ACT_RELU2,
            _ => return None,
        })
    }

    fn assemble(&mut self, src: &str) {
        for raw in src.lines() {
            let line = match raw.find(';') {
                Some(i) => &raw[..i],
                None => raw,
            };
            let line = line.trim();
            if line.is_empty() || line.starts_with('.') {
                continue;
            }
            if let Some(l) = line.strip_suffix(':') {
                let idx = self.text.len() as u32;
                self.labels.insert(l.trim().to_string(), idx);
                continue;
            }
            self.emit_line(line);
        }
    }

    // 形状推断：SHAPE 显式声明优先；未声明的槽位从指令语义推出来，
    // 省得脚本里把每个临时槽都写一遍（也避免漏写导致 0x0 槽位）。
    fn infer_shapes(&mut self) {
        // 先应用显式 SHAPE
        for i in 0..self.text.len() {
            let ins = self.text[i];
            if ins.op == OP_SHAPE {
                self.force_shape(ins.dst, ins.a, ins.b);
            }
        }
        // 应用 SLICE 的显式宽度需要先知道 rows，所以放在传播之后（见下）。
        // 先迭代传播（指令流是 DAG，几轮足够）
        for _ in 0..4 {
            for i in 0..self.text.len() {
                let ins = self.text[i];
                let (d, a, b) = (ins.dst, ins.a, ins.b);
                match ins.op {
                    // dst[M,N] = a[M,K] · b[N,K]^T（NT）或 a[M,K] · b[K,N]（NN）
                    OP_GEMM => {
                        let (m, _k) = self.shape_of(a);
                        let (b_rows, b_cols) = self.shape_of(b);
                        let nn_out = if (ins.fl as u16) & FL_TRANS_B == 0 {
                            b_rows
                        } else {
                            b_cols
                        };
                        if m > 0 && nn_out > 0 {
                            self.set_shape(d, m, nn_out);
                        }
                    }
                    // 逐元素：dst 跟随 a
                    OP_EWADD | OP_EWMUL | OP_EWSUB | OP_EWDIV | OP_ACT | OP_EXP | OP_NEG
                    | OP_LOG | OP_CLAMP | OP_COPY => {
                        let (r, c) = self.shape_of(a);
                        if r > 0 && c > 0 {
                            self.set_shape(d, r, c);
                        }
                    }
                    // BIAS：dst 保持 a 的形状
                    OP_BIAS => {
                        let (r, c) = self.shape_of(d);
                        if r == 0 {
                            let (r2, c2) = self.shape_of(a);
                            self.set_shape(d, r2, c2);
                        }
                    }
                    // 归一化/softmax：dst 跟随 a
                    OP_LN | OP_RMSNORM | OP_SOFTMAX | OP_MASK_MUL => {
                        let (r, c) = self.shape_of(a);
                        if r > 0 && c > 0 {
                            self.set_shape(d, r, c);
                        }
                    }
                    // O6-A 归约：MAX 是逐行归约 ⇒ dst[rows,1]；
                    // TOPK_IDX 只对第 0 行取 argmax ⇒ dst[1,1]。
                    //
                    // 为什么必须在这里推断而不能靠 `_ => {}` 兜底：
                    // 没有推断时输出槽保持 0x0，而 MAX/TOPK_IDX 的 dst 槽
                    // 在 0x0 下宽度为 0，归约结果**写不进去也不报错**
                    // ⇒ 生成循环拿到一个恒定 0 ⇒ 永远重复 token 0，
                    // 这是「输入不合法但工具不报错、继续跑出坏结果」那一类。
                    // 因此这里推断、引擎侧再对 0x0 硬失败，两侧都堵。
                    OP_MAX => {
                        let (r, _c) = self.shape_of(a);
                        if r > 0 {
                            self.set_shape(d, r, 1);
                        }
                    }
                    OP_TOPK_IDX => {
                        let (_r, c) = self.shape_of(a);
                        if c > 0 {
                            self.set_shape(d, 1, 1);
                        }
                    }
                    // P7 反向：softmax_bwd dx 跟随 dy；ln_bwd dx 跟随 dy（=x 形状）
                    OP_SOFTMAX_BWD => {
                        let (r, c) = self.shape_of(a);
                        if r > 0 && c > 0 {
                            self.set_shape(d, r, c);
                        }
                    }
                    OP_LN_BWD => {
                        let (r, c) = self.shape_of(a);
                        if r > 0 && c > 0 {
                            self.set_shape(d, r, c);
                        }
                        // dx 与 x 同形状（dt 里编的显式形状优先，不被覆盖）
                        self.set_shape(b, r, c);
                    }
                    // outer_bwd：dst=dv 的长度取自 v（a=ds 的列数），
                    // 形状写成 [1, n] 行向量，与 OUTER 的 v/k 存放约定一致。
                    OP_OUTER_BWD => {
                        let (_sr, sc) = self.shape_of(a);
                        let (dr, dc) = self.shape_of(d);
                        if sc > 0 && dr == 0 && dc == 0 {
                            self.set_shape(d, 1, sc);
                        }
                    }
                    // emb_bwd：dst=dtable 的形状来自常量表，不推断
                    OP_EMB_BWD | OP_ADAMW => {}
                    // 转置：dst[C,M] = a[M,C]
                    OP_TRANSPOSE => {
                        let (r, c) = self.shape_of(a);
                        if r > 0 && c > 0 {
                            self.set_shape(d, c, r);
                        }
                    }
                    // 列拼接：列数相加
                    OP_CONCAT => {
                        let (r, c1) = self.shape_of(a);
                        let (_r2, c2) = self.shape_of(b);
                        if r > 0 && c1 > 0 && c2 > 0 {
                            self.set_shape(d, r, c1 + c2);
                        }
                    }
                    // 列切片：行数不变，列数取源列数减偏移
                    OP_SLICE => {
                        let (r, c) = self.shape_of(a);
                        let off = ins.b as u32;
                        if r > 0 && c > off {
                            self.set_shape(d, r, c - off);
                        }
                    }
                    // P16 形状推断 —— 与 OP_MAX / OP_TOPK_IDX 是同一类需求：
                    // **没有推断，输出槽就停在 0x0**，而引擎侧对 0x0 会
                    // exit(9)，于是「按文档写好的程序」在装载后立刻死掉，
                    // 报的却是「槽位宽度为 0」而不是「你漏了 SHAPE」。
                    //
                    // TOPK_BY_STRENGTH: dst 是 [L,2] 的索引对表，L = a 的元素数。
                    //
                    // P16-5 分段模式（`seg`）：a 是 [n,m]，dst 是 [n,2k]。
                    // 行数可以推断（= a 的行数），但**列数不能** —— k 是运行时
                    // 标量槽，汇编期不知道它的值。所以只推行数，列宽留给脚本
                    // 的显式 `SHAPE sel, n, 2k`（显式声明优先于推断）。
                    // 若脚本忘了写，dst 列宽保持 0 ⇒ 引擎的
                    // "segmented dst must be [n,2k]" 守卫会硬报错，
                    // 而不是静默算错 —— 这正是要的方向。
                    OP_TOPK_BY_STRENGTH => {
                        let (sr, sc) = self.shape_of(a);
                        if (ins.fl as u16) & (FL_SELGATE as u16) != 0 {
                            if sr > 0 {
                                let (dr, dc) = self.shape_of(d);
                                if dr == 0 {
                                    self.set_shape(d, sr, dc);
                                }
                            }
                        } else {
                            let l = sr * sc;
                            if l > 0 {
                                self.set_shape(d, l, 2);
                            }
                        }
                    }
                    // SPARSE_MATVEC: dst 是 [n_neurons(+1)]，n_neurons 由
                    // offsets 槽的**行数减一**决定（offsets 长度 = n+1）。
                    //
                    // 宽度取 1（列向量）而不是 [1,n]：引擎对 dr/dc 的校验是
                    // 「dc == 1」，与 SHAPE 的常见写法（SHAPE v, N, 1）一致；
                    // 若脚本显式写了 `SHAPE y, 1, N`，显式声明优先、会覆盖这里。
                    //
                    // ⚠️ 这里**不**把 nnz 那个额外格算进去：形状推断按
                    // 「语义上的输出」= [n,1] 给，nnz 那一格由脚本显式
                    // `SHAPE` 声明成 [n+1,1]（显式优先，不会被这里覆盖）。
                    // 反过来（推断成 n+1）会让**不测 nnz 的生产路径**也拿到
                    // 一个多出来的格子，等于让生产路径为验收仪器付代价。
                    OP_SPARSE_MATVEC => {
                        let (orows, ocols) = self.shape_of(a);
                        let n1 = orows * ocols;
                        if n1 >= 2 {
                            self.set_shape(d, n1 - 1, 1);
                        }
                    }
                    _ => {}
                }
            }
        }
        // 传播稳定后，钉住 SLICE 的显式列宽（此时 rows 已就绪），再跑一轮
        let sw = self.slice_w.clone();
        let mut pinned_any = false;
        for (slot, w) in sw {
            let r = self.slots[slot as usize].rows;
            if r > 0 {
                self.force_shape(slot, r, w);
                pinned_any = true;
            }
        }
        if pinned_any {
            for _ in 0..4 {
                for i in 0..self.text.len() {
                    let ins = self.text[i];
                    let (d, a, b) = (ins.dst, ins.a, ins.b);
                    match ins.op {
                        OP_GEMM => {
                            // 输出宽度随布局而变（与 interp.rs 保持一致）：
                            //   NT（默认，fl 无 nn）: N = B 的行数（B=[N,K]）
                            //   NN（fl 有 nn）      : N = B 的列数（B=[K,N]）
                            let (m, _k) = self.shape_of(a);
                            let (b_rows, b_cols) = self.shape_of(b);
                            let nn_out = if (ins.fl as u16) & FL_TRANS_B == 0 {
                                b_rows
                            } else {
                                b_cols
                            };
                            if m > 0 && nn_out > 0 {
                                self.set_shape(d, m, nn_out);
                            }
                        }
                        OP_EWADD | OP_EWMUL | OP_EWSUB | OP_EWDIV | OP_ACT | OP_EXP | OP_NEG
                        | OP_LOG | OP_CLAMP | OP_COPY => {
                            let (r, c) = self.shape_of(a);
                            if r > 0 && c > 0 {
                                self.set_shape(d, r, c);
                            }
                        }
                        OP_LN | OP_RMSNORM | OP_SOFTMAX | OP_MASK_MUL => {
                            let (r, c) = self.shape_of(a);
                            if r > 0 && c > 0 {
                                self.set_shape(d, r, c);
                            }
                        }
                        OP_SOFTMAX_BWD => {
                            let (r, c) = self.shape_of(a);
                            if r > 0 && c > 0 {
                                self.set_shape(d, r, c);
                            }
                        }
                        OP_LN_BWD => {
                            let (r, c) = self.shape_of(a);
                            if r > 0 && c > 0 {
                                self.set_shape(d, r, c);
                                self.set_shape(b, r, c);
                            }
                        }
                        OP_OUTER_BWD => {
                            let (_sr, sc) = self.shape_of(a);
                            let (dr, dc) = self.shape_of(d);
                            if sc > 0 && dr == 0 && dc == 0 {
                                self.set_shape(d, 1, sc);
                            }
                        }
                        OP_CONCAT => {
                            let (r, c1) = self.shape_of(a);
                            let (_r2, c2) = self.shape_of(b);
                            if r > 0 && c1 > 0 && c2 > 0 {
                                self.set_shape(d, r, c1 + c2);
                            }
                        }
                        OP_TRANSPOSE => {
                            let (r, c) = self.shape_of(a);
                            if r > 0 && c > 0 {
                                self.set_shape(d, c, r);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    fn shape_of(&self, slot: u32) -> (u32, u32) {
        let sd = &self.slots[slot as usize];
        (sd.rows, sd.cols)
    }

    fn emit_line(&mut self, line: &str) {
        let parts: Vec<&str> = line.split_whitespace().collect();
        let opname = parts[0].to_ascii_uppercase();
        let op = match Asm::opcode(&opname) {
            Some(o) => o,
            None => panic!("未知指令: {}", opname),
        };

        // ⭐ P10: 拒绝「汇编器认识、但引擎根本没实现」的 opcode。
        //
        // 为什么必须在这里拦（第 14 类静默降级的预防）：
        // 这些 opcode 在 format.rs 里定义了编号、在白皮书 §3.4 的 opcode 表里
        // 也列着，汇编器照单全收、**能成功产出 .vmb**；但 interp.rs 里
        // 没有对应的执行分支，运行到就落进 catch-all `exit(5) "unimpl op"`。
        //
        // 对使用者来说这是「按文档写了个合法指令，模型却在半路死掉」，
        // 而错误信息只有一个 `unimpl op`，不说是哪一条。
        // ⇒ 改成**汇编期硬失败并指名道姓**，问题在离现场最近的地方暴露。
        //
        // 清单可复现：python tools/p10_opcode_gaps.py
        if let Some(why) = UNIMPLEMENTED_OPS.iter().find(|(n, _)| *n == opname) {
            panic!(
                "指令 {} 在格式里已定义、但没有引擎实现（运行到会 exit(5)）。{}",
                opname, why.1
            );
        }

        let mut regs: Vec<&str> = Vec::new();
        let mut flags: u16 = 0;
        let mut imm: Option<i64> = None;

        // 逗号是可选分隔符：先剥掉尾部逗号，再按 flag/操作数分类
        let toks: Vec<&str> = parts[1..]
            .iter()
            .map(|p| p.trim_end_matches(','))
            .filter(|p| !p.is_empty())
            .collect();

        // ⭐ P10: `@label` 必须在送进 regs 之前摘出来。
        //
        // 修复前的行为（实测，P10 探针）：`@loop` 被当作操作数名送进 `slot()`，
        // 于是
        //   (1) 槽表里多出一个 `@loop [act] 0x0` 的垃圾槽（汇编器只给一条
        //       "形状仍是 0x0" 的警告，不报错）；
        //   (2) 更致命的是它**占掉一个 regs 位置**，把后面的真实操作数整体
        //       左移一位：`CMPLE i, lim, @loop` 编成 dst=0 a=2 b=3，
        //       而正确值是 dst=0 a=0 b=2。
        // `lim` 被挤进 `a`、`@loop` 占了 `b` ⇒ 比较的是 (lim, @loop槽)，
        // **静默算错**。
        //
        // 现在按「标签」语义处理：只允许出现一次，位置不参与 regs 编号。
        let mut label: Option<String> = None;
        for &p in &toks {
            if let Some(name) = p.strip_prefix('@') {
                if name.is_empty() {
                    panic!("标签名不能为空（`@` 后面要有名字）");
                }
                if label.is_some() {
                    panic!("一条指令只能有一个跳转标签，实际写了两个：`{}`", line);
                }
                label = Some(name.to_string());
            }
        }

        // 裸 flag 名（relu / causal / acc ...）按 flag 解释，不占操作数位。
        // 这样 `ACT dst, src, relu` 和 `SOFTMAX dst, src, causal` 都能写。
        // 注意：必须先确认它确实是已知 flag 名，否则会吃掉同名的槽位。
        for &p in &toks {
            // 标签已单独摘出（见上），不再参与 flag/操作数分类。
            if p.starts_with('@') {
                continue;
            }
            if let Some(eq) = p.find('=') {
                let (k, v) = (&p[..eq], &p[eq + 1..]);
                if k == "imm" {
                    imm = Some(v.parse().expect("imm 必须是整数"));
                } else if k == "k" {
                    // TOPK_THRESH 的 k 值直接编进 fl（它是小整数，16 位足够）
                    flags = v.parse().expect("k 必须是整数");
                } else {
                    flags |= Asm::flag(k).unwrap_or_else(|| panic!("未知标志: {}", k));
                }
            } else if Asm::flag(p).is_some() && Asm::takes_flag(op, p) {
                // `bias` 既是 GEMM 的 flag 名，也常被用作 LN 的第 4 个操作数（槽位名）。
                // 若这条指令是四操作数指令且已经攒够 3 个操作数，则第 4 个位置
                // **必须**按槽位解释，不能被 flag 吃掉。
                // （这个坑 P8 实际踩到：`LN out, x, g, bias` 里 bias 被当 flag 吞掉，
                //   dt 落成 0 → LN 用错 bias 槽 → 数值静默出错。）
                let fourth_is_slot = matches!(op, OP_LN | OP_RMSNORM | OP_LN_BWD | OP_OUTER_BWD)
                    && regs.len() >= 3;
                if fourth_is_slot {
                    regs.push(p);
                } else {
                    flags |= Asm::flag(p).unwrap();
                }
            } else {
                regs.push(p);
            }
        }

        // SHAPE: dst=槽位, a=rows, b=cols（a/b 是数字，不是槽位名）
        if op == OP_SHAPE {
            let slot = self.slot(regs[0]);
            let rows: u32 = regs.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
            let cols: u32 = regs.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
            self.text.push(Ins {
                op,
                dt: DT_F32 as u16,
                fl: 0,
                dst: slot,
                a: rows,
                b: cols,
            });
            return;
        }

        // SLICE / WRITECOL: dst=槽位, a=槽位, b=起始列（数字）[, width]
        //   宽度记到 slice_w 表里，等 infer_shapes 阶段（形状已知后）再应用，
        //   因为 emit_line 时 src 的形状可能还没推出来。
        // PUSH / POP / STACK_INIT: 数字操作数编进 b
        //   PUSH stack, src, step    → 2 个槽 + 1 个数字
        //   POP  dst, stack, step    → 2 个槽 + 1 个数字
        //   STACK_INIT stack, max    → 1 个槽 + 1 个数字  ← 只有一个槽！
        if op == OP_PUSH || op == OP_POP || op == OP_STACK_INIT {
            let d0 = self.slot(regs[0]);
            // STACK_INIT 没有第 2 个槽位操作数，只有数字 max_steps。
            // 早期版本无条件 slot(regs[1])，会把数字注册成一个垃圾槽位
            //（成员 A2 发现：槽表里出现一个叫 "3" 的 0x0 槽）。
            let n_slots = if op == OP_STACK_INIT { 1 } else { 2 };
            let a0 = if n_slots == 2 {
                self.slot(regs[1])
            } else {
                0
            };
            let step: u32 = regs
                .get(n_slots)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            self.text.push(Ins {
                op,
                dt: DT_F32 as u16,
                fl: 0,
                dst: d0,
                a: a0,
                b: step,
            });
            return;
        }

        if op == OP_SLICE || op == OP_WRITECOL {
            let d0 = self.slot(regs[0]);
            let s0 = self.slot(regs[1]);
            let off: u32 = regs.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
            if op == OP_SLICE {
                if let Some(w) = regs.get(3).and_then(|s| s.parse().ok()) {
                    self.slice_w.push((d0, w));
                }
            }
            self.text.push(Ins {
                op,
                dt: DT_F32 as u16,
                fl: 0,
                dst: d0,
                a: s0,
                b: off,
            });
            return;
        }

        // LOOP_CNT ctr, @label —— 计数器槽在 regs[0]，标签走 fixup（写 dst）。
        // 计数器就是普通激活槽，形状必须是标量（[1,N]，通常 SHAPE ctr,1,1）。
        if op == OP_LOOP_CNT {
            if regs.is_empty() {
                panic!("LOOP_CNT 需要 1 个计数器槽：`LOOP_CNT ctr, @label`");
            }
            let ctr = self.slot(regs[0]);
            self.loop_targets.push(self.text.len()); // 待 resolve 后校验「向后跳」
            self.text.push(Ins {
                op,
                dt: DT_F32 as u16,
                fl: 0,
                dst: 0, // 由 resolve() 填标签
                a: ctr,
                b: 0,
            });
            self.expect_label(&label, &opname, "LOOP_CNT");
            return;
        }

        // CMPLE lhs, rhs, @label —— **条件跳转**，不是三寄存器运算。
        //
        // 引擎侧语义（interp.rs:680）：a[0] <= b[0] 则 ip = dst。
        // 所以字段是 dst=目标, a=lhs, b=rhs，**不能用通用路径**
        // （通用路径会把 lhs 当 dst、rhs 当 a ⇒ 比较 (rhs, 目标槽)，静默算错）。
        //
        // 修复前这里走的就是通用路径，实测编成 dst=0 a=2 b=3（`lim` 进了 a、
        // `@loop` 垃圾槽占了 b），比较的是 (lim, @loop) 两个都在运行时是 0
        // ⇒ 恒真 ⇒ 立即跳转。**这是 P10 探针抓到的第 14 个静默降级。**
        if op == OP_CMPLE {
            if regs.len() < 2 {
                panic!("CMPLE 需要 2 个操作数 + 1 个标签：`CMPLE lhs, rhs, @label`");
            }
            let lhs = self.slot(regs[0]);
            let rhs = self.slot(regs[1]);
            if regs.len() > 2 {
                panic!(
                    "CMPLE 只接受 2 个操作数（lhs, rhs）+ 1 个标签，实际给了 {} 个操作数。\
                     （修复前多出来的会被静默当作操作数，导致比较对象错位。）",
                    regs.len()
                );
            }
            self.text.push(Ins {
                op,
                dt: DT_F32 as u16,
                fl: 0,
                dst: 0, // 由 resolve() 填标签
                a: lhs,
                b: rhs,
            });
            self.expect_label(&label, &opname, "CMPLE");
            return;
        }

        // ROPE dst, src, theta [, pos0]
        //   dst=输出[S,hd], a=源[S,hd], b=theta 标量槽, dt=起始位置 pos0（数字）
        // 起始位置放进 dt：这条指令不需要「第 4 个槽」，而 dt 是 u16，
        // 正好放得下 <=65535 的位置。窗口更长的序列配合 LOOP_CNT 分块调用。
        if op == OP_ROPE {
            if regs.len() < 3 {
                panic!("ROPE 需要 3 个操作数 + 可选起始位置：`ROPE dst, src, theta [, pos0]`");
            }
            let d0 = self.slot(regs[0]);
            let s0 = self.slot(regs[1]);
            let th = self.slot(regs[2]);
            let pos0: u32 = regs.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
            if pos0 > u16::MAX as u32 {
                panic!(
                    "ROPE 的起始位置 {} 超过 dt 字段（u16）上界 {}。\
                     更长的序列请配合 LOOP_CNT 分块调用（这也让指令流与长度无关）。",
                    pos0,
                    u16::MAX
                );
            }
            self.text.push(Ins {
                op,
                dt: pos0 as u16,
                fl: 0,
                dst: d0,
                a: s0,
                b: th,
            });
            return;
        }

        let dst = regs.first().map(|s| self.slot(s)).unwrap_or(0);
        let a = regs.get(1).map(|s| self.slot(s)).unwrap_or(0);
        let b = regs.get(2).map(|s| self.slot(s)).unwrap_or(0);

        // SPARSE_MATVEC dst, offsets, src, w, x [, nnz]
        //   4 个槽位角色 + 1 个可选 flag 装不进 dst/a/b 三个字段，
        //   所以第 4 个槽（权重）走 fl 低 6 位、第 5 个槽（x）走 dt。
        //   —— 与 OP_ROPE（pos0 走 dt）、OP_LN（bias 走 dt）同一套做法：
        //      **不加宽指令**，而是复用这条指令确凿不用的字段。
        //
        // 必须在这里显式绑定，不能走通用路径：通用路径会把第 4 个操作数
        // 当 flag/垃圾丢掉 ⇒ 权重槽号落成 0 ⇒ 引擎读到 slot0 的垃圾
        // ⇒ 数值静默错。OP_LAYERB 与 CMPLE 都因此各踩过一次（见它们的注释）。
        if op == OP_SPARSE_MATVEC {
            if regs.len() < 4 {
                panic!(
                    "SPARSE_MATVEC 需要 4 个操作数：`SPARSE_MATVEC dst, offsets, src, w [, x] [nnz]`；\
                     实际给了 {} 个",
                    regs.len()
                );
            }
            // 第 5 个槽（输入向量 x）可选，走 dt。不写则 dt=0 ⇒ x 落在 slot0。
            // 这不是「默认无害」：slot0 通常是垃圾槽，所以缺 x 时宁可硬报错。
            if regs.len() < 5 {
                panic!(
                    "SPARSE_MATVEC 需要 5 个操作数：`SPARSE_MATVEC dst, offsets, src, w, x [nnz]`。\
                     漏掉 x 会让 dt 落成 0，引擎去读 slot0 的垃圾当作输入向量。"
                );
            }
            let d0 = self.slot(regs[0]);
            let o0 = self.slot(regs[1]);
            let s0 = self.slot(regs[2]);
            let w0 = self.slot(regs[3]);
            let x0 = self.slot(regs[4]);
            // P16-5: `selgate` 模式下第 6 个操作数是 TOPK_BY_STRENGTH 的
            // 选择表 `sel`，且**必须紧跟在 src 槽之后**（约定 src+1）。
            //
            // 为什么要显式校验：这是本项目已经出现过两次的隐式相邻约定
            // （LN_BWD 的 gamma/dg/db、OUTER_BWD 的 k/dk），而两次都因为
            // 不校验而把结果写进了别的槽。这里如果写成「引擎按 src+1 读、
            // 脚本却把 sel 声明在别处」，引擎会读到 src 旁边的**另一个数组**
            // ——两列都是 f32，形状还可能恰好对得上 ⇒ 静默算错。
            // 所以：脚本必须把 sel 槽紧跟在 src 之后，否则汇编期硬报错。
            let selgate = flags & (FL_SELGATE as u16) != 0;
            if selgate {
                if regs.len() < 6 {
                    panic!(
                        "SPARSE_MATVEC 的 selgate 模式需要第 6 个操作数（选择表 sel）：\
                         `SPARSE_MATVEC dst, offsets, src, w, x, sel, selgate`。\
                         并且 sel 必须**紧跟在 src 之后**声明（引擎按 src+1 读）。"
                    );
                }
                let g0 = self.slot(regs[5]);
                if g0 != s0 + 1 {
                    panic!(
                        "SPARSE_MATVEC selgate 的 sel 槽必须是 src 槽的紧邻下一个\
                         （src={}, 期望 sel={}, 实际 sel={}）。\
                         引擎按 `src + 1` 取选择表 —— 声明在别处会读到另一个数组，\
                         两列都是 f32，属于静默算错那一类。",
                        s0, s0 + 1, g0
                    );
                }
            }
            if w0 > SPARSE_WSLOT_MAX {
                panic!(
                    "SPARSE_MATVEC 的权重槽号 {} 超过 fl 低 6 位的上界 {}。\
                     低 6 位是这条指令唯一装得下第 4 个槽的地方（指令是 16B 定长，\
                     dst/a/b 三个字段已被占用）。槽号是**绝对索引**，截断即静默错位 ——\
                     所以这里硬报错。请把权重槽往前挪，或改用 COPY 把它搬到低位槽。",
                    w0, SPARSE_WSLOT_MAX
                );
            }
            // flags 里已含 `nnz`（若写了）。低 6 位写权重槽号，高位标志保留。
            let fl_hi = flags & !(SPARSE_WSLOT_MASK as u16);
            self.text.push(Ins {
                op,
                dt: x0 as u16,
                fl: (fl_hi as u8) | (w0 as u8),
                dst: d0,
                a: o0,
                b: s0,
            });
            return;
        }

        // P13 task E: LAYERB takes ONE register (the delta slot) and it must
        // land in `a`, with dst = 0.
        //
        // WHY THIS IS EXPLICIT RATHER THAN LEFT TO THE DEFAULT: the generic
        // rule above puts a single register in `dst` and leaves `a` = 0, so
        // `LAYERB lstride` encoded as `dst=39 a=0`. The engine read `a` and
        // therefore read SLOT 0 every iteration -- the base never advanced.
        // That cost two debug cycles because the 1-layer case is bit-exact
        // either way (base 0 is correct for one layer), so the bug only showed
        // up at 2+ layers. Pin the encoding here, in one place, and the
        // engine's `a` read is then correct by construction.
        if op == OP_LAYERB {
            // Resolve the slot FIRST (self.slot takes &mut self), then push.
            let dslot = match regs.first() {
                Some(n) => self.slot(n),
                None => panic!("LAYERB 需要 1 个操作数：持有字节增量的标量槽"),
            };
            self.text.push(Ins {
                op,
                fl: 0,
                dt: 0,
                dst: 0,
                a: dslot,
                b: 0,
            });
            return;
        }

        // 四操作数指令：第 4 个操作数编进 dt 字节
        //   LN / RMSNORM : bias 槽
        //   LN_BWD       : dg 槽（bias 梯度）；db 槽约定为 dg 槽紧邻的下一个槽
        //   OUTER_BWD    : k 槽；dk 槽约定为 k 槽紧邻的下一个槽
        let mut dt: u32 = DT_F32 as u32;
        if (op == OP_LN || op == OP_RMSNORM) && regs.len() >= 4 {
            dt = self.slot(regs[3]);
        }
        // P13 task D: an int8 GEMM names its per-output-row SCALE slot as the
        // 4th operand. Ins has op/fl/dt/dst/a/b with no spare byte (verified
        // against the repr(C) definition), so `dt` is the only place it fits --
        // the same convention OP_LN and OP_ROPE already use.
        //
        // It is validated here rather than left implicit: if the 4th operand is
        // present but the B operand is NOT int8, or vice versa, that is a
        // program error that would otherwise silently multiply by whatever
        // `dt` happened to contain.
        if op == OP_GEMM && regs.len() >= 4 {
            let s = self.slot(regs[3]);
            // `b` is already the slot index of the 3rd register. Its dtype is
            // only known for CONST slots, so compare on the raw slot entry.
            let is_i8 = self.slot_is_const_dtype(b, DT_I8);
            if !is_i8 {
                // Index rather than borrow: `as_str()` on this element type hits
                // an unstable feature (measured), and the map/borrow dance is
                // not worth the churn for an error message.
                let bname: &str = match regs.get(2) {
                    Some(v) => v,
                    None => "?",
                };
                panic!(
                    "GEMM 给了第 4 个操作数（scale 槽 {}），但权重槽 {} 不是 i8 常量。\
                     只有 int8 权重才需要 scale；f32/bf16 权重请去掉第 4 个操作数。",
                    s, bname
                );
            }
            dt = s;
        }
        if op == OP_LN_BWD && regs.len() >= 4 {
            let gi = self.slot(regs[3]);
            dt = gi;
            // 引擎按 gi/gi+1/gi+2 取 gamma/dg/db（见 interp.rs）。
            // 这是隐式相邻约定，**必须显式校验**，否则 dg/db 会静默写进别的槽
            // —— 成员 A2 实测踩到过（dg 写到了 ds）。
            if regs.len() >= 5 {
                let want = self.slot(regs[4]);
                if want != gi + 1 {
                    panic!(
                        "LN_BWD 的 dg 槽必须是 gamma 槽的紧邻下一个（gamma={}, 期望 dg={}, 实际 dg={}）。",
                        gi,
                        gi + 1,
                        want
                    );
                }
            }
            if (gi + 2) as usize >= self.slots.len() {
                panic!(
                    "LN_BWD: gamma 槽 {} 之后没有 dg/db 两个槽（需要 {} 和 {}）。\
                     请在 gamma 后面按顺序声明 dgamma 和 dbias。",
                    gi,
                    gi + 1,
                    gi + 2
                );
            }
        }
        if op == OP_OUTER_BWD {
            // OUTER_BWD dv, ds, v, k [, dk]
            //   5 个角色装不进 4 个字段 → k 与 dk 打包成紧邻一对，dt 存 k 的槽号。
            if regs.len() < 4 {
                panic!("OUTER_BWD 需要 4 个操作数：dv, ds, v, k（dk 紧随 k 之后）");
            }
            let ki = self.slot(regs[3]);
            dt = ki;
            if regs.len() >= 5 {
                let want = self.slot(regs[4]);
                if want != ki + 1 {
                    panic!(
                        "OUTER_BWD 的 dk 槽必须是 k 槽的紧邻下一个（k={}, 期望 dk={}, 实际 dk={}）。\
                         请按顺序声明 k 和 dk。",
                        ki,
                        ki + 1,
                        want
                    );
                }
            }
            if (ki + 1) as usize >= self.slots.len() {
                panic!(
                    "OUTER_BWD: k 槽 {} 之后没有 dk 槽（需要 {}）。请在 k 后面声明 dk。",
                    ki,
                    ki + 1
                );
            }
        }

        // dt 是 4 操作数指令的「第 4 个操作数槽号」。
        // P8 之前它是 u8，槽位 >255 会被 `as u8` 静默截断 → LN 用错 bias → 数值静默错。
        // 现在加宽到 u16，并在超界时硬报错（成员 C2 发现该截断是放大 ESERNN 的真阻塞）。
        if dt > u16::MAX as u32 {
            panic!(
                "槽位号 {} 超过指令 dt 字段（u16）上界 {}。请拆分模型或提高 dt 位宽。",
                dt,
                u16::MAX
            );
        }
        self.text.push(Ins {
            op,
            dt: dt as u16,
            fl: flags as u8,
            dst: imm.map(|v| v as u32).unwrap_or(dst),
            a,
            b,
        });

        // 需要跳转标签的指令（无条件 JMP / 条件 BRANCH / 计数 BRANCH!）。
        //
        // 跳转目标统一写在 `dst` —— 引擎侧按 `dst` 读（P10 修正）。
        //
        // 修复前：`resolve()` 写 `dst`，但引擎 `OP_JMP` 读的是 `a`，
        // 而 `a` 恒为 0 ⇒ **标签被无声丢弃，永远跳到指令 0**。
        // 探针实测：标签在 ins 3 时，程序仍跳回 ins 0（输出重复了 ins 0 的打印）。
        // 这是「静默跳错」，比不支持跳转更危险。
        // （CMPLE / LOOP_CNT 在通用路径之前就 return 了，各自调用 expect_label。）
        if op == OP_JMP || op == OP_BRANCH {
            let l = label.clone();
            self.expect_label(&l, &opname, &opname);
        } else if label.is_some() {
            // 其它指令不接受标签：静默忽略会让「以为设了跳转、其实没设」，
            // 属于同一类静默降级，一律硬报错。
            panic!(
                "指令 {} 不支持跳转标签（`@...`）。只有 JMP / BRANCH / CMPLE / LOOP_CNT 需要标签。",
                opname
            );
        }
    }

    /// 跳转类指令的统一收尾：登记标签回填。
    ///
    /// 为什么抽成函数：`CMPLE` / `LOOP_CNT` 在通用路径**之前**就 `return` 了，
    /// 如果只在末尾登记 fixup，这两条指令的标签会被静默丢掉
    /// （实测：`dst` 停在 0 ⇒ 永远跳到指令 0，正是要修的那个 bug 的翻版）。
    fn expect_label(&mut self, label: &Option<String>, opname: &str, what: &str) {
        match label {
            Some(t) => {
                let idx = self.text.len() - 1;
                self.fixups.push((idx, t.clone()));
            }
            None => panic!(
                "{} 需要跳转标签（形如 `@loop`），但没给。\
                 没有标签的跳转在修复前会静默变成「跳到指令 0」——宁可硬报错。",
                what
            ),
        }
        let _ = opname;
    }

    fn resolve(&mut self) {
        let fixups = self.fixups.clone();
        for (idx, label) in fixups {
            let target = *self
                .labels
                .get(&label)
                .unwrap_or_else(|| panic!("未定义标签: {}", label));
            self.text[idx].dst = target;
        }

        // P10: LOOP_CNT 的目标必须是**向后**跳（target < 该指令的位置）。
        //
        // 为什么硬校验：向前跳的「循环」其实是个 if，一条 LOOP_CNT 就写错了语义。
        // 更糟的是如果 target == 自身位置，计数器永远减不到 0 之前的判断
        // 会变成**静默死循环**（程序不报错、不退出，只是永远跑）。
        // 「输入不合法但工具不报错、继续跑出坏结果」= 静默降级，一律硬失败。
        for &idx in &self.loop_targets {
            let target = self.text[idx].dst as usize;
            if target >= idx {
                panic!(
                    "LOOP_CNT (指令 #{}) 的跳转目标必须**向后**（循环体起点），\
                     但标签指向 #{}（>= 自身）。向前跳是 if、等于自身是死循环。\
                     请把标签放在循环体之前。",
                    idx, target
                );
            }

            // ⭐ 计数器「重复递减」检查（P10 实测踩到，写进了探针历史）。
            //
            // `LOOP_CNT` **自己**就会把计数器减 1。如果循环体里**又**减了一次
            // （很自然的写法：`EWSUB ctr, ctr, one` + `LOOP_CNT ctr, @loop`），
            // 每轮实际减 2 ⇒ **静默少跑一半的迭代**，没有报错、数值也不 NaN。
            //
            // 实测（P10 探针）：`SHAPE ctr,1,1` + 体内 EWSUB + LOOP_CNT，
            // ctr=3 时打印出 `2, 1`（应该是 `3, 2, 1`），**丢了一次迭代**。
            // 这正是「输入不合法但工具不报错」的典型 —— 所以硬报错。
            let ctr = self.text[idx].a;
            for k in target..idx {
                let ins = self.text[k];
                // 只看会**写**计数器的算子（dst == ctr），且排除 LOOP_CNT 自身
                let writes_ctr = match ins.op {
                    OP_EWADD | OP_EWSUB | OP_EWMUL | OP_EWDIV | OP_COPY | OP_SCALE
                    | OP_LOAD | OP_SAVE | OP_ACT | OP_NEG => ins.dst == ctr,
                    _ => false,
                };
                if writes_ctr {
                    panic!(
                        "LOOP_CNT (指令 #{}) 的循环体（#{}..#{}）里有指令 #{} 也在写计数器槽 {}。\
                         LOOP_CNT 自己会减 1，循环体再减一次就会**每轮减 2**、\
                         静默少跑一半迭代（P10 实测过）。\
                         请删掉循环体里的计数器运算，或改用另一个槽做累加。",
                        idx, target, idx - 1, k, ctr
                    );
                }
            }
        }
    }

    fn add_const(&mut self, name: &str, rows: u32, cols: u32, dtype: u8, data: Vec<u8>) {
        self.consts.push(Const {
            name: name.to_string(),
            rows,
            cols,
            dtype,
            data,
        });
    }

    fn emit_binary(&self) -> Vec<u8> {
        let header = 32usize;
        let mut const_data: Vec<u8> = Vec::new();
        let mut offsets: Vec<u32> = Vec::new();
        for c in &self.consts {
            offsets.push(const_data.len() as u32);
            const_data.extend_from_slice(&c.data);
            while const_data.len() % CONST_ALIGN != 0 {
                const_data.push(0);
            }
        }

        // META: 头 28B + const表 + 槽位表 + 名字哈希表
        let n_const = self.consts.len() as u32;
        let n_slot = self.slots.len() as u32;
        let names_off = META_HDR + self.consts.len() * CONST_ENT + self.slots.len() * SLOT_ENT;
        let slots_off = META_HDR + self.consts.len() * CONST_ENT;

        // 激活区/状态区/梯度区：按最终形状紧密排布（64B 对齐）
        let mut act_bytes = 0u32;
        let mut state_bytes = 0u32;
        let mut grad_bytes = 0u32;
        let mut act_off = vec![0u32; self.slots.len()];
        let mut state_off = vec![0u32; self.slots.len()];
        let mut grad_off = vec![0u32; self.slots.len()];
        for (i, sd) in self.slots.iter().enumerate() {
            let need = sd.rows * sd.cols * 4;
            match sd.kind {
                Kind::Act => {
                    act_off[i] = act_bytes;
                    act_bytes = (act_bytes + need + 63) & !63;
                }
                Kind::State => {
                    state_off[i] = state_bytes;
                    state_bytes = (state_bytes + need + 63) & !63;
                }
                Kind::Grad => {
                    grad_off[i] = grad_bytes;
                    grad_bytes = (grad_bytes + need + 63) & !63;
                }
                Kind::Const => {}
            }
        }

        let mut meta: Vec<u8> = Vec::new();
        meta.extend_from_slice(&n_const.to_le_bytes());
        meta.extend_from_slice(&n_slot.to_le_bytes());
        meta.extend_from_slice(&(slots_off as u32).to_le_bytes());
        meta.extend_from_slice(&act_bytes.to_le_bytes());
        meta.extend_from_slice(&state_bytes.to_le_bytes());
        meta.extend_from_slice(&grad_bytes.to_le_bytes());
        meta.extend_from_slice(&(names_off as u32).to_le_bytes());
        for (i, c) in self.consts.iter().enumerate() {
            meta.extend_from_slice(&offsets[i].to_le_bytes());
            meta.extend_from_slice(&(c.data.len() as u32).to_le_bytes());
            meta.extend_from_slice(&c.rows.to_le_bytes());
            meta.extend_from_slice(&c.cols.to_le_bytes());
            meta.push(c.dtype);
            meta.extend_from_slice(&[0u8; 3]);
            meta.extend_from_slice(&hash32(&c.name).to_le_bytes());
        }
        for (i, sd) in self.slots.iter().enumerate() {
            let (kind, idx) = match sd.kind {
                Kind::Const => {
                    let ci = self
                        .consts
                        .iter()
                        .position(|c| c.name == sd.name)
                        .expect("常量槽找不到对应常量");
                    (SLOT_CONST, ci as u32)
                }
                Kind::Act => (SLOT_ACT, act_off[i]),
                Kind::State => (SLOT_STATE, state_off[i]),
                Kind::Grad => (SLOT_GRAD, grad_off[i]),
            };
            meta.extend_from_slice(&kind.to_le_bytes());
            meta.extend_from_slice(&idx.to_le_bytes());
            meta.extend_from_slice(&sd.rows.to_le_bytes());
            meta.extend_from_slice(&sd.cols.to_le_bytes());
        }
        // 名字哈希表（按槽位顺序）：供运行时按名查找激活/梯度/超参槽
        for sd in &self.slots {
            meta.extend_from_slice(&hash32(&sd.name).to_le_bytes());
        }

        let text_off = header;
        let text_len = self.text.len() * INS_SIZE;
        let const_off = text_off + text_len;
        let meta_off = const_off + const_data.len();

        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&(text_off as u32).to_le_bytes());
        out.extend_from_slice(&(text_len as u32).to_le_bytes());
        out.extend_from_slice(&(const_off as u32).to_le_bytes());
        out.extend_from_slice(&(const_data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(meta_off as u32).to_le_bytes());
        out.extend_from_slice(&(meta.len() as u32).to_le_bytes());
        for ins in &self.text {
            // P8 布局：op:u8 | fl:u8 | dt:u16 | dst:u32 | a:u32 | b:u32 = 16B（无填充）
            out.push(ins.op);
            out.push(ins.fl);
            out.extend_from_slice(&ins.dt.to_le_bytes());
            out.extend_from_slice(&ins.dst.to_le_bytes());
            out.extend_from_slice(&ins.a.to_le_bytes());
            out.extend_from_slice(&ins.b.to_le_bytes());
        }
        out.extend_from_slice(&const_data);
        out.extend_from_slice(&meta);
        out
    }
}

// 每条指令的操作数个数（超过这个数的裸词按 flag 解释）
fn max_operands(op: u8) -> usize {
    match op {
        // 单操作数指令：第 1 个操作数放 dst（与 PRINT / LN_EPS 的解释器惯例一致）
        OP_HALT | OP_PRINT | OP_LN_EPS => 1,
        OP_ACT | OP_SOFTMAX | OP_LN => 3,
        OP_GEMM | OP_GEMV | OP_OUTER | OP_MASK_MUL | OP_EWADD | OP_EWMUL | OP_EWSUB | OP_EWDIV
        | OP_SCALE => 3,
        _ => 3,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("用法: vmlm-asm <in.vml> <out.vmb> [name:file.bin:rows:cols ...]");
        std::process::exit(1);
    }
    let src = std::fs::read_to_string(&args[1]).expect("读不到 .vml");
    let mut asm = Asm::new();

    // 1) 先登记常量（名字要能在 slot() 里被识别为常量槽）
    //
    // P12：语法扩展为 `name:file:rows:cols[:dtype]`，dtype ∈ {f32,bf16}。
    // 为什么需要：MiniCPM / Llama 的权重是 **bf16**。此前汇编器硬编码
    // DT_F32，于是 bf16 字节会被原样标成 f32 —— 引擎读出来是垃圾数。
    // 现在 bf16 常量按 **2 字节/元素** 校验长度，**不做任何转换**直接写盘，
    // 引擎侧由 `gemm_nt_bf16` 在寄存器里加宽。
    //
    // 长度校验是必须的：字节数 ÷ 元素大小的**整除性**如果不检查，
    // 一个 f32 文件被标成 bf16（或反之）会静默错位一半的数据。
    for w in &args[3..] {
        let f: Vec<&str> = w.split(':').collect();
        if f.len() < 4 {
            panic!("权重参数格式: name:file:rows:cols[:f32|bf16]，收到: {}", w);
        }
        let data = std::fs::read(f[1]).expect("读不到权重文件");
        let rows: u32 = f[2].parse().expect("rows 必须是整数");
        let cols: u32 = f[3].parse().expect("cols 必须是整数");
        let dtype = match f.get(4).map(|s| *s) {
            None | Some("f32") => DT_F32,
            Some("bf16") => DT_BF16,
            // P13 task D: int8 weights for the dequantising GEMM.
            Some("i8") => DT_I8,
            Some(other) => panic!("未知 dtype: {}（只支持 f32 / bf16 / i8）", other),
        };
        let esz = dtype_size(dtype);
        let want = rows as usize * cols as usize * esz;
        if data.len() != want {
            panic!(
                "常量 {} 的字节数不对：{} 字节，但 {}x{} {} 需要 {} 字节（{} 字节/元素）。\
                 形状或 dtype 写错了 —— 这会导致**静默按错误步长读内存**。",
                f[0], data.len(), rows, cols,
                match dtype {
                    DT_BF16 => "bf16",
                    DT_I8 => "i8",
                    _ => "f32",
                },
                want, esz
            );
        }
        asm.add_const(f[0], rows, cols, dtype, data);
    }

    // 2) 扫一遍源码建槽位表（保证顺序稳定，与指令内 [dst,a,b] 次序无关）
    //    只登记「操作数」：跳过 flag=value，跳过 SHAPE 的数值操作数
    for raw in src.lines() {
        let line = match raw.find(';') {
            Some(i) => &raw[..i],
            None => raw,
        };
        let line = line.trim();
        if line.is_empty() || line.starts_with('.') || line.ends_with(':') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        let is_shape = parts[0].eq_ignore_ascii_case("SHAPE");
        let opname = parts[0].to_ascii_uppercase();
        // 该行的裸 flag 名集合（供后续跳过它们，不建槽位）
        let op = Asm::opcode(&opname).unwrap_or(OP_HALT);
        // LN/RMSNORM/LN_BWD 有第 4 个操作数（bias / dg），也要登记为槽位
        let max_regs = if opname == "LN" || opname == "RMSNORM" || opname == "LN_BWD" {
            4
        } else if opname == "LOOP_CNT" {
            1 // 只有计数器槽，标签不是槽位
        } else if opname == "SPARSE_MATVEC" {
            // P16: 5 个槽位操作数（dst, offsets, src, w, x）。
            // 这里若漏了，第 5 个槽（x）**不会进槽表** ⇒ 它的槽号在
            // emit_line 里被分配成另一个槽，而指令里 dt 存的是那个号 ——
            // 但预扫 pass 建的槽表顺序与 emit_line 的分配顺序不一致时，
            // 槽号是绝对索引，偏移即静默错位（本项目被这条咬过多次）。
            // P16-5: selgate 模式多一个 sel 槽（第 6 个），故取 6。
            // 多登记一个不存在的名字只会多一个 0x0 槽 + 一条警告；
            // 少登记则是**静默错位**，所以这里宁可放宽。
            6
        } else {
            3
        };
        let mut n = 0usize;
        // CMPLE lhs, rhs, @label —— 第 3 个是标签，不是槽位；
        // 且它的寄存器次序是 (lhs, rhs)，只有 2 个槽位操作数。
        let is_cmple = opname == "CMPLE";
        for &p in &parts[1..] {
            let p = p.trim_end_matches(',');
            if p.is_empty() || p.contains('=') {
                continue; // 空 token / flag=value 不是槽位
            }
            // ⭐ P10: `@label` 不是槽位。
            //
            // 这个「预扫一遍建槽表」的 pass 与 emit_line 是**两处独立**的解析，
            // 之前都缺 `@` 判断：emit_line 侧修好之后，这里仍会给 `@loop`
            // 建出一个 `[act] 0x0` 垃圾槽（实测：`JMP @loop2` 这种**零其它操作数**
            // 的指令照样冒出 slot `@loop2`）。垃圾槽会污染槽表并触发
            // "形状仍是 0x0" 警告 —— 更糟的是它让槽位编号整体偏移，
            // 而槽号是编进指令的**绝对索引**，偏移即静默错位。
            //
            // 教训：修一个解析器时，必须找出**所有**解析同一份语法的位置。
            if p.starts_with('@') {
                continue;
            }
            if is_shape {
                if n >= 1 {
                    // SHAPE 的第 2/3 操作数必须是纯数字（行/列），
                    // 写错（如 "decayCol,16"）会让槽位名与数字粘连，
                    // 这里直接报错而不是静默建出垃圾槽位。
                    if p.parse::<u32>().is_err() {
                        panic!("SHAPE 的参数必须是数字，收到: {}", p);
                    }
                    n += 1;
                    continue;
                }
            } else if (opname == "SLICE" || opname == "WRITECOL") && n >= 2 {
                // 第 3 个操作数是列偏移（数字）
                n += 1;
                continue;
            } else if opname == "ROPE" && n >= 3 {
                // ROPE 的第 4 个操作数是起始位置（数字），不是槽位。
                // 不拦住的话它会变成一个叫 "0" 的垃圾槽位污染槽表，
                // 并让槽号整体偏移 —— 与 STACK_INIT 的 max_steps 同类。
                n += 1;
                continue;
            } else if opname == "STACK_INIT" && n >= 1 {
                // STACK_INIT 只有 2 个操作数：stack（槽）, max_steps（数字）。
                // 第 2 个就是数字，必须在 n>=1 就拦住，否则 max_steps 会被
                // 注册成一个叫 "3" 的 0x0 槽位污染槽表（成员 A2 发现）。
                n += 1;
                continue;
            } else if (opname == "PUSH" || opname == "POP") && n >= 2 {
                // PUSH/POP 第 3 个操作数是 step 号（数字）
                n += 1;
                continue;
            } else if Asm::takes_flag(op, p) {
                continue; // 裸 flag 名（relu/acc/causal...）不是槽位
            }
            if n >= max_regs {
                continue;
            }
            // CMPLE 只有 2 个槽位操作数（lhs, rhs），标签已在上方跳过。
            if is_cmple && n >= 2 {
                continue;
            }
            asm.slot(p);
            n += 1;
        }
    }

    // 3) 汇编 + 回填标签
    asm.assemble(&src);
    asm.resolve();
    // 4) 应用 SHAPE 声明的形状
    asm.infer_shapes();

    let bin = asm.emit_binary();
    std::fs::write(&args[2], &bin).expect("写不了 .vmb");
    println!(
        "OK: {} 条指令, {} 个常量, {} 个槽位, {} 字节 → {}",
        asm.text.len(),
        asm.consts.len(),
        asm.slots.len(),
        bin.len(),
        args[2]
    );

    // ---- 静默降级检查（Lead 加）----
    // 缺权重参数时，汇编器会把它们当「未声明的激活槽」处理，形状推出 0，
    // 然后产出一个 0 常量的坏 .vmb —— 而且打印 "OK"。
    // 这种静默降级只有端到端数值对比才能发现（Lead 亲历过一次）。
    // 这里做两个静态检查，把问题在汇编期就暴露出来。
    let n_zero_act = asm
        .slots
        .iter()
        .filter(|s| matches!(s.kind, Kind::Act) && s.rows == 0 && s.cols == 0)
        .count();
    let n_grad = asm.slots.iter().filter(|s| matches!(s.kind, Kind::Grad)).count();
    if asm.consts.len() == 0 {
        eprintln!(
            "⚠️  警告：0 个常量。若这个模型本该有权重，说明你漏了 `name:file.bin:rows:cols` 参数。"
        );
    }
    if n_zero_act > 0 {
        eprintln!(
            "⚠️  警告：{} 个激活槽形状仍是 0x0（无法推断）。\
             常见原因：槽位名拼错、缺 SHAPE 声明、或该槽只有写没有读。",
            n_zero_act
        );
    }
    if n_grad > 0 && asm.consts.len() == 0 {
        eprintln!("⚠️  警告：有梯度槽但没有常量 —— 梯度区会是空的，训练不会有任何效果。");
    }

    for (i, s) in asm.slots.iter().enumerate() {
        let k = match s.kind {
            Kind::Const => "const",
            Kind::Act => "act",
            Kind::State => "state",
            Kind::Grad => "grad",
        };
        println!("   slot{} {} [{}] {}x{}", i, s.name, k, s.rows, s.cols);
    }
}
