
// [抽库 L2b] opcode 表搬到 vmlm-trans::opcode（`vmlm_core::format::OP_*` 路径不变）
pub use vmlm_trans::opcode::*;
// VMLM 格式常量：引擎与工具共享的唯一真相源
// 只放常量，不放逻辑 —— 引擎引用它，工具也引用它

pub const MAGIC: u32 = 0x4D4C_4D56; // "VMLM" LE
pub const VERSION: u16 = 1;

// 16 字节定长指令
pub const INS_SIZE: usize = 16;
pub const CONST_ALIGN: usize = 64;





                                //   dst = 跳转目标（标签，向后）
                                //   a   = 计数器槽（必须是 [1,N] 标量）
                                //   b   = 未使用（保留）




/// `OP_SPARSE_MATVEC` 的 fl 低位：第 4 个操作数（连接权重槽）的槽号。
///
/// 低 6 位放槽号，bit6 起留给标志 —— 因为 `fl` 是 u8，而引擎侧
/// `(ins.fl as u16) & FL_XXX` 的写法意味着**高位**才是既有标志位的地盘。
/// 把槽号放低位、标志放高位，两者不会互相污染。
pub const SPARSE_WSLOT_MASK: u8 = 0b0011_1111;
/// 槽号能编进 `fl` 低 6 位的上界；超过它汇编器硬报错（不静默截断）。
pub const SPARSE_WSLOT_MAX: u32 = SPARSE_WSLOT_MASK as u32;




// META 段布局：
//   [0]  u32 n_const      常量数
//   [4]  u32 n_slot       槽位数
//   [8]  u32 slots_off    （相对 META 起点）槽位表偏移
//   [12] u32 act_bytes    激活区字节数
//   [16] u32 state_bytes  状态区字节数
//   [20] u32 grad_bytes   梯度区字节数（与常量池同布局）
//   [24] u32 names_off    槽位名字哈希表偏移（相对 META 起点，n_slot × 4B）
//   [28] const_table: n_const × 24B
//   槽位表: n_slot × 16B { kind:u32, idx_or_off:u32, rows:u32, cols:u32 }
//     另存名字哈希表: n_slot × 4B（紧跟在槽位表之后），用于按名查找训练用的
//     激活/梯度/超参槽（const 的名字哈希已在常量表里）。
//     kind=0 常量槽 → idx 是常量表下标
//     kind=1 激活槽 → idx 是激活区字节偏移
//     kind=2 状态槽 → idx 是状态区字节偏移
//     kind=3 梯度槽 → idx 是梯度区字节偏移
//     kind=4 参数槽 → idx 是常量表下标（可训练，Adam/SGD 就地写）
pub const META_HDR: usize = 28;
pub const CONST_ENT: usize = 24;
pub const SLOT_ENT: usize = 16;

pub const SLOT_CONST: u32 = 0;
pub const SLOT_ACT: u32 = 1;
pub const SLOT_STATE: u32 = 2;
pub const SLOT_GRAD: u32 = 3; // 梯度槽：与常量池同布局的梯度区
pub const SLOT_PARAM: u32 = 4; // 参数槽：可训练常量（Adam/SGD 就地写）

// ---- dtype ----
//
// ⚠️ P12：这些编号**长期没有被引擎读过**，这是一个静默降级。
// 常量表条目里一直有 dtype 字段，但 `interp.rs` 从不检查它 ——
// 于是任何非 f32 的常量都会被**当成 f32 解释**，读出来是垃圾数，
// 而且不报错。MiniCPM 的权重正是 bf16，所以这是「实际加载模型」的硬阻塞。
//
// 现在：常量表每项带 dtype，装载时**按 dtype 决定如何取数**；
// 引擎不认识的 dtype 直接 `exit(9)`，不再静默按 f32 读。
pub const DT_F32: u8 = 0;
pub const DT_F16: u8 = 1;
pub const DT_I8: u8 = 2;
pub const DT_I32: u8 = 3;
/// bf16 = f32 的高 16 位（截断尾数）。转换**精确且零成本**：
/// 左移 16 位再按 f32 解释，不需要任何数学库。
/// 这是 MiniCPM / Llama 系的事实标准存储格式。
pub const DT_BF16: u8 = 4;

/// 该 dtype 每个元素的字节数（0 = 未知）。
///
/// 为什么需要它：常量池的偏移是**字节**偏移，而槽位表按元素寻址。
/// 若装载时不校验元素大小，bf16 权重会被按 f32 的步长走，
/// **第二个元素开始全错** —— 这是最隐蔽的一类错位。
pub const fn dtype_size(dt: u8) -> usize {
    match dt {
        DT_F32 => 4,
        DT_BF16 | DT_F16 => 2,
        DT_I8 => 1,
        DT_I32 => 4,
        _ => 0,
    }
}

// ---- 激活函数编号（fl 高位）----
pub const ACT_RELU: u16 = 0;
pub const ACT_GELU: u16 = 1;
pub const ACT_SILU: u16 = 2;
pub const ACT_SIGMOID: u16 = 3;
pub const ACT_TANH: u16 = 4;
pub const ACT_RELU2: u16 = 5;


// ============================ S5: 按神经元组织的 KV-Cache ============================
//
// 「神经元」在这里有一个必须写死的定义，否则整套布局没有意义：
//   一个**神经元** = Q/K/V 投影的一个输出维度 = 一个 `head_dim` 通道。
//   依据是参考实现 sparse_dyn/README.md 第 3-8 行：
//     「阶段 1 在 FFN 中间维度上验证机制；阶段 2 把同一套机制扩展到
//       Q/K/V 投影的输出维度级（每个 d_model 维度 = 一个"神经元"）。」
//   本节的布局 [n_layers][n_kv_heads][head_dim] 与任务书 S5 给的
//   `[n_layers][n_kv_heads][head_dim] → offset` **完全一致** —— head_dim
//   那一维就是神经元。它**不是** FFN 的中间维 INTER（4608）。
//
// ---- 为什么要有 section（而不是继续用 per-slot 的 SHAPE）----
// 现状（P13 任务 C，见 tools/p12_build_minicpm.py:520-537）：
//   每个 kv head 一个 state 槽 `st_kc{L}_{h}` / `st_vc{L}_{h}`，形状
//   **转置**成 [HD, PMAX]。转置的用意是「追加位置 p」退化成一次**列写**
//   （WRITECOL），读的时候是连续列切片 —— 这已经满足「连续 load、无 gather」。
//
//   缺的不是布局，是**索引**：神经元的地址从来没有被显式写下来过，
//   它是靠「槽名 st_kc{L}_{h}」这个字符串在汇编期隐式绑定的。
//   ⇒ 磁盘上没有索引，运行时也无法用 `base + id*stride + slot` 寻址，
//     而这两条正是 S5 的验收项。
//
// 本节把这个索引变成**磁盘上的显式数据**：
//   section 内容 = n_layers * n_kv_heads * head_dim 个 u32，
//   第 [L][h][d] 项的语义见 NKV_ENTRY 的文档。
//
// ---- 为什么偏移量是 u32 而不是指针 ----
// .vmb 是**可搬运**的产物（AOT 内嵌、mmap、拆权重文件都依赖这一点）。
// 存绝对地址会让同一份 .vmb 在换基址后静默指向别处 —— 那正是本项目
// 一律转成硬失败的那类静默降级。存**相对槽位/元素偏移**，装载时再加基址。
pub const SEC_NEURON_KV: u32 = 1; // section tag（0 保留 = 无本 section）

// ============================ P17: conn-sparse section tag ============================
//
// WHY THIS CONSTANT HAD TO MOVE HERE (P17 prerequisite, not a new feature):
//   `core/src/conn_sparse.rs` defined its own `SEC_CONN_SPARSE = 3` from P16 on,
//   but `format.rs` -- the file that owns the section tag table -- did not have
//   it. Consequence, measured rather than assumed: the ONLY artifact in the tree
//   carrying a conn section was `examples/p16_conn/conn_sparse.vmb`, a hand-built
//   example. **No production `.vmb` could carry one**, because the tag was not
//   part of the format's tag table; the gap is stated verbatim in P16_report.md
//   §8 ("生产 .vmb 携带 conn 段 ... 还缺一次接线").
//
// WHY THAT BLOCKS P17's TRAINING SIDE SPECIFICALLY:
//   Training-side "sparse connection weights" means a set of N=20 connections
//   per neuron whose WEIGHTS ARE TRAINED. Those weights have to live somewhere
//   that survives the training run and can be loaded again for inference. The
//   conn section is that place. Without the tag, T1's initialization and T2's
//   trained weights have no on-disk representation, so "trained sparse
//   connection weights" could not exist as an artifact -- only as in-memory
//   state inside one process.
//
// TAG VALUE 3 IS NOT FREE-CHOICE: it is already what `conn_sparse::SEC_CONN_SPARSE`
// says and what the P16 example artifact was written with. Re-numbering it would
// invalidate that artifact and the byte-for-byte cross-check against the
// independent Python emitter (tools/p16_conn_format.py). So this is a pure
// ADDITION to the tag table; no existing tag moves.
//
// 0 stays reserved = "no such section"; 1 = neuron_kv (S5); 2 is still unassigned.
pub const SEC_CONN_SPARSE: u32 = 3;

/// neuron_kv_offsets 的一项 = 一个**神经元**（= 一个 head_dim 通道，
/// 即 (layer L, kv head h, dim d) 三元组）在 KV-Cache 里的起始**元素**偏移。
///
/// 项数 = n_layers * n_kv_heads * head_dim，索引顺序为
///   idx = (L * n_kv_heads + h) * head_dim + d
/// 即 **行主序、head_dim 最内**，与上面 `[n_layers][n_kv_heads][head_dim]`
/// 的写法一致。
///
/// 不变式（由 `neuron_kv::NeuronKvLayout::validate` 在装载时**硬校验**）：
///   max_seq_per_neuron <= stride
/// 也就是「一个神经元用到的槽位数，放得进它自己的步长里」。因为
/// `base(i) = i * stride`，相邻两个 neuron 的区间分别是
/// `[i*stride, i*stride + max_seq)` 与 `[(i+1)*stride, ...)`，
/// 只要 `max_seq <= stride` 就必然不重叠。
///
/// ⚠️ 这里第一版写的是 `offset[idx] + max_seq <= stride` —— **那是错的**。
/// 由于 `offset[idx] = idx * stride`，该式在 idx>=1 时恒为假
/// （offset[1]+PMAX = 131072+1024 > 131072），却和实现里的另一个（也过严的）
/// 条件并存，等于文档与代码互相矛盾。由独立验证者以地址级算术指出（S5-B）。
/// 现在文档与 `validate()` 用**同一个**判据。
///
/// 允许 `stride > max_seq`（对齐/padding 是合法选择），所以是 `<=` 而非 `=`。
pub const NKV_ENTRY: usize = 4; // 每项 4 字节（u32）

/// neuron_kv_offsets section 的头部（在 u32 数组之前）。
/// 显式记下三个维度，而不是让读取方从别处推 —— 尺寸推错是静默错位，
/// 而错位的常量池会读成「看似合理的垃圾数」（见 P12 对 dtype 的教训）。
///   [0] u32 n_layers
///   [4] u32 n_kv_heads
///   [8] u32 head_dim
///   [12] u32 max_seq_per_neuron
///   [16] u32 stride          （元素数；见上面的不变式）
///   [20.. ] u32 offsets[n_layers * n_kv_heads * head_dim]
pub const NKV_HDR: usize = 20;

// FNV-1a 32 位：名字 → u32（工具与引擎必须一致）
pub const fn hash32(name: &str) -> u32 {
    let b = name.as_bytes();
    let mut h: u32 = 0x811c_9dc5;
    let mut i = 0;
    while i < b.len() {
        h ^= b[i] as u32;
        h = h.wrapping_mul(0x0100_0193);
        i += 1;
    }
    h
}
