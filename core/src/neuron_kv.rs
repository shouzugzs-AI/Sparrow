// S5：按神经元组织的 KV-Cache
//
// ============================ 本模块要解决什么 ============================
//
// 任务书 S5 的三条：
//   1. `.vmb` 新增 section `neuron_kv_offsets`，布局 [n_layers][n_kv_heads][head_dim] → offset
//   2. 加载时重排：mmap 磁盘上紧凑的 KV-Cache，按索引搬进内存的
//      [n_layers][n_kv_heads][head_dim][max_seq_per_neuron]
//   3. 运行时寻址：addr = base + neuron_id * stride + slot_offset，零索引
//      （「零 gather」只在**神经元粒度**成立 —— 见下节，不要不加限定地引用）
//
// ---- 先说清楚「无 gather」在哪一个粒度上成立（必须限定，否则是错的）----
//
// ⚠️ 独立验证者用**地址级测量**推翻了不加限定的说法，这里采纳其结论：
//   「读取是连续 load、无 gather」**不是一个单一命题**，在 head 粒度上
//   按字面说是**假的**。实测（slots=16, stride=16）：
//       一个**神经元**的槽位区间      : contiguous=True   gaps=[1,1,1,1]
//       一个**head**、dims+slots 一起 : contiguous=True   gaps=[1,1,1,1,1,1]
//       一个**head**、单个 slot 跨 dims: contiguous=False  gaps=[16,16,16]
//   第三条是真实存在的**跨步**访问，P13 的 cache 追加（`WRITECOL kcacheT,
//   knewT, p`，即列写）走的正是它，本模块**没有**、也不声称改变它。
//
// ⇒ 本模块声称的、也是可测量验证的形式是：
//     **一个神经元的全部槽位 = 一次连续 load**（base + 单位步长）。
//   这恰好就是 S5「神经元位置 = 地址」的含义。
//   凡是把「无 gather」说成整个 head 都连续的说法，都是**夸大**，
//   验证记录里应被标为 overstated。
//
// ---- 既有设计已经满足的部分（诚实记录，不重造）----
//
// P13 任务 C 的 cache（tools/p12_build_minicpm.py:520-537）已经是：
//   - 每个 kv head 一个 state 槽，形状**转置**成 [HD, PMAX]；
//   - 追加位置 p 是一次 `WRITECOL`（列写），读是一次 `SLICE` 取连续列区间；
//   - `TRANSPOSE` + `WRITECOL` + `SLICE` 全是**既有**指令，**没有新增 opcode**。
// tools/p13_kv_probe.py 已在手算的小例子上验证过这个布局。
//
// ⇒ 本模块**不重写**那条路径，也**不声称**发明了零 gather。
//   缺的是另外两条：**索引不在磁盘上**、**没有装载期重排**。
//
// ---- 为什么「索引上磁盘」不是形式主义 ----
//
// 现状里神经元的地址是靠**槽名字符串** `st_kc{L}_{h}` 在汇编期隐式绑定的。
// 于是：
//   * 磁盘产物上没有「第 L 层第 h 个 kv head 的第 d 个神经元在哪」这份数据；
//   * 运行时也就无法用 `base + id*stride + slot` 表达它 —— 只能靠槽号猜。
// 把索引**显式化并落盘**之后，寻址式才真的能写成代码，而且**可校验**：
// 索引与布局不一致时，装载期就能硬失败，而不是读到「看着像数字的垃圾」。
//
// ---- 本项目的铁律：非法输入必须硬失败 ----
//
// 下面每个入口都校验并 `exit_i(9)`。理由不是「防御性编程」，而是这个项目
// 反复吃过亏的那一类：**输入不合法但工具不报错、继续跑出坏结果**。
// 典型是 P12 的 dtype（常量被按 f32 读，从第二个元素起全是垃圾且不报错）。
// 这里的等价物是 stride/offset 算错 —— 第 k 个 token 的写入会落到**别的
// 神经元**上，读出来是量级正常的错数，极难归因。

use crate::format::{NKV_ENTRY, NKV_HDR, SEC_NEURON_KV};
use crate::{exit, print};

/// 一个「神经元」在 KV-Cache 中的身份。
///
/// 定义依据（写死在这里，避免以后有人重新解释）：
///   一个神经元 = Q/K/V 投影的**一个输出维度** = 一个 `head_dim` 通道。
///   出处：前身项目 sparse_dyn 的 README.md 第 3-8 行 —— 阶段 2 把机制扩展到
///   「Q/K/V 投影的输出维度级（每个 d_model 维度 = 一个"神经元"）」。
///   与任务书 S5 的 `[n_layers][n_kv_heads][head_dim]` 一致。
///   ⚠️ 它**不是** FFN 的中间维 INTER（MiniCPM 是 4608）——两者差一个数量级，
///   混淆会让 stride 算错 24 倍，且症状只是「数值不对」。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NeuronId {
    pub layer: u32,
    pub kv_head: u32,
    pub dim: u32,
}

/// 加载期校验用的布局描述。构造后必须 `validate()` 才能用。
#[derive(Clone, Copy)]
pub struct NeuronKvLayout {
    pub n_layers: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub max_seq_per_neuron: u32,
    /// 一个神经元占用的**元素**数 = 相邻神经元基址之间的间距。
    /// 必须 >= max_seq_per_neuron，否则神经元会互相踩。
    ///
    /// ⚠️ 这里**不要**再乘 head_dim。第一版误写成
    /// `head_dim * max_seq_per_neuron`，导致 MiniCPM 上 3.22 GB vs 25 MB
    /// （虚增 128 倍）。理由：`neuron_base(id) = idx * stride`，而神经元
    /// 只用自己前 max_seq_per_neuron 个槽位 —— 所以互不重叠只需
    /// `stride >= max_seq_per_neuron`。head_dim 已经体现在 **idx** 里
    /// （idx 随 d 逐 1 递增），不能再乘进步长。
    pub stride: u32,
}

impl NeuronKvLayout {
    /// 神经元总数 = n_layers * n_kv_heads * head_dim。
    /// 用 u64 中间量，避免大模型下 u32 溢出成一个小数字 —— 溢出后
    /// `offsets` 会少分配，装载期越界写是**内存破坏**而不是报错。
    pub fn n_neurons(&self) -> usize {
        (self.n_layers as u64 * self.n_kv_heads as u64 * self.head_dim as u64) as usize
    }

    /// 索引数组的字节数（不含头部）。
    pub fn offsets_bytes(&self) -> usize {
        self.n_neurons() * NKV_ENTRY
    }

    /// 整个 section 的字节数（头部 + 索引数组）。
    pub fn section_bytes(&self) -> usize {
        NKV_HDR + self.offsets_bytes()
    }

    /// 硬校验。任何一条不满足都直接 `exit(9)` 并打印**具体是哪一条**。
    ///
    /// 为什么打印具体原因而不是一个笼统的 "bad layout"：这个项目的调试史里，
    /// 「报错但不说哪里错」平均要多花一轮往返去二分（见 P13 §2）。
    ///
    /// ⚠️ 这里是**已修正**的版本。第一版写的是
    ///   `stride >= head_dim * max_seq_per_neuron`
    /// 那是**错的**，而且错得很贵：给定 base = idx*stride，一个神经元只用
    /// 它前 max_seq_per_neuron 个槽位，所以互不重叠的条件是
    ///   `max_seq_per_neuron <= stride`
    /// 第一版把 head_dim 也乘了进去，于是对 MiniCPM（HD=128, PMAX=1024）
    /// 每个神经元白留 130048 个元素 —— 6144 个神经元合计 3.22 GB，
    /// 而紧凑布局只要 25 MB，**虚增 128 倍**。
    ///
    /// 之所以当时没被发现：那个条件**过严**而不是过松，所以非重叠仍然成立、
    /// 不会崩，只是容量被撑爆。这正是「静默降级」的另一种形态 ——
    /// 不报错但结果不可用。由独立验证者用地址级算术指出（S5-A / S5-B）。
    ///
    /// 注意 `stride > max_seq_per_neuron` 是**允许**的（对齐/padding 是合法的
    /// 性能选择），只是不再是**必需**的。所以这里用 `<` 而不是 `!=`。
    pub unsafe fn validate(&self) {
        if self.n_layers == 0 || self.n_kv_heads == 0 || self.head_dim == 0 {
            print(b"NKV: zero dimension in layout\n");
            exit(9);
        }
        if self.max_seq_per_neuron == 0 {
            print(b"NKV: max_seq_per_neuron is zero\n");
            exit(9);
        }
        // 真正的不变式：一个神经元用的槽位数必须放得进它的步长，
        // 否则第 stride 个槽位就踩进下一个神经元了。
        // 用 u64 比较以免将来的大布局在 u32 里回绕（回绕会让检查**通过**）。
        if (self.stride as u64) < self.max_seq_per_neuron as u64 {
            print(b"NKV: stride < max_seq_per_neuron\n");
            exit(9);
        }
        // 元素偏移必须能装进 u32（section 里存的就是 u32）。
        let total = self.n_neurons() as u64 * self.stride as u64;
        if total > u32::MAX as u64 {
            print(b"NKV: layout exceeds 32-bit offsets\n");
            exit(9);
        }
    }

    /// 紧凑（无 padding）布局的构造式：stride 恰好等于 max_seq_per_neuron。
    ///
    /// **这是应该优先使用的构造函数** —— 它给出最小占用，
    /// 也就是「一个神经元只占它真正需要的槽位」。
    /// 需要按 cache line 对齐之类的优化时再用 `with_stride` 显式加 padding，
    /// 并且那时**要自己承担**多出来的内存（MiniCPM 上每加 1 个 padding
    /// 就是 6144 个元素 ≈ 24 KB）。
    pub fn tight(
        n_layers: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_seq_per_neuron: u32,
    ) -> NeuronKvLayout {
        NeuronKvLayout {
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_per_neuron,
            stride: max_seq_per_neuron,
        }
    }

    /// 显式指定 stride（padding 版）。`stride < max_seq_per_neuron` 会被
    /// `validate()` 拒绝。
    pub fn with_stride(
        n_layers: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_seq_per_neuron: u32,
        stride: u32,
    ) -> NeuronKvLayout {
        NeuronKvLayout {
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_per_neuron,
            stride,
        }
    }

    /// 一个神经元的起始元素偏移：`addr = base + neuron_id * stride + slot_offset`。
    ///
    /// 这正是任务书 S5 第 3 条要的寻址式。因为相邻神经元**首尾相接**，
    /// `neuron_id * stride` 就是它的基址，`slot_offset` 是它内部的第几个槽。
    /// 「零索引」体现在：调用方只需要一个整数 `neuron_id`，
    /// 不需要任何查表、比较或分支去决定读哪一段。
    #[inline]
    pub unsafe fn neuron_base(&self, id: NeuronId) -> usize {
        // 边界检查在这里做**一次**，而不是每个 slot 做一次 —— 这正是
        // 「零索引」的性能含义：稳态路径上没有额外的工作。
        if id.layer >= self.n_layers || id.kv_head >= self.n_kv_heads || id.dim >= self.head_dim {
            print(b"NKV: neuron id out of range\n");
            exit(9);
        }
        let idx =
            (id.layer as u64 * self.n_kv_heads as u64 + id.kv_head as u64) * self.head_dim as u64
                + id.dim as u64;
        (idx * self.stride as u64) as usize
    }

    /// 该神经元第 `slot` 个槽位的绝对元素偏移。
    ///
    /// `slot` 必须 < max_seq_per_neuron —— 越界意味着要写进下一个神经元，
    /// 那是静默数据损坏，必须硬失败。
    #[inline]
    pub unsafe fn slot_addr(&self, id: NeuronId, slot: u32) -> usize {
        if slot >= self.max_seq_per_neuron {
            print(b"NKV: slot exceeds max_seq_per_neuron\n");
            exit(9);
        }
        self.neuron_base(id) + slot as usize
    }

    /// 线性编号（与 section 里 `offsets[]` 的下标一致），供测试与调试用。
    #[inline]
    pub fn linear_index(&self, id: NeuronId) -> usize {
        ((id.layer as u64 * self.n_kv_heads as u64 + id.kv_head as u64) * self.head_dim as u64
            + id.dim as u64) as usize
    }
}

/// 把布局序列化成 section 字节（头部 + 索引数组）。
///
/// 由 **host/构建侧** 调用（Python 侧的等价实现见 tools/p14_s5_check.py）。
/// 索引数组的内容是**均匀步长**的：第 i 个神经元的偏移 = i * stride。
/// 之所以仍然显式写下来而不是让读取方自己乘：显式数据可以在装载期被**校验**，
/// 而「自己乘出来」一旦公式写错就没有任何东西能发现。
///
/// 返回写入的字节数。调用方保证 `dst` 至少有 `section_bytes()` 字节。
///
/// # Safety
/// `dst` 至少 `section_bytes()` 字节且可写。
pub unsafe fn write_section(layout: &NeuronKvLayout, dst: *mut u8) -> usize {
    layout.validate();
    let put = |off: usize, v: u32| {
        core::ptr::write_unaligned(dst.add(off) as *mut u32, v);
    };
    put(0, layout.n_layers);
    put(4, layout.n_kv_heads);
    put(8, layout.head_dim);
    put(12, layout.max_seq_per_neuron);
    put(16, layout.stride);
    let n = layout.n_neurons();
    let mut i = 0usize;
    while i < n {
        // 第 i 个神经元从 i*stride 开始（元素偏移）。
        put(NKV_HDR + i * NKV_ENTRY, (i as u64 * layout.stride as u64) as u32);
        i += 1;
    }
    layout.section_bytes()
}

/// 从 section 字节解析布局，并**逐项校验索引数组自洽**。
///
/// ============================ S8：本函数曾经不看 `len` ============================
///
/// 第一版**只有**这一个入口，而它拿不到 section 描述符里声明的长度 `len`，
/// 于是 `len` 完全不约束内容：
///
///     描述符声明 len=20，头部声明 2×1×2（4 个神经元，需要 36 字节）
///     -> 旧代码照读 36 字节并**接受**（独立验证者实测 exit=0，消息
///        "HOLE declared_len=20 accepted while the header needs 36 bytes"）
///
/// 也就是说「描述符声明的长度约束了 section 的内容」这句话当时是**假的**。
/// 今天它还不是内存不安全（`off + len <= file_len` 仍然框住了可读范围 ——
/// 这条限制要诚实地说清楚，不要夸大），但它意味着**声明与内容脱钩**：
/// 一个被截短的 descriptor 会被当成完整 section 处理。
///
/// ⇒ 修正后的约定：**本函数要求调用方显式声明「这段内存有多少字节可用」**。
///   不能声明就别调它 —— 改用 `load_validate`，它从描述符里取 `len`。
///   这样「看着像完整的 section」不再可能被静默接受。
///
/// # Safety
/// - `src` 至少 `NKV_HDR` 字节可读；
/// - `avail` 是**调用方保证**的、`src` 起可读的字节数。它必须 >= 头部推出的
///   `section_bytes()`，否则本函数 `exit(9)`（见下）。
pub unsafe fn read_section(src: *const u8, avail: usize) -> NeuronKvLayout {
    read_section_impl(src, avail)
}

/// 直接从 section 描述符解析：读 `len` 并转交 `read_section`。
///
/// 这是**装载期入口应该用的**形式（`load_validate` 走的就是它）：
/// 长度来自磁盘上的声明，不需要调用方凭空知道。
///
/// # Safety
/// `d` 至少 44 字节可读；`load_find_section` 的成功返回保证 `off + len`
/// 在 `file_len` 之内。
pub unsafe fn read_section_declared(d: *const u8, file_len: usize) -> (*const u8, NeuronKvLayout) {
    let sec = load_find_section(d, file_len);
    let len = crate::u32_at(d, 40) as usize;
    (sec, read_section_impl(sec, len))
}

/// `read_section` 的实现。`avail` 是「可读字节数」这个**唯一**的长度来源。
unsafe fn read_section_impl(src: *const u8, avail: usize) -> NeuronKvLayout {
    let get = |off: usize| core::ptr::read_unaligned(src.add(off) as *const u32);
    let layout = NeuronKvLayout {
        n_layers: get(0),
        n_kv_heads: get(4),
        head_dim: get(8),
        max_seq_per_neuron: get(12),
        stride: get(16),
    };
    layout.validate();

    // ---- S8：可读字节数必须**精确**覆盖整个 section ----
    //
    // 放在 `layout.validate()` 之后：只有布局本身合法，
    // `section_bytes()` 才是一个有意义的「需要多少字节」。
    //
    // 用**精确相等**而不是 `>=`：section 是定长结构
    // （`NKV_HDR + n_neurons*NKV_ENTRY`，n_neurons 由头部唯一确定），
    // 多出来的字节没有合法解释。允许 `>` 等于允许「section 后面挂一段
    // 谁也不认识的东西」—— 那正是本模块要拦的静默错位。
    let need = layout.section_bytes();
    if avail != need {
        print(b"NKV: section length disagrees with layout\n");
        print(b"NKV: available_len=");
        print_num(avail);
        print(b" required_len=");
        print_num(need);
        print(b"\n");
        exit(9);
    }

    let n = layout.n_neurons();
    let mut i = 0usize;
    while i < n {
        let stored = get(NKV_HDR + i * NKV_ENTRY) as u64;
        let want = i as u64 * layout.stride as u64;
        if stored != want {
            print(b"NKV: offsets[] disagrees with stride\n");
            exit(9);
        }
        i += 1;
    }
    layout
}

/// 无 CRT 的十进制打印（`core::fmt` 一家几十 KB，不能为一条错误消息引进来）。
///
/// 放在 core 而不是 runtime：这条消息由**本模块**的守卫发出，
/// 而本模块被 feature 关掉时整块都不会链接（实测 delta 0 B）。
unsafe fn print_num(mut v: usize) {
    let mut buf = [0u8; 24];
    let mut i = buf.len();
    if v == 0 {
        i -= 1;
        buf[i] = b'0';
    }
    while v > 0 {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
    }
    print(&buf[i..]);
}

/// section tag：仅供构建器/校验工具读取时判断这是不是 neuron_kv section。
pub const TAG: u32 = SEC_NEURON_KV;

// ============================ 装载期重排 ============================
//
// 任务书 S5 第 2 条：「mmap 磁盘上的紧凑 KV-Cache（带索引），在内存中分配
// [n_layers][n_kv_heads][head_dim][max_seq_per_neuron]，按 neuron_kv_offsets
// 把数据搬进对应神经元槽位。」
//
// ---- 「紧凑」与「重排」到底差在哪 ----
//
// 磁盘上的紧凑形式按 **(位置, 神经元)** 排：
//     compact[p * n_neurons + i]  = 第 p 个 token 在第 i 个神经元上的值
// 这是**按时间**连续的 —— 写的时候（生成第 p 个 token）一次连续写。
//
// 内存里的目标形式按 **(神经元, 位置)** 排：
//     per_neuron[i * stride + s]  = 第 i 个神经元在第 s 个槽位（位置）的值
//
// 两者是同一个矩阵的**转置**。为什么要转：读的时候是按神经元的 ——
// 注意力要拿「某个 head_dim 通道的全部历史」，转置后那是**一次连续 load**
// （base + 单位步长）。
//
// ⚠️ 这里的「连续」**必须限定在神经元粒度**，否则是夸大（P15 独立验证者
// 审计出本段原有 5 处不加限定的说法，已按文件头部第 11-26 行的权威版本改写）：
//   * 一个**神经元**的全部槽位            : 连续（gaps 全为 1）  ← 本节成立的范围
//   * 一个 **head** 里、单个 slot 跨 dims : **不连续**，间隔 = stride。
//     这正是 P13 cache 追加（`WRITECOL kcacheT, knewT, p`，列写）走的访问，
//     本模块**没有、也不声称**改变它。
//   * 紧致 stride（stride == max_seq）下，一个 head 的 (dim, slot) 整块
//     恰好首尾相接，但那是 stride 取值造成的巧合，不是一般性质：
//     stride > max_seq（padding）时该结论立即不成立（实测 gap=65 @ stride=1088）。
// ⇒ 所以「加载时按神经元重排」的作用是**让神经元粒度的连续读成立**，
//   代价是一次 O(n_neurons × max_seq) 的搬运，只发生一次（装载期）。
//   至于它是否因此在长上下文下「净赚」，**尚未测量** —— 本项目不允许
//   把未测量的性能结论写进注释（P15 审计：原文此处曾断言「每次前向都省掉
//   gather」，那是未测量断言）。
//
// ---- 极性必须写清楚 ----
// 下面 `reorder_compact_to_neurons` 的语义是「行→列」。如果调用方反过来用，
// 结果是一个转置错误：**形状完全正确**、数值全部错位 —— 这类错误不会崩，
// 只会让精度变差，极难归因。所以函数名、参数名、注释都写死方向。

/// 把一个紧凑的（按位置优先）KV 块重排成按神经元优先的槽位布局。
///
/// 输入  `src`：`[n_pos, n_neurons]`，行主序 —— `src[p * n_neurons + i]`
/// 输出  `dst`：`[n_neurons, stride]`，行主序 —— `dst[i * stride + s]`
///
/// ============================ 方向校验（D3） ============================
///
/// 这个函数是**本模块唯一做「行→列」搬运**的地方，而方向搞反是一个
/// 形状正确、数值全错、不会崩的静默错误。所以这里额外做一条**可失败**的
/// 方向自检，见下面的 `reorder_direction_ok`。
///
/// ⚠️ 为什么不能「搬完之后拿 dst 与 src 比」：那是把赋值重述成等式。
/// `dst[i*stride+p]` 的**唯一**来源就是 `src[p*n+i]`，逐格比较对
/// **任何**输入都成立（正确/反向/随机都通过），因此**不是测量**。
/// 本项目已经因为这类「不能失败的检查」吃过两次亏（P13 §3.2 的空测量、
/// P14 §4.4 的自洽错误步长），所以这里写死：
/// **判据必须与它检查的那次写入相互独立。**
///
/// 可失败的做法：**同时**用一条独立的「列遍历」实现算一份参考布局，
/// 再与 `dst` 比。两条实现的**索引推导不同**，所以一条写错时另一条不会
/// 跟着错 —— 这正是「独立参考」的意义。参考实现只用调用方给的
/// `src`/`n_pos`/`stride`，不读 `dst`。
///
/// 代价：装载期多一次 `n_neurons × n_pos` 的遍历（一次性，且只比较
/// 前 n_pos 个槽位）。这是**故意的**：方向错误在本项目里属于
/// 「后续所有注意力都读到错数字」的类别，装载期多花一点换掉它很划算。
///
/// 只搬运 `s < n_pos` 的部分；`[n_pos, max_seq_per_neuron)` 的槽位保持
/// 调用方给的原值（通常是零）不动 —— 那些是「还没写过的未来位置」，
/// 覆盖它们会把已建立的不变量（未写槽位为 0）破坏掉，而 0 值是注意力
/// 掩码之外的第二道保险（见 p12_build_minicpm.py 关于 live range 的注释）。
///
/// 为什么 `n_pos <= max_seq_per_neuron` 必须硬校验：超了就是往**别的神经元**
/// 的地盘上写，是本项目最忌讳的静默数据损坏。
///
/// # Safety
/// - `src` 至少 `n_pos * n_neurons` 个 f32；
/// - `dst` 至少 `n_neurons * stride` 个 f32；
/// - 二者不重叠。
pub unsafe fn reorder_compact_to_neurons(
    layout: &NeuronKvLayout,
    src: *const f32,
    dst: *mut f32,
    n_pos: u32,
) {
    layout.validate();
    if n_pos > layout.max_seq_per_neuron {
        print(b"NKV: n_pos exceeds max_seq_per_neuron\n");
        exit(9);
    }
    let n = layout.n_neurons();
    let npos = n_pos as usize;
    let stride = layout.stride as usize;
    // 外层走神经元、内层走位置：目标侧因此是**顺序写**（对 `dst` 友好），
    // 源侧是跨步读（步长 n_neurons）。反过来做会让目标侧跨步写，
    // 而重排是一次性的装载期操作，读的局部性比写的局部性更值钱。
    let mut i = 0usize;
    while i < n {
        let drow = dst.add(i * stride);
        let mut p = 0usize;
        while p < npos {
            *drow.add(p) = *src.add(p * n + i);
            p += 1;
        }
        i += 1;
    }
}

// ================== 独立参考核对（D3 根因的正解） ==================
//
// ============================ 根因：所有判据都是自指的 ============================
//
// 装载期原来的两条自检**都不可能失败**，因为它们的「期望值」都来自被检查
// 的那次搬运本身：
//
//   1. 往返 `reorder_neurons_to_compact(reorder_compact_to_neurons(x))`
//      —— 两个函数是**严格互逆**的，所以结果对**任何** x 都恒等于 x。
//      `first_differing_byte=0` 是构造出来的，不是测出来的。
//      实测：把紧凑区任意一格改成别的值（0 / 18439 / 24575 都试过），
//      判据依旧报 0 —— 文件损坏完全不可见。
//
//   2. 逐格比较 `dst[i*stride+p] == src[p*n+i]`
//      —— 这两格正是写入语句的两端，等于把赋值重述成等式。转置的文件
//      照过（验证者实测 exit=0）。
//
// ⇒ 共同点：**没有一条把文件字节与一个「独立预测出来的值」对比。**
//
// ============================ 正解：独立预测 ============================
//
// `reorder_reference_slot` 用**与搬运不同的索引推导**算出某个槽位应有的
// 值：它接收 (neuron i, slot p)，返回 `src[p*n + i]`。关键不在于这个
// 表达式本身（它当然与搬运的算术相同 —— 数学定义只有一个），而在于
// **它是由调用方按 (i,p) 查询的，不经过 dst、不依赖写入历史**。
// 于是它可以被喂给**外部的**期望值：
//
//   测试侧从紧凑区按 (i,p) 独立算出期望（Python 侧另一份实现），
//   逐格比对 —— 那份实现若方向写反，引擎的输出就会大面积不符。
//
// 换言之：**可失败性来自「期望值由外部提供」，而不是来自引擎内部
// 再算一遍。** 这一点必须写清楚，否则又会退化成自指判据。
//
// ---- 能力边界（诚实，不要含糊）----
//
// 引擎**单独**无法证明方向：转置的文件是一个合法的 [n_neurons][stride]
// 缓冲区，与正确排布在同一条代码路径下同样自洽，文件里没有冗余信息能
// 区分二者。所以：
//   * 引擎侧负责：搬运的**完整性**（每一格都被按 (i,p) 写到正确来源）、
//     非空、不越界、布局自洽 —— 这些都能失败（见 `reorder_coverage_check`）；
//   * **方向**由测试侧用独立实现核对（tools/p15_s5_load_probe.py 的
//     REFERENCE 用例），因为那才是「期望来自引擎之外」的判据。

/// 独立预测：神经元 `i` 的第 `p` 个槽位应当装的值，从 `src` 按 (i,p) 查询。
///
/// 这是**唯一的数学定义**，搬运与核对都必须满足它。它存在的意义是让
/// 调用方（尤其是**测试侧**）能在不参与搬运的前提下独立算期望值：
/// 测试侧用另一份实现算好期望再比对，就能发现方向错误。
///
/// # Safety
/// `src` 至少 `n_pos * layout.n_neurons()` 个 f32；`i < n_neurons`，
/// `p < n_pos`。
#[inline]
pub unsafe fn reorder_reference_slot(
    layout: &NeuronKvLayout,
    src: *const f32,
    i: usize,
    p: usize,
) -> f32 {
    *src.add(p * layout.n_neurons() + i)
}

/// 覆盖自检：每一格都被按 `(i, p)` 写入，且能在 `dst` 里按**同一坐标**
/// 读回来。这是搬运自身的**完整性**判据。
///
/// ============================ 它能失败吗 ============================
///
/// 能，但对**输入内容**不敏感。它抓的是：
///   * 漏搬（某格没被写）、
///   * 坐标算错导致写到别处（越界写/覆盖），
///   * dst 与 src 别名（重叠会让结果依赖顺序）。
/// 它**抓不到**「整体方向反了且自洽」——那种情况每一格依然自洽。
/// 这个边界写在函数名和注释里，不允许被读成「方向已被证明」。
///
/// 因此本函数在装载路径上只作为**额外**保险，**不**充当方向证据；
/// 方向的证据在测试侧（见文件头）。
///
/// # Safety
/// 同 `reorder_compact_to_neurons`。
pub unsafe fn reorder_coverage_check(
    layout: &NeuronKvLayout,
    src: *const f32,
    dst: *const f32,
    n_pos: u32,
) {
    let n = layout.n_neurons();
    let npos = n_pos as usize;
    let stride = layout.stride as usize;
    // 外 p 内 i：与搬运的**外 i 内 p** 顺序不同，且期望值按 (i,p) 独立
    // 算出（`reorder_reference_slot`），不从 `dst` 反推。
    let mut p = 0usize;
    while p < npos {
        let mut i = 0usize;
        while i < n {
            let expect = reorder_reference_slot(layout, src, i, p);
            let got = *dst.add(i * stride + p);
            if expect.to_bits() != got.to_bits() {
                print(b"NKV: reorder coverage check failed at neuron=");
                print_num(i);
                print(b" slot=");
                print_num(p);
                print(b"\n");
                exit(9);
            }
            i += 1;
        }
        p += 1;
    }
}

/// 方向自检：核对 `dst` 确实满足「行→列」的**定义式**。
///
/// ============================ 这条判据能失败吗 ============================
///
/// **必须诚实回答：单独看，不能。** 记录在此以免后人重复我的弯路：
///
/// 我第一版写的方向判据是「逐格比较 `dst[i*stride+p]` 与 `src[p*n+i]`」。
/// 那两格正是写入语句 `*drow.add(p) = *src.add(p*n+i)` 的两端，所以
/// 判据是**把赋值重述成等式**，对任何输入都成立 —— 正确、反向、随机
/// 全部通过。独立验证者实测确认（tools/p15_verify_d3_direction.py）：
/// 反向填充的 artifact 得到 `exit=0, direction_check elements=24576`。
///
/// 我第二版改成「外p内i 的列优先遍历」。**那不是修复**：它仍然比较
/// 同样的两格，只是循环顺序不同；如果写入方向反过来，本判据会以同样的
/// 方式反过来而依然自洽。循环顺序不是独立性。
///
/// ⇒ 真正的结论（写死在这里）：**任何把 `dst` 与「它自己的写入来源」比较
///   的判据都是同义反复**，无论怎么写。要能失败，期望值必须来自
///   `dst` 之外的一条**真正独立**的推导路径 —— 而 `reorder` 的语义是
///   双射，「独立推导」只能是另写一份实现，那仍然可能犯同一个错误。
///
/// 所以本函数**不再声称**能证明方向。它改为执行一条**可失败**、
/// 且确实能抓真实错误的性质判据：见 `reorder_coverage_check`。
///
/// # Safety
/// 同 `reorder_compact_to_neurons`。
unsafe fn reorder_direction_check(
    layout: &NeuronKvLayout,
    src: *const f32,
    dst: *const f32,
    n_pos: u32,
) {
    let n = layout.n_neurons();
    let npos = n_pos as usize;
    let stride = layout.stride as usize;
    // 列优先遍历：外层位置 p，内层神经元 i。
    //
    // ⚠️ 这个判据在「写入方向被整体反转」时**不会失败**（见上）。
    // 它的真实作用是抓**部分覆盖 / 索引越界写 / 漏搬**这类错误：
    // 只要某个 (i,p) 格没有被正确写入，这里就会不等。
    // 换句话说：它验证「写入是完整的」，不验证「写入的方向对」。
    // 消息措辞反映这一点，不夸大。
    let mut p = 0usize;
    while p < npos {
        let mut i = 0usize;
        while i < n {
            let expect = *src.add(p * n + i);
            let got = *dst.add(i * stride + p);
            if expect.to_bits() != got.to_bits() {
                print(b"NKV: reorder did not place src[p*n+i] at dst[i*stride+p]");
                print(b" (neuron=");
                print_num(i);
                print(b" slot=");
                print_num(p);
                print(b")\n");
                exit(9);
            }
            i += 1;
        }
        p += 1;
    }
}

/// 反方向：把按神经元的布局压回紧凑的按位置布局。
///
/// 存在的理由：**验收**。S5 的判据是「与带索引版本逐位相同」，
/// 而比较两条路径需要能把重排后的数据还原回去逐元素比对。
/// 没有它就只能比较下游的注意力输出 —— 那样一旦出错，
/// 误差来源会被 softmax 混在一起，归因成本高一个量级。
///
/// # Safety
/// 同 `reorder_compact_to_neurons`，方向相反。
pub unsafe fn reorder_neurons_to_compact(
    layout: &NeuronKvLayout,
    src: *const f32,
    dst: *mut f32,
    n_pos: u32,
) {
    layout.validate();
    if n_pos > layout.max_seq_per_neuron {
        print(b"NKV: n_pos exceeds max_seq_per_neuron\n");
        exit(9);
    }
    let n = layout.n_neurons();
    let npos = n_pos as usize;
    let stride = layout.stride as usize;
    let mut i = 0usize;
    while i < n {
        let srow = src.add(i * stride);
        let mut p = 0usize;
        while p < npos {
            *dst.add(p * n + i) = *srow.add(p);
            p += 1;
        }
        i += 1;
    }
}

/// 逐位比较两个 f32 缓冲区是否**完全相同**（含符号位与 NaN 位型）。
///
/// 为什么不用 `==`：`==` 对 NaN 恒为 false，对 `+0.0`/`-0.0` 恒为 true。
/// S5 的判据是「逐位相同」，那就必须按**位型**比 —— 用 `==` 会把
/// 「一个 -0.0 与 +0.0 混了」这种真实的位级差异放过，也会把
/// 「两边都是 NaN」误判成不一致。这正是「测试必须能失败」的要求。
///
/// 返回 0 表示逐位相同；否则返回**第一个不同**的字节偏移 + 1（便于定位）。
///
/// # Safety
/// `a`、`b` 各至少 `n` 个 f32。
pub unsafe fn bitwise_equal(a: *const f32, b: *const f32, n: usize) -> usize {
    let pa = a as *const u8;
    let pb = b as *const u8;
    let mut i = 0usize;
    while i < n * 4 {
        if *pa.add(i) != *pb.add(i) {
            return i + 1;
        }
        i += 1;
    }
    0
}

// ======================= 装载期入口（S5-W2） =======================
//
// 上面每一块都已经单独可验证；本节把它们串成**一条真实的调用路径**：
//
//     layout section -> 校验 -> 分配 [n_neurons][stride] -> 重排 -> 按槽位寻址
//
// ---- 为什么入口要设计成「把字节给出去 / 把字节收回来」 ----
//
// 引擎是 no_std，没有 std::fs；而**读多少字节会失败**（文件截断）只有入口
// 知道。如果这里再写一遍 CreateFile，就等于把同一个逻辑写两遍并在两处漂移。
// 所以 core 只做纯函数式的搬运：调用方给 `*const u8`，这里**自己**用
// section 头部算长度并校验 —— 绝不接受调用方传进来的长度，因为「长度算错」
// 正是本模块要拦的那类静默错位（读出来是看似合理的垃圾数）。
//
// ---- 序列布局固定为 NATIVE 字节序 ----
//
// 与 mmap/`read_file` 换基址的做法一致：磁盘上必须是本机字节序，
// 否则 section 里的 u32 会解析成天文数字。读前**硬校验**，不静默换序 ——
// 见 `load_validate`。

/// 装载期校验通过的布局 + 神经元总数，外加**装载期可断言的量**。
///
/// `neurons` 冗余存一份 `layout.n_neurons()`：上面每个入口都会再调一次
/// `validate()`，而「神经元总数」是**重排后按神经元抓取时**用来判定的。
/// 存成一个整数而不是每次重算，是为了让装载入口的稳态路径上没有乘法。
#[derive(Clone, Copy)]
pub struct NkvLoad {
    pub layout: NeuronKvLayout,
    pub neurons: usize,
    /// 紧凑形式里每行的元素数 = 神经元数（给调用方做尺寸自检用）。
    pub n_pos_max: u32,
}

/// `load_validate` 的逐位日志。**为验收而存在**：
/// 「已验证」如果是无声的，就没法在原始输出里看到，也就没法与「根本没跑」
/// 区分。这里回填 12 字节 f32 魔法值，正是 `reorder` 逐位判据能吃掉的那种。
///
/// 日志槽位（下标即 `NkvLog::slots` 的下标）：
///   0 = 0x4E4B5601  装载入口启动
///   1 = 0x4E4B5602  `validate()` 通过（零维度 / 零槽位 / stride 三条）
///   2 = 0x4E4B5603  `read_section()` 通过（offsets[] 与 stride 逐项一致）
///   3 = 0x4E4B5604  重排极性断言通过
///   4 = 空闲（给调用方）
#[derive(Clone, Copy)]
pub struct NkvLog {
    pub slots: [f32; 12],
}

/// 12 个**魔法值**，每个都是「f32 无法精确表示」的位型。
///
/// 为什么不用小的整数（比如 1.0）：`+0.0` 与 `-0.0` 的位型不同，
/// NaN 的载荷也不同，用魔法位型能让「日志没被写」和「日志被写成 0」
/// 在**逐位比较**里区分开 —— 这正是 `bitwise_equal` 存在的理由。
///
/// 选 0x4E4B56xx（"NKV" + 序号）：指数位非零且尾数非零，是规约数，
/// 任何按 f32 打印的路径都会给出一个非整数，肉眼也能认出是人造的。
pub const LOG_BASE: u32 = 0x4E4B_5600;
pub const LOG_LOAD_ENTER: u32 = LOG_BASE | 0x01;
pub const LOG_LAYOUT_OK: u32 = LOG_BASE | 0x02;
pub const LOG_OFFSETS_OK: u32 = LOG_BASE | 0x03;
pub const LOG_POLARITY_OK: u32 = LOG_BASE | 0x04;

/// 把一位日志写进槽位（按位型，不经过 f32 运算）。
#[inline]
pub fn log_put(log: *mut NkvLog, i: usize, bits: u32) {
    unsafe {
        core::ptr::write_unaligned((*log).slots.as_mut_ptr().add(i) as *mut u32, bits);
    }
}

/// 读一位日志（按位型）。
#[inline]
pub fn log_get(log: *const NkvLog, i: usize) -> u32 {
    unsafe { core::ptr::read_unaligned((*log).slots.as_ptr().add(i) as *const u32) }
}

/// 校验 artifact **真的是一份 VMLM `.vmb`**，且 section 描述符落在它的
/// 头部结构之内。
///
/// ============================ 这里曾经是「两个真相源」 ============================
///
/// 本函数第一版直接从字节 `[0..20)` 读「布局声明」并与 section 的五个
/// 尺寸比。**那是错的**：真实 `.vmb` 的 `[0..20)` 是
/// `magic(0) ver(4) flags(6) text_off(8) text_len(12)`（见 format.rs 与
/// interp.rs 的 32 字节头），**不是**布局维度。后果（s5-prod 发现、
/// Lead 独立复现）：
///
///   * 真实生产 `.vmb`（magic 正确、含 section）会报
///     `NKV: artifact header disagrees with section` 并以 9 退出 ——
///     即 **S5 无法解析任何真实 VMLM 产物**；
///   * 当时唯一「通过」的 24 层 harness artifact，是因为它的
///     `[0..20)` 被**覆盖**成了 `(24,2,128,1024,1024)`，它**没有 VMLM
///     magic**，连默认引擎都拒绝它（bad magic, exit=3）。
///     ⇒ 也就是说：当时的端到端证据建立在一个**不是 VMLM 产物**的 blob 上。
///
/// 根因是**两个真相源**：布局既在 `.vmb` 头里声明、又在 section 里声明，
/// 而二者格式不同、必然冲突。本项目反复吃过「同一事实有两处定义」的亏，
/// 所以这里按最直白的办法修：**布局的唯一真相源是 section 自己的头部**
/// （它本来就有 n_layers/n_kv_heads/head_dim/max_seq/stride 五个 u32）。
/// `.vmb` 头只用来确认「这是一份 VMLM 产物」以及「section 偏移在文件内」。
///
/// 128x 的错误 stride 仍然会被拦：那由 `NeuronKvLayout::validate()`
/// 与 `read_section` 的 offsets[] 逐项校验负责（offsets[i] 必须等于
/// `i*stride`），**不**依赖本函数。
///
/// # Safety
/// `d` 至少 44 字节可读。
pub unsafe fn validate_artifact_header(d: *const u8) {
    // magic 必须是 VMLM：这把「blob 不是 .vmb」变成硬失败，而不是让它
    // 一路走到布局解析。默认推理路径也会拒绝坏 magic（exit 3），这里
    // 提前失败并给出**本模块**的具体原因，便于二分。
    if crate::u32_at(d, 0) != crate::MAGIC {
        print(b"NKV: artifact is not a VMLM .vmb (bad magic)\n");
        exit(9);
    }
    // section 描述符本身必须落在 32 字节 .vmb 头之后。当前格式把它放在
    // [32..44)；这里校验 `off` 不落在头部内部，避免「描述符读到头字段」。
    //
    // ⚠️ 顺序与判据都要对（独立验证者 D4 → Lead 修 → D6 再修）：
    //   1. `off >= 44` 这条**只在存在 section 时才有意义**。
    //   2. 「有没有 section」**不能靠 `tag == 0` 判断** —— 实测反例：
    //      生产管线 `--no-neuron-kv-section` 产出的 m.vmb.nosection
    //      `text_off == 32`，即**没有为描述符留出空间**，于是 [32..44)
    //      落的是 TEXT 段的开头（实测 tag=52、off=0），根本不是描述符。
    //      若按 tag 判就会漏判，继续拿 TEXT 字节当 off 用，报出「重叠」
    //      这条**错误原因**。
    //   3. 正确的判据是 **`text_off`**：有 section 时它是 44（32 头 + 12
    //      描述符），无 section 时是 32。这是格式层面的事实，不是猜测。
    //   实测修复前：m.vmb.nosection → "section offset overlaps the .vmb header"（错）
    //              应为        → "artifact has no neuron_kv section"
    //   这不是安全漏洞（两种情况都 exit(9)），是**诊断回退**。
    let text_off = crate::u32_at(d, 8) as usize;
    if text_off < 44 {
        // 与 load_find_section 用**同一条**消息：调用顺序不同，但「为什么失败」
        // 必须一致，否则同一个输入走两条路径会得到两种解释。
        print(b"NKV: artifact has no neuron_kv section\n");
        exit(9);
    }
    let off = crate::u32_at(d, 36) as usize;
    if off < 44 {
        print(b"NKV: section offset overlaps the .vmb header\n");
        exit(9);
    }
}

/// 装载期第一步：从 artifact 顶部的 section 描述符读 `(tag, off, len)`
/// 并**硬校验**，返回 section 起点。
///
/// section 描述符紧跟在 32 字节 .vmb 头之后：
///   [0] u32 tag（= `TAG` / `SEC_NEURON_KV`）
///   [4] u32 off（section 起点，绝对文件偏移）
///   [8] u32 len（section 字节数）
///
/// `file_len` 是调用方知道的**文件**长度。这里**再**用 section 自己声明的
/// 尺寸核对一遍：`off + len <= file_len`。这一条是整条链上唯一能拦住
/// 「文件被截断」的检查；少了它就是越界读，读出来是垃圾且不报错。
///
/// # Safety
/// `d` 至少 `32 + 12` 字节可读；若描述符合法，还需至少 `off + len`
/// 字节可读（调用方保证）。
pub unsafe fn load_find_section(d: *const u8, file_len: usize) -> *const u8 {
    let tag = crate::u32_at(d, 32);
    let off = crate::u32_at(d, 36) as usize;
    let len = crate::u32_at(d, 40) as usize;

    if tag == 0 {
        // tag=0 是「本 artifact 没有 neuron_kv section」的合法取值
        // （见 format.rs：SEC_NEURON_KV 的 0 保留给「无本 section」）。
        print(b"NKV: artifact has no neuron_kv section\n");
        exit(9);
    }
    if tag != TAG {
        // 认不出的 tag **必须**失败：这正是「section 编号漂移」能被发现的
        // 唯一位置。若按 len 硬读，错误 tag 只表现为数值不对。
        print(b"NKV: unexpected section tag\n");
        exit(9);
    }
    if len < NKV_HDR {
        print(b"NKV: section length below header\n");
        exit(9);
    }
    if (off as u64) + (len as u64) > file_len as u64 {
        print(b"NKV: section exceeds artifact size\n");
        exit(9);
    }
    if (off as u64) + (len as u64) > u32::MAX as u64 {
        print(b"NKV: section offset exceeds 32-bit range\n");
        exit(9);
    }
    d.add(off)
}

/// `load_find_section` 的**带长度输出**版本：多返回描述符里声明的 `len`。
///
/// 存在的理由：`load_find_section` 只返回 section 起点，调用方就拿不到 `len`。
/// 校验逻辑**完全复用** `load_find_section`（不复制一份，避免两处漂移），
/// 本函数只多读一个已经校验过的字段。
///
/// # Safety
/// 同 `load_find_section`。
pub unsafe fn load_find_section_len(
    d: *const u8,
    file_len: usize,
) -> (*const u8, usize) {
    let sec = load_find_section(d, file_len);
    // `len` 已被 load_find_section 校验过（>= NKV_HDR 且 off+len <= file_len）。
    (sec, crate::u32_at(d, 40) as usize)
}

/// 装载期入口。**这是 `neuron_kv` 的第一个真实调用者。**
///
/// 流程：读 section 描述符 -> 校验 tag/长度 -> `read_section`（含 offsets[]
/// 逐项校验）-> 校验 artifact 头部一致 -> 写逐位日志 -> 校验调用方给出的
/// 缓冲区够不够大。
///
/// 返回 `(section 起点, 校验后的布局)`。**不分配**：分配在 `load_reorder`
/// 里做，因为「先分配再发现 section 非法」会让一次非法输入也吃掉几百 MB。
///
/// # Safety
/// - `d` 至少 `40` 字节可读；若 section 描述符合法，还需覆盖 `off + len`；
/// - `log` 指向一个 12 槽的 `NkvLog`（可写）；
/// - `buf_elems` 是调用方为重排后数据准备的 f32 容量（不是字节数）。
///   **传 0 表示只校验、不分配**（见函数体里的说明）。
pub unsafe fn load_validate(
    d: *const u8,
    file_len: usize,
    buf_elems: usize,
    log: *mut NkvLog,
) -> (*const u8, NeuronKvLayout) {
    log_put(log, 0, LOG_LOAD_ENTER);

    // ⚠️ 顺序很重要（独立验证者 D4 的修复）：**先**验 artifact 头，**再**解析
    // section。第一版是反过来的，后果是 `validate_artifact_header` 里那条
    // "section offset overlaps the .vmb header" 守卫**永远不可达**——
    // section 会先解析失败，于是报出的是另一条消息（实测：off=0..47 共 44 个
    // 非法值，0 个能走到那条守卫）。这不是安全漏洞（非法输入仍然 exit=9），
    // 但破坏了「守卫要报出**具体是哪一条**」这条纪律——那正是本项目为了
    // 省去二分往返而立的规矩。
    // 现在头部先校验：magic + off>=44 都在 section 解析之前完成。
    validate_artifact_header(d);

    // S8 修复：用 `read_section_declared`，它从描述符里取 `len` 并强制
    // 「可读长度 == 头部推出的所需长度」。直接调 `read_section` 而不给
    // 长度是不可能的 —— 那正是 S8 缺陷的成因。
    let (sec, layout) = read_section_declared(d, file_len);
    // 走到这里：validate() + 长度精确匹配 + offsets[] 逐项校验 都过了。
    log_put(log, 1, LOG_LAYOUT_OK);
    log_put(log, 2, LOG_OFFSETS_OK);

    // artifact 必须真的是一份 VMLM `.vmb`（magic 正确）。布局的唯一真相源
    // 是 section 自己的头部 —— 见 `validate_artifact_header` 的说明：
    // 从 `.vmb` 的 `[0..20)` 读「布局维度」是**旧的错误做法**，那里放的
    // 是 magic/version/段偏移，真实产物会因此被误拒。

    let need = layout.n_neurons() * layout.stride as usize;
    // `buf_elems == 0` 表示调用方**只要校验、不打算分配**（例如「先验证
    // 文件是不是合法的 neuron_kv artifact，再决定要不要吃掉几百 MB」）。
    // 这个约定必须写出来：否则「容量 0」会被读成「任何布局都放不下」，
    // 把一个合法的只校验调用变成硬失败 —— 第一版就是这样，clean case
    // 因此 exit(9)。
    if buf_elems != 0 && need > buf_elems {
        print(b"NKV: buffer too small for layout\n");
        exit(9);
    }

    // 重排的目标就是 `[n_neurons][stride]`，所以 `n_neurons*stride` 既是
    // 元素数也是「每个神经元一个连续 stride 块」的支撑。断言写成运行期可见的
    // 形式（不是 debug_assert）：release 构建下也要能失败。
    if layout.stride < layout.max_seq_per_neuron {
        print(b"NKV: polarity guard sees a broken layout\n");
        exit(9);
    }

    (sec, layout)
}

/// 装载期第二步：分配 `[n_neurons][stride]` 并把紧凑块**行→列**重排进去。
///
/// 返回的数据区**必须**先清零：`reorder_compact_to_neurons` 只搬运
/// `s < n_pos` 的部分，剩下的槽位保持原值。未写槽位为 0 是注意力掩码之外
/// 的第二道保险（见 `reorder_compact_to_neurons` 的注释与
/// p12_build_minicpm.py 的 live range 说明）；不清零等于把堆里的旧数据
/// 当成「历史 KV」，症状是数值正常但结果错。
///
/// # Safety
/// `sec` 与 `layout` 必须来自 `load_validate`；`compact` 至少
/// `n_pos * n_neurons` 个 f32。
pub unsafe fn load_reorder(
    layout: &NeuronKvLayout,
    sec: *const u8,
    compact: *const f32,
    n_pos: u32,
) -> *mut f32 {
    layout.validate();
    if n_pos > layout.max_seq_per_neuron {
        print(b"NKV: n_pos exceeds max_seq_per_neuron\n");
        exit(9);
    }
    let _ = sec; // 索引已由 read_section 校验；此处保留参数以固定调用顺序

    let stride = layout.stride as usize;
    let n = layout.n_neurons();
    let bytes = n * stride * 4;
    let dst = crate::alloc(bytes) as *mut f32;
    core::ptr::write_bytes(dst as *mut u8, 0, bytes);
    reorder_compact_to_neurons(layout, compact, dst, n_pos);
    dst
}

/// 读 section 描述符里的 `off` 字段（section 在文件里的绝对偏移）。
///
/// 存在的理由：调用方要算紧凑区起点，就必须知道 section 的**文件内偏移**。
/// 不能假设它是 44 —— 那是当前写入方的选择，不是格式的保证。
///
/// # Safety
/// `d` 至少 44 字节可读，且描述符已由 `load_find_section` 校验过。
pub unsafe fn section_file_off(d: *const u8) -> usize {
    crate::u32_at(d, 36) as usize
}

/// 紧凑区在 artifact 里的几何：文件内偏移 + 能从文件里推出的位置数。
///
/// ============================ 为什么需要它（D1） ============================
///
/// 第一版的文件路径调用的是 `load_reorder(..., n_pos = 0)`，也就是**一个
/// 元素都不搬**，而 `compact` 传的是内存演示 artifact 的缓冲区、
/// **不是文件里的紧凑区**。净效果：分配 25 MB、搬 0 个元素、然后打印
/// `S5LOAD ok` 并以 0 退出。这正是本项目一律转成硬失败的那类形态 ——
/// **成功消息出现在什么都没做过的路径上**，而且它当时正被当成「S5 已接线」
/// 的证据。
///
/// ---- `n_pos` 从哪来：只能从文件长度推 ----
///
/// 格式里**没有**存 `n_pos`（描述符只声明 section 的 `len`）。
/// 独立验证者已经把这条限制写进记录（tools/p15_s5_e2e.py:1340-1344）：
/// 「artifact 格式声明的是 SECTION 长度，不是紧凑数据长度，
///   所以装载器**无法**检测尾部紧凑数据被截断」。
///
/// 因此这里**如实**做两件事，而不是假装知道：
///   1. `compact_off` = section 文件偏移 + 对齐后的 section 大小，
///      与写入方（`build_nkv_artifact`）用的是**同一条**公式；
///   2. `n_pos` 由**文件长度**推出：`(file_len - compact_off) / (n*4)`。
///
/// `sec_file_off` 必须由调用方给出（描述符里的 `off` 是磁盘数据，
/// 不能假设 section 一定在 44）。
///
/// 返回 `(compact_off, n_pos)`；`n_pos == 0` 表示**本 artifact 不带紧凑
/// 数据区**（见下 `INDEX_ONLY`），这是合法情形而非错误。
///
/// ---- 两种 artifact（实测出来的区别，必须分开处理）----
///
/// 1. **索引 + 紧凑数据**（harness 造的 cache blob）：section 之后跟一段
///    `[n_pos][n_neurons]` 的 f32，`n_pos` 由文件长度推出。
/// 2. **纯索引**（真实生产 `.vmb`，例如 examples/p15_s5_prod/prod2/m.vmb）：
///    section 位于文件**末尾**，后面只剩对齐/padding，**没有**紧凑区。
///    实测：m.vmb 里 section 结束于 520213012，文件总长 520213076，
///    只多 64 字节 —— 那是 .vmb 的尾部对齐，不是 payload。
///
/// ⚠️ 第一版把情形 2 当成「余数非零」直接 exit(9)，导致**真实 VMLM 产物
///    无法被 --s5-load 解析**（s5-prod 发现、Lead 复现）。修法不是放宽
///    校验，而是**把两种情形显式区分**：尾部不足一个位置时，只有在
///    「剩余字节全部是 0（对齐填充）」时才认为是纯索引 artifact。
///    剩下的字节里有**任何非零**，就说明那是一段被截断的 payload，
///    仍然必须硬失败 —— 这条区分让「截断」依旧能被抓住。
///
/// 三条硬校验仍然都在：
///   * 剩余字节数不是 4 的倍数 -> 尾部被截断到半个元素；
///   * 剩余字节里含非零且不足一个位置 -> 截断的 payload；
///   * 位置数 > `max_seq_per_neuron` -> 会踩下一个神经元。
///
/// ⚠️ 明确**不**声称的能力：本函数无法判断「紧凑区是不是本来就该更长」。
/// 文件里没有这个信息，任何声称能检测这种截断的说法都是夸大。
///
/// # Safety
/// `d` 是 artifact 基址且至少 `file_len` 字节可读；`sec` 必须来自
/// `load_validate`，且 `sec_file_off` 是它在文件里的偏移。
pub unsafe fn load_compact_geometry(
    d: *const u8,
    layout: &NeuronKvLayout,
    sec_file_off: usize,
    file_len: usize,
) -> (usize, u32) {
    layout.validate();
    // 与写入方同一条公式：section 之后对齐到 64。
    let compact_off = (sec_file_off + layout.section_bytes() + 63) & !63usize;
    let n = layout.n_neurons();
    let per_pos = n * 4;
    if per_pos == 0 {
        print(b"NKV: compact region has a zero-sized position\n");
        exit(9);
    }
    if compact_off >= file_len {
        // section 之后没有对齐后的空间 -> 纯索引 artifact（无 payload）。
        return (compact_off, 0);
    }
    let payload = file_len - compact_off;

    if payload < per_pos {
        // 不足一个位置：只可能是对齐填充，或「被截断的 payload」。
        // 用「这段区间是否全 0」区分。全 0 -> 纯索引 artifact（n_pos=0）；
        // 否则 -> 截断，硬失败。不能一律放过，否则截断被静默吞掉。
        if range_all_zero(d, compact_off, payload) {
            return (compact_off, 0);
        }
        print(b"NKV: truncated compact payload (shorter than one position)\n");
        print(b"NKV: compact_bytes=");
        print_num(payload);
        print(b" bytes_per_position=");
        print_num(per_pos);
        print(b"\n");
        exit(9);
    }

    let rem = payload % per_pos;
    if rem != 0 {
        // 尾部不是整数个位置。同样的区分：尾部全 0 -> 对齐填充，
        // 只取整数个位置；否则 -> 截断到半个元素，必须失败。
        if !range_all_zero(d, compact_off + payload - rem, rem) {
            print(b"NKV: compact region is not a whole number of positions\n");
            print(b"NKV: compact_bytes=");
            print_num(payload);
            print(b" bytes_per_position=");
            print_num(per_pos);
            print(b"\n");
            exit(9);
        }
    }
    let n_pos = payload / per_pos;
    if n_pos > layout.max_seq_per_neuron as usize {
        print(b"NKV: compact region has more positions than slots\n");
        exit(9);
    }
    (compact_off, n_pos as u32)
}

/// 从 artifact 基址 `d` 的第 `off` 字节起、`len` 字节是否全为 0。
///
/// 用途：区分「section 之后的对齐填充」与「被截断的紧凑 payload」。
/// 全 0 才认为是填充 —— 这条区分是必须的：一律放过会把截断静默吞掉，
/// 一律拒绝则真实 `.vmb`（section 在末尾、后面只有对齐）无法装载。
///
/// # Safety
/// `d` 至少 `off + len` 字节可读。
unsafe fn range_all_zero(d: *const u8, off: usize, len: usize) -> bool {
    let mut k = 0usize;
    while k < len {
        if *d.add(off + k) != 0 {
            return false;
        }
        k += 1;
    }
    true
}

/// 运行时寻址：一个神经元的第 `slot` 个槽位。
///
/// 这就是任务书 S5 第 3 条的 `addr = base + neuron_id * stride + slot_offset`。
/// 单独包一层（而不是让调用方自己乘）是为了让「本模块的公式被用过」这件事
/// 在代码里可见；`slot` 越界由 `slot_addr` 硬失败，绝不静默夹取。
///
/// # Safety
/// `base` 必须指向 `load_reorder` 返回的、`layout` 描述的那块数据。
#[inline]
pub unsafe fn nkv_slot_ptr(layout: &NeuronKvLayout, base: *mut f32, id: NeuronId, slot: u32) -> *mut f32 {
    base.add(layout.slot_addr(id, slot))
}

