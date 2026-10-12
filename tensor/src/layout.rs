//! L1 布局：CSR section 字节布局与硬校验（从 core/src/conn_sparse.rs 逐字节搬运）。
//! 唯一新增：`use vmlm_sys::io::*;`（原文件里 print/print_num/exit 在同 crate 内可见）。
//! 另外 `read_u32`（小端读取，原在 conn_sparse.rs）随布局一起搬来。

use vmlm_sys::io::*;

/// section 头部字节数。五个 u32，**逐个字节偏移都在下面写死**：
///   [0]  u32 magic         = CS_MAGIC（"CSR1"）
///   [4]  u32 n_neurons     本 section 覆盖的神经元数
///   [8]  u32 k0            每神经元保留的连接数（0 = 不要求恰好 k0）
///   [12] u32 total_pairs   E = conn_src/conn_weights 的元素数
///   [16] u32 layer_id      本 section 属于哪一层（多 section 时用来配对）
///
/// ⚠️ 为什么头部要有自己的 magic：`n_neurons`/`k0`/`total_pairs` 都是
/// **看起来合理的整数**。一份「别的 section 的字节」被按本格式解析时，
/// 三个字段会得出一个自洽但错误的布局。magic 是唯一能把这种错配
/// 变成硬失败的东西（数值本身没有冗余）。
pub const CS_HDR: usize = 20;

/// 头部 magic："CSR1" 的小端 u32 = 'C' | 'S'<<8 | 'R'<<16 | '1'<<24。
///
/// 用 '1' 结尾（而不是 0）：版本演进时第二个版本用 "CSR2"，
/// 于是「版本漂移」也会在 magic 这一条上失败，而不是悄悄按 v1 解析。
pub const CS_MAGIC: u32 = 0x3152_5343;

/// 头部字段偏移（**给测试与 Python 侧引用，不要在各处重打字面量**）。
pub const CS_OFF_MAGIC: usize = 0;
pub const CS_OFF_N_NEURONS: usize = 4;
pub const CS_OFF_K0: usize = 8;
pub const CS_OFF_TOTAL_PAIRS: usize = 12;
pub const CS_OFF_LAYER: usize = 16;

/// `conn_src` 一项 = 一个 u32 神经元 id。
pub const CS_SRC_ENTRY: usize = 4;
/// `conn_weights` 一项 = 一个 f32（**按位存储**，不做任何数值转换）。
pub const CS_WGT_ENTRY: usize = 4;

/// 默认初始连接数 k0 = 20（任务书给定）。
///
/// 这个数**不是**可调参数随手改的：它是「连接稀疏」这条原则在存储层的
/// 上限表述，改了它整个 section 的字节数、以及原则 2 的算子循环上界都变。
/// 需要别的 k0 时用 `ConnSparseLayout::with_k0`，别改这里。
pub const K0_DEFAULT: u32 = 20;

/// 零索引寻址的步长式：神经元 i 的邻接表切片 = `conn_offsets[i]` 起。
///
/// 与 S5 的 `neuron_base` 同形（`addr = base + i*stride`），差别是 CSR 的
/// 每神经元长度**不等**，所以步长不是常量而是查表 —— 这正是 CSR 相对
/// 「定长记录」的代价，换来的是变长表达能力。稳态路径上它是
/// **一次减法 + 两次 load**，不是 O(k0) 循环。
#[inline]
pub fn conn_begin(offset_i: u32) -> usize {
    offset_i as usize
}

// ============================================================================
// 2. 布局 + 硬校验
// ============================================================================

/// 加载期/构建期共用的布局描述。构造后**必须** `validate()` 才能用。
#[derive(Clone, Copy)]
pub struct ConnSparseLayout {
    /// 本 section 覆盖的神经元数 n（= 层内 d_model 维度数）。
    pub n_neurons: u32,
    /// 每神经元保留的连接数。0 表示「不要求恰好 k0」（用于将来变长拓扑）。
    pub k0: u32,
    /// E = 连接总数 = conn_src / conn_weights 的元素数。
    pub total_pairs: u32,
    /// 本 section 属于哪一层（多 section 时配对用）。
    pub layer: u32,
}

impl ConnSparseLayout {
    /// 紧致构造式：**恰好** k0 条连接/神经元 ⇒ total_pairs = k0 * n_neurons。
    ///
    /// 这是应该优先使用的构造函数：它让 total_pairs 由前两个字段**导出**，
    /// 于是「头部自洽」（G7）是构造出来的，而不是靠调用方算对。
    /// 需要变长拓扑再用 `with_total_pairs`。
    pub fn tight(n_neurons: u32, k0: u32, layer: u32) -> ConnSparseLayout {
        ConnSparseLayout {
            n_neurons,
            k0,
            total_pairs: n_neurons.wrapping_mul(k0),
            layer,
        }
    }

    /// 显式指定 total_pairs。`k0 > 0` 时它会与 `k0 * n_neurons` 比对（G7）。
    pub fn with_total_pairs(
        n_neurons: u32,
        k0: u32,
        total_pairs: u32,
        layer: u32,
    ) -> ConnSparseLayout {
        ConnSparseLayout {
            n_neurons,
            k0,
            total_pairs,
            layer,
        }
    }

    /// conn_offsets 数组的元素数 = n_neurons + 1。
    ///
    /// 用 u64 再转 usize：`n_neurons` 是 u32，`+1` 在 n_neurons == u32::MAX
    /// 时会回绕成 0，于是「偏移表有 0 项」会通过后面所有的范围检查 ——
    /// 这是本项目最忌讳的那类**回绕让检查通过**。
    pub fn offsets_entries(&self) -> usize {
        (self.n_neurons as u64 + 1) as usize
    }

    /// conn_offsets 的字节数。
    pub fn offsets_bytes(&self) -> usize {
        ((self.n_neurons as u64 + 1) * 4) as usize
    }

    /// conn_src 的字节数。
    pub fn src_bytes(&self) -> usize {
        self.total_pairs as usize * CS_SRC_ENTRY
    }

    /// conn_weights 的字节数。
    pub fn weights_bytes(&self) -> usize {
        self.total_pairs as usize * CS_WGT_ENTRY
    }

    /// **整个 section 的字节数**。这是尺寸可校验性的唯一公式：
    /// 给定头部三个字段，需要多少字节**没有第二种答案**。
    pub fn section_bytes(&self) -> usize {
        CS_HDR + self.offsets_bytes() + self.src_bytes() + self.weights_bytes()
    }

    /// 头部中各数组的起始偏移（都在 section 内，绝对而非相对）。
    pub fn offsets_off(&self) -> usize {
        CS_HDR
    }
    pub fn src_off(&self) -> usize {
        CS_HDR + self.offsets_bytes()
    }
    pub fn weights_off(&self) -> usize {
        self.src_off() + self.src_bytes()
    }

    /// 硬校验。任何一条不满足都 `exit(9)` 并打印**具体是哪一条 + 数值**。
    ///
    /// 为什么打印具体原因而不是笼统的 "bad layout"：见 neuron_kv.rs 的
    /// 同一段理由 —— 「报错但不说哪里错」平均要多花一轮往返去二分。
    ///
    /// 注意本函数只校验**头部自身**的自洽性；「字节数是否匹配」由
    /// `read_section` 用 `section_bytes()` 校验（那里才拿得到 avail）。
    pub unsafe fn validate(&self) {
        // G2: 零神经元 / 零连接总数
        if self.n_neurons == 0 {
            print(b"CS: n_neurons is zero\n");
            exit(9);
        }
        // G3: total_pairs 上溢（u32）。total_pairs 存在头部 u32 里，
        //     溢出后它会**变成一个小数字**，于是 section_bytes() 也算小，
        //     整个 section 被截短成一个看起来合法的更小 section ——
        //     这正是本项目一律转成硬失败的那类静默降级。
        //     用 u64 中间量判断，不在 u32 里乘。
        if self.k0 != 0 {
            let want = self.k0 as u64 * self.n_neurons as u64;
            if want > u32::MAX as u64 {
                print(b"CS: total pair count overflows u32 (k0 * n_neurons)\n");
                exit(9);
            }
        }
        // G7: k0>0 时 total_pairs 必须**恰好**是 k0*n_neurons。
        //     这条是 S8 缺陷类的正面防线：如果只有 G1（长度 vs 头部），
        //     那么一个「n_neurons 与 total_pairs 同时被改小」的头部
        //     会自洽地通过。把 total_pairs 绑到 k0*n_neurons 之后，
        //     三个字段必须**同时**撒谎才可能自洽。
        if self.k0 != 0 {
            let want = (self.k0 as u64 * self.n_neurons as u64) as u32;
            if self.total_pairs != want {
                print(b"CS: total_pairs != k0 * n_neurons\n");
                print(b"CS: total_pairs=");
                print_num(self.total_pairs as usize);
                print(b" k0=");
                print_num(self.k0 as usize);
                print(b" n_neurons=");
                print_num(self.n_neurons as usize);
                print(b" expected=");
                print_num(want as usize);
                print(b"\n");
                exit(9);
            }
        }
        // G4: 整个 section 必须放得进 u32 的**字节偏移**空间。
        //     section 描述符里的 off/len 是 u32；超过之后 off+len 会回绕，
        //     而回绕会让「off+len <= file_len」这条检查**通过**。
        if self.section_bytes() as u64 > u32::MAX as u64 {
            print(b"CS: section exceeds 32-bit offsets\n");
            exit(9);
        }
    }

    /// 神经元 i 的邻接表切片 `[begin, end)`，元素下标（不是字节）。
    ///
    /// `offsets` 指向已校验的 conn_offsets 数组。本函数**重复**做边界检查
    /// （i < n_neurons）：调用方可能直接持有裸指针，而越界读在 CSR 上
    /// 表现为「读到邻居的连接」，静默且量级正常。
    ///
    /// # Safety
    /// `offsets` 至少 `offsets_entries()` 个 u32。
    #[inline]
    pub unsafe fn slice_of(&self, offsets: *const u32, i: u32) -> (usize, usize) {
        if i >= self.n_neurons {
            print(b"CS: neuron index out of range\n");
            exit(9);
        }
        let b = read_u32(offsets.add(i as usize)) as usize;
        let e = read_u32(offsets.add(i as usize + 1)) as usize;
        if b > e {
            print(b"CS: neighbor slice end < begin\n");
            exit(9);
        }
        (b, e)
    }
}

// ============================================================================
// 3. 确定性初始化（种子化 PRNG）

pub unsafe fn read_u32(p: *const u32) -> u32 {
    core::ptr::read_unaligned(p)
}
