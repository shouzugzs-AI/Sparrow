//! [抽库 L4-2] B1 稀疏拓扑类型 `SparseConnW` + `impl`（从 `core/src/conn_train.rs` 逐字节搬运）。
//!
//! 逐字外改动：`crate::print` / `crate::exit` / `crate::interp::print_u32`
//! → `vmlm_sys::io::{print, exit, print_u32}`（L0）；新增 `use` 头。
#![allow(clippy::missing_safety_doc)]

#[derive(Clone, Copy)]
pub struct SparseConnW {
    /// 连接总数 E = n_neurons * k0。
    pub e: usize,
    /// 每神经元连接数（训练中固定）。
    pub k0: usize,
    /// 本层编号（喂给 neuron_conn_src 的确定性 PRNG）。
    pub layer: u32,
    /// 每神经元输出维度 = 源维度（B1 同维嵌入 => 层内方阵）。
    pub n_neurons: u32,
    /// 权重 [E]，行主序：w[i*k0 + j]。
    pub w: *mut f32,
    /// 源 id [E]（u32-as-f32，与引擎的索引惯例一致）。
    ///
    /// 为什么把 src 也存进 f32 数组而不是 u32：引擎全篇用 f32 携带索引
    /// （OP_EMB 的 row id、SPARSE_MATVEC 的 offsets 都是 f32），
    /// 混用 u32 会让 host 侧与引擎侧各写一份转换，而转换错位正是
    /// 「看似合理的垃圾数」的来源。索引 < 2^24 时 f32 精确。
    ///
    /// ⚠️ **本字段类型一字未改**（仍为 f32），因为它被 core/ 与 train/ 下
    /// **92 处**调用点按 f32 读取。B1 的 SIMD 内核需要 u32 索引
    /// （实测发布形态 1.43x，且 bit-exact），所以**另加 `src_u32` 镜像**，
    /// 而不是改本字段类型 —— 取舍记录见 `tools/b1_u32_decision.py`。
    pub src: *mut f32,
    /// 源 id [E] 的 **u32 镜像**，供 B1 的 SIMD 内核使用（省掉 `cvttps2dq`）。
    ///
    /// **存在理由（实测）**：`vmlm-b1-shipped-u32` 在**发布形态**上实测
    /// **1.43x**（0.534 -> 0.374 ns/conn），两种布局输出 **bit-exact 0/1536**。
    /// 代价 4 B/连接（每矩阵 120 KB；120 矩阵共 14.1 MiB，**仅训练侧工作区**；
    /// **磁盘产物不变**，它本来就是 u32：`CS_SRC_ENTRY = 4`）。
    ///
    /// 为什么用镜像而不是改 `src` 的类型：`src` 有 **92 处**调用点
    /// （含训练器、验收驱动、探针）。改类型要重写全部 92 处，而失败形态是
    /// **静默的索引错位** —— 正是本项目反复对抗的「看似合理的垃圾数」。
    /// 镜像的爆炸半径为零：既有读取方一行都不用改。
    ///
    /// **一致性契约**：`src` 在生产路径上**只有一处写入**（`init_sparse_conn`），
    /// 所以镜像的同步点是唯一的。由 `sync_src_u32()` 重建、
    /// `assert_src_u32_consistent()` 校验。若将来出现第二个写入点，
    /// **必须**同时更新镜像，否则断言会硬失败报出首个不一致位置。
    pub src_u32: *mut u32,
    /// 每神经元被 top-k 命中的连接掩码 [E]：1.0 = 激活，0.0 = 未激活。
    ///
    /// 这是机制 2 在训练侧的**唯一载体**：反向时梯度只写掩码为 1 的位置。
    /// 形状是 [E] 而不是 [n,k0] 的位图，理由与 src 相同（引擎侧统一 f32）。
    pub act: *mut f32,
}

impl SparseConnW {
    /// 分配的**字节数**（供 host 侧一次性分配，避免逐块分配）。
    pub fn bytes(n_neurons: u32, k0: u32) -> usize {
        let e = n_neurons as usize * k0 as usize;
        // w + src + act 各 4 字节，另加 src_u32 镜像 4 字节 => 4 项
        e * 4 * 4
    }

    /// 从 `src` 重建 u32 镜像。
    ///
    /// 索引 < 2^24 时 `f32 -> u32` **无损**，所以这一步不改变任何连接的身份
    /// —— 这正是 `vmlm-b1-shipped-u32` 里「身份一致性 0/30720」测的东西。
    ///
    /// # Safety
    /// `src` / `src_u32` 各至少 `e` 个元素且可读写。
    pub unsafe fn sync_src_u32(&self) {
        for t in 0..self.e {
            *self.src_u32.add(t) = *self.src.add(t) as u32;
        }
    }

    /// 校验镜像与 `src` 一致；不一致就**硬失败**并打印首个位置与两个值。
    ///
    /// 为什么不返回 bool：本项目口径是「报错但不说哪里错」平均要多花一轮
    /// 往返去二分（见 conn_sparse.rs 的同一段理由）。
    ///
    /// # Safety
    /// 同 `sync_src_u32`。
    pub unsafe fn assert_src_u32_consistent(&self) {
        for t in 0..self.e {
            let a = *self.src.add(t) as u32;
            let b = *self.src_u32.add(t);
            if a != b {
                vmlm_sys::io::print(b"CONN_TRAIN: src/src_u32 diverged at t=");
                vmlm_sys::io::print_u32(t as u32);
                vmlm_sys::io::print(b" src=");
                vmlm_sys::io::print_u32(a);
                vmlm_sys::io::print(b" mirror=");
                vmlm_sys::io::print_u32(b);
                vmlm_sys::io::print(b"\n");
                vmlm_sys::io::exit(9);
            }
        }
    }

    /// 校验结构几何自洽：`e == n_neurons * k0`。不一致就**硬失败**。
    ///
    /// 为什么需要它（实测事故，不是假想）：B1 的 SIMD 内核按 `k0` 以 **8 宽**
    /// 读 `src_u32`。若某个调用方构造出 `k0` 与缓冲区不符的 `SparseConnW`
    /// （例如把 `k0` 改成 `k0+1` 去做"错行宽"对照），最后一行会读越过
    /// `src_u32` 末尾 32 字节 —— 而 `src_u32` 位于分配区的**最后一块**，
    /// 于是直接 **0xC0000005，且没有任何诊断信息**（实测 `b1_canfail` 的
    /// wrong-5 就是这样崩的）。
    ///
    /// 旧标量版"侥幸"没崩：它一次读一个 f32，越界一格仍落在分配的余量里。
    /// **"旧代码活下来了"不是正确性证明**，所以这里补显式守卫，
    /// 把静默崩溃换成可读报错 —— 与 build_all.ps1 记录的那一类口径一致
    /// （栈帧过大跳过 guard page 也是同一类"没有诊断的崩溃"）。
    ///
    /// # Safety
    /// 无（只读自身字段）；标为 unsafe 只因 `vmlm_sys::io::print` / `vmlm_sys::io::exit`
    /// 在本 crate 里是 unsafe fn（它们直接走 kernel32 绑定）。
    /// 调用方仍须保证实际缓冲区至少有 `e` 个元素。
    pub unsafe fn assert_geometry(&self) {
        let want = self.n_neurons as usize * self.k0;
        if self.e != want {
            vmlm_sys::io::print(b"CONN_TRAIN: SparseConnW geometry inconsistent: e=");
            vmlm_sys::io::print_u32(self.e as u32);
            vmlm_sys::io::print(b" but n_neurons*k0=");
            vmlm_sys::io::print_u32(want as u32);
            vmlm_sys::io::print(b"\n");
            vmlm_sys::io::exit(9);
        }
    }

    /// # Safety
    /// `base` 至少 `Self::bytes(n_neurons, k0)` 字节且可写。
    pub unsafe fn from_base(
        base: *mut u8,
        n_neurons: u32,
        k0: u32,
        layer: u32,
    ) -> SparseConnW {
        let e = n_neurons as usize * k0 as usize;
        let w = base as *mut f32;
        let src = w.add(e);
        let act = src.add(e);
        // src_u32 镜像放在 act 之后（第四块）。布局顺序刻意保持
        // w/src/act 前三块不变，这样任何按偏移硬编码的既有代码不受影响。
        let src_u32 = (act.add(e)) as *mut u32;
        SparseConnW {
            e,
            k0: k0 as usize,
            layer,
            n_neurons,
            w,
            src,
            src_u32,
            act,
        }
    }

    #[inline]
    pub fn idx(&self, i: usize, j: usize) -> usize {
        i * self.k0 + j
    }
}
