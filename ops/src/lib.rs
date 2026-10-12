//! `vmlm-ops` —— L2a 算子层。
//!
//! * `kernels`：GEMM / matvec / attention / norm / activation / loss 等内核（默认 no_std，引擎要用）
//! * `par`：多线程 region 切分与归约（`thread-std` feature 门控，只有 trainer 用）
//!
//! 兼容 shim：`format` / `interp` 两个模块把下层（`vmlm-tensor`、`vmlm-sys`）的符号重导出，
//! 使搬来的内核里 `use crate::format;`、`crate::interp::HAS_AVX2` 等路径**逐字不变**。
#![cfg_attr(not(feature = "thread-std"), no_std)]
#![allow(clippy::missing_safety_doc)]

pub mod kernels;

/// L4-2 下移：B1 稀疏拓扑类型 `SparseConnW` + 其 `impl`（原 `core/src/conn_train.rs`）。
#[cfg(feature = "conn-sparse")]
pub mod conn;
/// L4-2 下移：B1 稀疏层内核族（原 `core/src/b1_pure.rs`）。
#[cfg(feature = "conn-sparse")]
pub mod b1_pure;

/// shim：数值契约常量（ACT_*/OP_EW*）来自 L1。
pub mod format {
    pub use vmlm_tensor::act_op::*;
}

/// shim：CPU 能力位与 IO 原语来自 L0。
pub mod interp {
    pub use vmlm_sys::cpu::HAS_AVX2;
    pub use vmlm_sys::io::{exit_i, print_f32};
}

pub use vmlm_sys::io::print;
pub use vmlm_sys::mem::{alloc, dealloc};

#[cfg(feature = "thread-std")]
pub mod par;
