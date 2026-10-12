//! `vmlm-trans` —— L2b 转译层。
//!
//! * `opcode`：opcode 表（**语义契约**，与 L2a `vmlm-ops` 共用同一张表：ops 通过
//!   `vmlm_tensor::act_op` 拿到逐元素算子常量，本层持有全部 73 个 opcode）
//! * `ins`：解释器/转译器共享的指令表示 `Ins`
//! * `jit`：字节码 → 机器码的代码生成器
//!
//! 兼容 shim：`format`（opcode + act_op）、`interp`（exit_i）以及根上的 `print`/`exit`/`alloc`/`dealloc`，
//! 使搬来的代码里 `use crate::format::*;`、`crate::print` 等路径**逐字不变**。
#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

pub mod ins;
pub mod jit;
pub mod opcode;

/// shim：opcode + 逐元素算子常量。
pub mod format {
    pub use crate::opcode::*;
    pub use vmlm_tensor::act_op::*;
}

/// shim：L0 的退出原语。
pub mod interp {
    pub use vmlm_sys::io::exit_i;
}

pub use vmlm_sys::io::{exit, print};
pub use vmlm_sys::mem::{alloc, dealloc};
