//! [抽库 L2a] 内核已搬到 `vmlm_ops::kernels`（逐字节搬运）。
//! 这里只重导出 ⇒ `vmlm_core::kernels::*` 的全部调用点（含 `b1_pure`、`interp`、`train`）零改动。
pub use vmlm_ops::kernels::*;
