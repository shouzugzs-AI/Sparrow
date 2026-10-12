//! [抽库 L4-2] `b1_pure`（B1 稀疏层内核族）已下移到 `vmlm_ops::b1_pure`（L2a）。
//! 这里只重导出 ⇒ `vmlm_core::b1_pure::*` 的全部调用点零改动。
pub use vmlm_ops::b1_pure::*;
