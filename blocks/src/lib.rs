//! `vmlm-blocks` —— **L3 通用网络块层**。
//!
//! 本层只被 trainer / `vmlm-models` 链接（引擎不链接它 ⇒ 引擎体积不受影响），且**保持逐位一致**：
//! 搬来的代码里除了 L4-1 记的那两处路径改写之外逐字未动。
//!
//! **依赖方向已回到计划方向**：`blocks` 只依赖 `vmlm-ops`（内核 + 调度）与 `vmlm-sys`
//! （`HAS_AVX2`），**不再依赖 `vmlm-core`** —— L3-1~L3-3 期间的临时反向依赖已在 L4-1 消除。
//!
//! 批次：
//! * L3-1 = attention 稠密块 + 通用原语（`attn.rs` / `util.rs`）
//! * L3-2/L3-3/L3-4 的 Head 块、稀疏层块、attention 稀疏包装在 L4-1 上移到 **L4 `vmlm-models`**
//!   （它们是模型级：以 B1 拓扑 `SparseConnW` 为参数）
//!
//! 见 `docs/抽库进展.md` 的 L3/L4 章。

pub mod attn;
pub mod util;
