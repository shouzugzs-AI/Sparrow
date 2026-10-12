//! `vmlm-models` —— **L4 模型层**。
//!
//! 内容 = **B1 模型层**（从 `vmlm-blocks` 上移，逐字节搬运）：
//! * `b1`：稀疏层前向/反向（`par_b1_forward` / `par_b1_backward`）+ t8（source-major）整栈 scratch 集群
//! * `head`：块对齐 Head（`par_head_fwd/bwd`、`head_fwd_wt_one`、`par_head_fwd_wt`、`repack_head_wt`、`head_bwd_batched`）
//! * `attn_sparse`：稀疏投影 attention（消融二）
//!
//! 划分依据：这三个模块都以 B1 稀疏拓扑 `SparseConnW` 为参数 ⇒ 属**模型级**；
//! 而 `vmlm-blocks` 留下的稠密 attention 核心与通用原语是**通用块**。
//!
//! 依赖方向：`models → blocks`（L4 → L3，符合计划方向）+ `vmlm-ops` / `vmlm-sys`。
//! **尚存一条临时依赖 `models → vmlm-core`**（`conn_train::SparseConnW`、`b1_pure::*` 还没下移，
//! 见 `docs/抽库进展.md` 的 L4-2）。

pub mod attn_sparse;
pub mod model;
pub mod b1;
pub mod head;
