//! `vmlm-tensor` —— L1 张量层。
//!
//! 说明（纪律：只搬不改）：本仓库内核全部走裸指针，**没有** Tensor/Slice/View 抽象，
//! 也没有独立 arena 类型（对齐靠 `vmlm_sys::mem::alloc` + 内核里 64 字节 slack）。
//! 因此本层只收纳真实存在的原语，不新造抽象：
//! * `layout`：CSR section 字节布局（`ConnSparseLayout` + 头部常量 + 拓扑 PRNG 常量）
//! * `dtype`：bf16 ⇄ f32 转换、bf16 加宽（SIMD）
#![no_std]
#![allow(clippy::missing_safety_doc)]

pub mod act_op;
pub mod dtype;
pub mod layout;
