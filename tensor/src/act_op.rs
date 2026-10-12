//! L1 数值契约：激活类型与逐元素算子的常量（从 core/src/format.rs 逐字节搬运）。

pub const ACT_RELU: u16 = 0;
pub const ACT_GELU: u16 = 1;
pub const ACT_SILU: u16 = 2;
pub const ACT_SIGMOID: u16 = 3;
pub const ACT_TANH: u16 = 4;
pub const ACT_RELU2: u16 = 5;
pub const OP_EWADD: u8 = 20;
pub const OP_EWMUL: u8 = 21;
pub const OP_EWSUB: u8 = 22;
pub const OP_EWDIV: u8 = 23;
