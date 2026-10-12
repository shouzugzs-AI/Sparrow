//! L2b 指令表示：解释器与转译器共享（从 core/src/interp.rs 逐字节搬运）。

///  16 ///
/// 8  docs`op:u8 | dt:u16 | fl:u8 | _pad:u8 | dst:u32 | a:u32 | b:u32`
///
/// **`dt` u8 u168 */// 4 `LN`LN_BWD`OUTER_BWD`4  `dt`///  `dt:u8`  0~255hidden * 256**
/// Fixed-size 16-byte instruction.
///
/// Field layout (P8 revision): `op:u8 | dt:u16 | fl:u8 | _pad:u8 | dst:u32 | a:u32 | b:u32`
///
/// `#[repr(C)]` IS LOAD-BEARING: the interpreter reinterprets the raw .vmb
/// TEXT bytes as `*const Ins`, so the in-memory field order MUST match the
/// file encoding exactly. Without it, Rust is free to reorder fields and the
/// engine silently decodes garbage (P10: this attribute was lost once during
/// an editing accident and every instruction read as op=0/HALT -- the engine
/// exited 0 having done nothing, which is the worst possible failure mode).
///
/// Why `dt` widened u8 -> u16 (P8 fix): four-operand instructions
/// (LN / LN_BWD / OUTER_BWD) pack the 4th operand into `dt`. At u8 that
/// silently wrapped for models with >255 slots (C2 measured 376 slots at
/// 3 layers, hidden=256) -> LN used the wrong bias slot. `fl` only uses
/// bits 0..5 so a u8 suffices, freeing the byte for `dt`; the instruction
/// stays 16 bytes so TEXT parsing is unchanged.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Ins {
    pub op: u8,
    pub fl: u8,
    pub dt: u16,
    pub dst: u32,
    pub a: u32,
    pub b: u32,
}
