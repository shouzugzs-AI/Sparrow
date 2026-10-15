//! Sparrow V0.1.2 P0-2 —— 读 `.vmb` 的 META 段（零依赖，手写解析）
//!
//! 为什么前端要认识 `.vmb` 的布局：循环化（`LOOP_CNT` + `LAYERB`）需要
//! "每层常量块的字节步长"，而**步长是汇编器的布局函数**。前端有两种做法：
//!
//! | 做法 | 风险 |
//! |---|---|
//! | 在前端复刻 `CONST_ALIGN` 规则算出来 | 汇编器改了对齐/顺序 → **静默算错**，而错步长只让第 1 层正确 |
//! | 汇编后**读出来**（这里） | 依赖 `.vmb` 格式，但格式本身是硬校验的（magic/version/长度） |
//!
//! 本模块走第二条，并且把"预测值 vs 实际值"做成**硬失败**（`main.rs` 的 E210）：
//! 前端仍然会算一份预测值（`--backend vml` 模式下没有汇编器可读），
//! 但一旦汇编器真的产出 `.vmb`，就必须拿实际值对账 —— 对不上就报错，
//! **不产出**一个步长未知的 `.vmb`。
//!
//! 唯一真相源：`vmlm/core/src/format.rs`（`MAGIC` / `VERSION` / `META_HDR` /
//! `CONST_ENT` / `CONST_ALIGN`）。这里的常量是它的镜像，一旦不一致，
//! 头部校验或长度表校验会失败，不会静默。

/// `"VMLM"` 小端。
pub const MAGIC: u32 = 0x4D4C_4D56;
pub const VERSION: u16 = 1;
/// 文件头固定 32 B：magic(4) + version(2) + pad(2) + 6×u32。
pub const HEADER: usize = 32;
/// META 头部 28 B。
pub const META_HDR: usize = 28;
/// 常量表每项 24 B。
pub const CONST_ENT: usize = 24;
/// 每个常量块补齐到 64 B。
pub const CONST_ALIGN: usize = 64;

/// 常量表的一项。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConstEnt {
    /// 在常量池里的**字节**偏移（池起点 = 0）。
    pub off: u32,
    /// 数据字节数（未补齐）。
    pub len: u32,
    pub rows: u32,
    pub cols: u32,
    pub dtype: u8,
}

fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// 把 `n` 向上补齐到 64 的倍数（汇编器 `emit_binary` 的 `while len % 64 { push(0) }`）。
pub fn align_up(n: u64, align: u64) -> u64 {
    if align == 0 {
        return n;
    }
    (n + align - 1) / align * align
}

/// 一个 f32 是否**精确**等于整数 `v`（用于给 `LAYERB` 的步长把关）。
///
/// 为什么必须查：`LAYERB` 的增量槽是 f32，`layer_base` 是字节偏移。
/// 若步长不是 f32 能精确表示的整数，`f32→usize` 会取到**相邻的另一个数**，
/// 于是从第 2 层起静默读错权重。步长是 64 的倍数且 < 2^30 时必然精确
/// （`stride/64 < 2^24`），所以这条检查在真实模型上恒真，但它是**免费**的。
pub fn f32_exact_int(v: u64) -> bool {
    let f = v as f32;
    f.is_finite() && f >= 0.0 && (f as u64) == v && (f as f64) == (v as f64)
}

/// 解析 `.vmb` 的常量表。任何一处不符就返回 `Err`（**不**做宽容解析）。
pub fn read_const_table(bytes: &[u8]) -> Result<Vec<ConstEnt>, String> {
    if bytes.len() < HEADER {
        return Err(format!("文件只有 {} B，连 32 B 头都不够", bytes.len()));
    }
    let magic = u32_at(bytes, 0);
    if magic != MAGIC {
        return Err(format!("magic 不是 VMLM（读到 0x{magic:08X}）"));
    }
    let ver = u16_at(bytes, 4);
    if ver != VERSION {
        return Err(format!("version 是 {ver}，只认识 {VERSION}"));
    }
    let text_len = u32_at(bytes, 12) as usize;
    let const_off = u32_at(bytes, 16) as usize;
    let const_len = u32_at(bytes, 20) as usize;
    let meta_off = u32_at(bytes, 24) as usize;
    let meta_len = u32_at(bytes, 28) as usize;
    if meta_off + meta_len > bytes.len() || meta_len < META_HDR {
        return Err(format!(
            "META 区间越界：off={meta_off} len={meta_len} 文件={}",
            bytes.len()
        ));
    }
    if const_off + const_len > bytes.len() {
        return Err(format!("常量池区间越界：off={const_off} len={const_len}"));
    }
    let meta = &bytes[meta_off..meta_off + meta_len];
    let n_const = u32_at(meta, 0) as usize;
    let n_slot = u32_at(meta, 4) as usize;
    if META_HDR + n_const * CONST_ENT > meta_len {
        return Err(format!(
            "常量表放不下：n_const={n_const} 需要 {} B，META 只有 {meta_len} B",
            META_HDR + n_const * CONST_ENT
        ));
    }
    let mut out = Vec::with_capacity(n_const);
    for i in 0..n_const {
        let o = META_HDR + i * CONST_ENT;
        let e = ConstEnt {
            off: u32_at(meta, o),
            len: u32_at(meta, o + 4),
            rows: u32_at(meta, o + 8),
            cols: u32_at(meta, o + 12),
            dtype: meta[o + 16],
        };
        // 每条常量块的 [off, off+len) 必须落在常量池里 —— 布局类错误里
        // 这也是"看起来合理"的那一类，所以逐条查。
        if e.off as usize + e.len as usize > const_len {
            return Err(format!(
                "第 {i} 条常量越界：off={} len={} 池={}",
                e.off, e.len, const_len
            ));
        }
        out.push(e);
    }
    let _ = (text_len, n_slot);
    Ok(out)
}

/// META 段的概要（槽位复用要拿它证明"激活区真的变小了"）。
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MetaInfo {
    pub n_const: u32,
    pub n_slot: u32,
    /// 激活区字节数（每个激活槽按最终形状 64 B 对齐紧密排布）。
    pub act_bytes: u32,
    pub state_bytes: u32,
    pub grad_bytes: u32,
    pub consts: Vec<ConstEnt>,
}

/// 解析 `.vmb` 的 META 概要 + 常量表。任何一处不符就 `Err`。
pub fn read_meta(bytes: &[u8]) -> Result<MetaInfo, String> {
    let consts = read_const_table(bytes)?;
    let meta_off = if bytes.len() >= HEADER {
        u32_at(bytes, 24) as usize
    } else {
        return Err("文件太短".into());
    };
    let meta = &bytes[meta_off..];
    Ok(MetaInfo {
        n_const: u32_at(meta, 0),
        n_slot: u32_at(meta, 4),
        act_bytes: u32_at(meta, 12),
        state_bytes: u32_at(meta, 16),
        grad_bytes: u32_at(meta, 20),
        consts,
    })
}

/// `.vmb` 的段界（调试/报错时打给人看）。
#[derive(Clone, Copy, Debug)]
pub struct Sections {
    pub text_off: usize,
    pub text_len: usize,
    pub const_off: usize,
    pub const_len: usize,
    pub meta_off: usize,
    pub meta_len: usize,
}

pub fn sections(bytes: &[u8]) -> Option<Sections> {
    if bytes.len() < HEADER {
        return None;
    }
    Some(Sections {
        text_off: u32_at(bytes, 8) as usize,
        text_len: u32_at(bytes, 12) as usize,
        const_off: u32_at(bytes, 16) as usize,
        const_len: u32_at(bytes, 20) as usize,
        meta_off: u32_at(bytes, 24) as usize,
        meta_len: u32_at(bytes, 28) as usize,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_matches_assembler_rule() {
        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(64, 64), 64);
        assert_eq!(align_up(65, 64), 128);
        assert_eq!(align_up(4 * 5, 64), 64);
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut b = vec![0u8; 64];
        b[0] = 1;
        assert!(read_const_table(&b).is_err());
    }

    #[test]
    fn f32_exactness() {
        assert!(f32_exact_int(0));
        assert!(f32_exact_int(64 * 1024));
        assert!(f32_exact_int(28_383_232));
        assert!(!f32_exact_int((1u64 << 24) + 1));
    }

    #[test]
    fn parses_meta_summary() {
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(&MAGIC.to_le_bytes());
        b.extend_from_slice(&VERSION.to_le_bytes());
        b.extend_from_slice(&0u16.to_le_bytes());
        b.extend_from_slice(&32u32.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&32u32.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&32u32.to_le_bytes()); // meta_off
        b.extend_from_slice(&28u32.to_le_bytes()); // meta_len（0 条常量）
        b.extend_from_slice(&0u32.to_le_bytes()); // n_const
        b.extend_from_slice(&7u32.to_le_bytes()); // n_slot
        b.extend_from_slice(&28u32.to_le_bytes()); // slots_off
        b.extend_from_slice(&1536u32.to_le_bytes()); // act_bytes
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        let m = read_meta(&b).expect("应解析成功");
        assert_eq!(m.n_slot, 7);
        assert_eq!(m.act_bytes, 1536);
        assert!(m.consts.is_empty());
    }

    #[test]
    fn parses_a_handmade_header() {
        // 头 + 1 条常量（4 B）+ META（28 B + 24 B）
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(&MAGIC.to_le_bytes());
        b.extend_from_slice(&VERSION.to_le_bytes());
        b.extend_from_slice(&0u16.to_le_bytes());
        b.extend_from_slice(&32u32.to_le_bytes()); // text_off
        b.extend_from_slice(&0u32.to_le_bytes()); // text_len
        b.extend_from_slice(&32u32.to_le_bytes()); // const_off
        b.extend_from_slice(&4u32.to_le_bytes()); // const_len
        b.extend_from_slice(&36u32.to_le_bytes()); // meta_off
        b.extend_from_slice(&(META_HDR as u32 + CONST_ENT as u32).to_le_bytes());
        assert_eq!(b.len(), HEADER);
        b.extend_from_slice(&7.0f32.to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes()); // n_const
        b.extend_from_slice(&0u32.to_le_bytes()); // n_slot
        b.extend_from_slice(&[0u8; 20]);
        assert_eq!(b.len(), 36 + META_HDR);
        b.extend_from_slice(&0u32.to_le_bytes()); // off
        b.extend_from_slice(&4u32.to_le_bytes()); // len
        b.extend_from_slice(&1u32.to_le_bytes()); // rows
        b.extend_from_slice(&1u32.to_le_bytes()); // cols
        b.push(0); // dtype f32
        b.extend_from_slice(&[0u8; 3]);
        b.extend_from_slice(&0u32.to_le_bytes());
        let t = read_const_table(&b).expect("应解析成功");
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].off, 0);
        assert_eq!(t[0].len, 4);
        assert_eq!(t[0].rows, 1);
        assert_eq!(t[0].cols, 1);
        assert_eq!(sections(&b).unwrap().meta_off, 36);
    }
}
