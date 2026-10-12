// 用**实测**确定 SIMD 重排的确切 lane 语义，而不是靠手推。
//
// 背景：P12 的 bf16 x16 变体两次写错（误差 7.06e-2 / 5.07e-1），
// 都是因为我凭记忆推断 `unpacklo_epi16` / `permute4x64` 的 lane 行为。
// `unpacklo_epi16` 在 AVX2 里是**按 128 位 lane 独立工作**的，很容易记错。
//
// 这个 bin 直接打印每条 shuffle 后的字节布局，一次就把真相钉死。
// （P11 的教训：测量/验证脚本必须先证明自己是对的。）

use std::arch::x86_64::*;

unsafe fn show(tag: &str, v: __m256i) {
    let mut buf = [0u16; 16];
    _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, v);
    print!("{:<28} u16: ", tag);
    for i in 0..16 {
        print!("{:3} ", buf[i]);
    }
    println!();
}

fn main() {
    unsafe {
        // 输入：u16 值 0..15，顺序排列
        let mut src = [0u16; 16];
        for i in 0..16 {
            src[i] = i as u16;
        }
        let packed = _mm256_loadu_si256(src.as_ptr() as *const __m256i);
        show("input", packed);

        // 模拟从 16 个 u16（即 16 个 bf16）里取偶/奇索引
        let mask_lo = _mm256_set1_epi32(0xFFFF);
        let lo16 = _mm256_and_si256(packed, mask_lo);
        let hi16 = _mm256_srli_epi32(packed, 16);
        show("lo16 = and 0xFFFF", lo16);
        show("hi16 = srli 16", hi16);

        let ulo = _mm256_unpacklo_epi16(lo16, hi16);
        let uhi = _mm256_unpackhi_epi16(lo16, hi16);
        show("unpacklo_epi16(lo,hi)", ulo);
        show("unpackhi_epi16(lo,hi)", uhi);

        // 目标：得到一个 u16 序列 [0,1,2,...,15]
        // 从上面两张表读出来，手工拼出正确组合：
        println!();
        println!("目标: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15");
        println!();
        println!("观察：lo16/hi16 把 u16 拆进了 32 位字的两个半区，");
        println!("      unpack 出来的是**带零洞**的布局，不是连续序列。");
        println!("      => 正确的路子是：不要拆半区，而是把 16 个 bf16 分成两个");
        println!("         128 位半区（各 8 个连续的 bf16），分别 cvtepu16_epi32。");
        println!();

        // 正确做法验证：低 128 位 = b0..b7，高 128 位 = b8..b15
        let lo128 = _mm256_castsi256_si128(packed);
        let hi128 = _mm256_extracti128_si256(packed, 1);
        let mut b = [0u16; 8];
        _mm_storeu_si128(b.as_mut_ptr() as *mut __m128i, lo128);
        print!("低 128 位 (= b0..b7)       u16: ");
        for i in 0..8 {
            print!("{:3} ", b[i]);
        }
        println!();
        _mm_storeu_si128(b.as_mut_ptr() as *mut __m128i, hi128);
        print!("高 128 位 (= b8..b15)      u16: ");
        for i in 0..8 {
            print!("{:3} ", b[i]);
        }
        println!();

        // 各自零扩展到 32 位 -> 8 个 u32（再左移 16 位就是 f32 位模式）
        let e0 = _mm256_cvtepu16_epi32(lo128);
        let e1 = _mm256_cvtepu16_epi32(hi128);
        let mut w = [0u32; 8];
        _mm256_storeu_si256(w.as_mut_ptr() as *mut __m256i, e0);
        print!("cvtepu16_epi32(低) -> u32 : ");
        for i in 0..8 {
            print!("{:3} ", w[i]);
        }
        println!();
        _mm256_storeu_si256(w.as_mut_ptr() as *mut __m256i, e1);
        print!("cvtepu16_epi32(高) -> u32 : ");
        for i in 0..8 {
            print!("{:3} ", w[i]);
        }
        println!();
        println!();
        println!("=> 结论：**一次 ymm load + 两次 cvtepu16_epi32 就得到连续 b0..b7 与 b8..b15**。");
        println!("   不需要 unpack，也不需要 permute。x16 变体应该这样写。");
        let _ = (ulo, uhi);
    }
}
