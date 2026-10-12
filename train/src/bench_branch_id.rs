// PATH: DENSE
//
// B 基准：把「gemm_nt 实际走哪条分支」**测出来**，而不是读判定式猜。
//
// 为什么必须单独测分支（P18-B §6 的教训）
//   `gemm_nt` 的判定式用的是 **m = T**（A 的行数），不是 m = d。
//   在 T=128 时 m=128 >= 16 恒成立，所以 d=175/176 也走 packed 路径 ——
//   这一点与「d 小就落到非 packed 内核」的直觉相反。
//   本探针用三条**行为可观测**的证据来定分支，三者互不依赖：
//     (1) 结果位模式与 gemm_nt_packed_bench (强制 packed) 逐位相同？
//     (2) 结果位模式与 gemm_nt_avx2 (强制 unpacked 8x8) 逐位相同？
//     (3) 结果位模式与 gemm_nt_scalar (强制标量) 逐位相同？
//   只有「确实被调用的那条」才会贡献完全相同的位模式；
//   其余两条会因累加顺序不同而出现差异（这正是 profile_probe.rs 里
//   `bit-different=172990` 那一行所记录的同一现象）。
//
// USAGE: vmlm-branch-id.exe [--t 128] [--ds 1536,196,176,175,160]

// PATH: DENSE —— 只用 DENSE 参考内核，不引用 ref_sparse_a，不 use vmlm_core::sparse。

extern "C" {
    fn malloc(n: usize) -> *mut u8;
    fn free(p: *mut u8);
}

unsafe fn alloc_f32(n: usize) -> *mut f32 {
    let p = malloc(n * 4) as *mut f32;
    if p.is_null() {
        println!("OOM");
        std::process::exit(9);
    }
    p
}

unsafe fn fill_rand(p: *mut f32, n: usize, seed: u32) {
    let mut s = seed | 1;
    for i in 0..n {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *p.add(i) = ((s >> 8) as f32 / 16777216.0 - 0.5) * 0.04;
    }
}

fn arg_u32(flag: &str, def: u32) -> u32 {
    let v: Vec<String> = std::env::args().collect();
    for i in 0..v.len() {
        if v[i] == flag && i + 1 < v.len() {
            return v[i + 1].parse().unwrap_or(def);
        }
    }
    def
}

fn arg_str(flag: &str, def: &str) -> String {
    let v: Vec<String> = std::env::args().collect();
    for i in 0..v.len() {
        if v[i] == flag && i + 1 < v.len() {
            return v[i + 1].clone();
        }
    }
    def.to_string()
}

fn main() {
    unsafe {
        vmlm_core::interp::init_cpu();
    }
    let has_avx2 = unsafe { vmlm_core::interp::HAS_AVX2 };
    let t = arg_u32("--t", 128) as usize;
    let ds = arg_str("--ds", "1536,196,176,175,160");

    println!("=== gemm_nt branch identification by BIT PATTERN ===");
    println!("HAS_AVX2 = {}   T (m) = {}", has_avx2, t);
    println!(
        "判定式 (kernels.rs:1124) = n>=8 && k>=8 && m>=16 && m*n*k>=1e6, 其中 m=T={}",
        t
    );

    for tok in ds.split(',') {
        let d: usize = match tok.trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        unsafe {
            let a = alloc_f32(t * d);
            let b = alloc_f32(d * d);
            let c_real = alloc_f32(t * d);
            let c_packed = alloc_f32(t * d);
            let c_unpacked = alloc_f32(t * d);
            let c_scalar = alloc_f32(t * d);
            fill_rand(a, t * d, 11);
            fill_rand(b, d * d, 22);
            for i in 0..t * d {
                *c_real.add(i) = 0.0;
                *c_packed.add(i) = 0.0;
                *c_unpacked.add(i) = 0.0;
                *c_scalar.add(i) = 0.0;
            }

            // 真实入口：分支由 kernels.rs 内部决定
            vmlm_core::kernels::gemm_nt(a, b, c_real, t, d, d, false);
            // 三个「强制」对照
            vmlm_core::kernels::gemm_nt_packed_bench(a, b, c_packed, t, d, d, false);
            vmlm_core::kernels::gemm_nt_avx2(a, b, c_unpacked, t, d, d, false);
            vmlm_core::kernels::gemm_nt_scalar(a, b, c_scalar, t, d, d, false);

            let n = t * d;
            let mut diff_packed = 0usize;
            let mut diff_unpacked = 0usize;
            let mut diff_scalar = 0usize;
            let mut max_packed = 0.0f32;
            let mut max_unpacked = 0.0f32;
            let mut max_scalar = 0.0f32;
            for i in 0..n {
                let rv = *c_real.add(i);
                let pv = *c_packed.add(i);
                let uv = *c_unpacked.add(i);
                let sv = *c_scalar.add(i);
                if rv.to_bits() != pv.to_bits() {
                    diff_packed += 1;
                    if (rv - pv).abs() > max_packed {
                        max_packed = (rv - pv).abs();
                    }
                }
                if rv.to_bits() != uv.to_bits() {
                    diff_unpacked += 1;
                    if (rv - uv).abs() > max_unpacked {
                        max_unpacked = (rv - uv).abs();
                    }
                }
                if rv.to_bits() != sv.to_bits() {
                    diff_scalar += 1;
                    if (rv - sv).abs() > max_scalar {
                        max_scalar = (rv - sv).abs();
                    }
                }
            }

            // 判定：与哪条强制路径逐位一致
            let mut verdict = "UNKNOWN";
            if diff_packed == 0 {
                verdict = "packed (gemm_nt_packed_kernel)";
            } else if diff_unpacked == 0 {
                verdict = "unpacked 8x8 (gemm_nt_avx2)";
            } else if diff_scalar == 0 {
                verdict = "scalar (gemm_nt_scalar)";
            }

            // 另一个可判据：packed 恒等于 unpacked 时说明两条都被走过/退化了
            let packed_eq_unpacked = diff_packed == diff_unpacked;
            let macs = (t as u64) * (d as u64) * (d as u64);
            println!(
                "\n  d={:<5} m*n*k = {} ({:.2}M, >=1e6: {})   m>=16: {}   k>=8: {}",
                d,
                macs,
                macs as f64 / 1e6,
                macs >= 1_000_000,
                t >= 16,
                d >= 8
            );
            println!(
                "    bit-diff vs packed   = {:<8} max|d|={:.3e}{}",
                diff_packed,
                max_packed,
                if diff_packed == 0 { "  <-- IDENTICAL" } else { "" }
            );
            println!(
                "    bit-diff vs unpacked = {:<8} max|d|={:.3e}{}",
                diff_unpacked,
                max_unpacked,
                if diff_unpacked == 0 { "  <-- IDENTICAL" } else { "" }
            );
            println!(
                "    bit-diff vs scalar   = {:<8} max|d|={:.3e}{}",
                diff_scalar,
                max_scalar,
                if diff_scalar == 0 { "  <-- IDENTICAL" } else { "" }
            );
            println!(
                "    => VERDICT: gemm_nt actually took 【{}】   (packed==unpacked diff? {})",
                verdict, packed_eq_unpacked
            );

            free(a as *mut u8);
            free(b as *mut u8);
            free(c_real as *mut u8);
            free(c_packed as *mut u8);
            free(c_unpacked as *mut u8);
            free(c_scalar as *mut u8);
        }
    }
}
