// P18 crash isolation: does kernels::gemm_nt fault on the trainer's exact shapes?
//
// Context: wiring gemm_nt into sparse_train24.rs replaced two scalar loops and
// the trainer then died with 0xC0000005 (exit -1073741819) immediately after
// "init: depth_scale=..." and before any step printed. The A/B probe had already
// called gemm_nt successfully on [128x1536]x[1536x1536], so the fault is
// shape- or layout-specific. This harness calls gemm_nt with EXACTLY the four
// shapes the trainer uses, one at a time, printing before each call so the
// faulting one is identifiable even though the process dies without a message.
//
// Key difference from the A/B probe that WORKED: that one used `acc=false` on a
// freshly-zeroed buffer for all four. The trainer additionally uses
// `acc=true` (Wo folding the residual). This probe tests both.
use vmlm_core::kernels::*;

extern "C" {
    fn malloc(n: usize) -> *mut u8;
}

unsafe fn alloc_f32(n: usize) -> *mut f32 {
    let p = malloc(n * 4) as *mut f32;
    if p.is_null() {
        println!("OOM n={}", n);
        std::process::exit(9);
    }
    for i in 0..n {
        *p.add(i) = 0.001 * ((i % 97) as f32);
    }
    p
}

fn main() {
    unsafe {
        vmlm_core::interp::init_cpu();
        println!("HAS_AVX2 = {}", vmlm_core::interp::HAS_AVX2);
    }
    let d = 1536usize;
    let ssz = 32usize;
    let vocab = 64usize;

    unsafe {
        let hl = alloc_f32(ssz * d);
        let wq = alloc_f32(d * d);
        let q = alloc_f32(ssz * d);
        let o = alloc_f32(ssz * d);
        let u = alloc_f32(ssz * d);
        let lm = alloc_f32(d * vocab);
        let lg = alloc_f32(ssz * vocab);

        // 1) qkv: m=ssz=32 (< 16? no, 32>=16; m*n*k = 32*1536*1536 = 75M >= 1M)
        //    => dispatch picks gemm_nt_packed
        println!("[1] gemm_nt(hl, wq, q, m=32, n=1536, k=1536, acc=false)  packed path");
        gemm_nt(hl, wq, q, ssz, d, d, false);
        println!("    ok");

        // 2) same but acc=true (the Wo shape)
        println!("[2] gemm_nt(o, wq, u, m=32, n=1536, k=1536, acc=true)   packed path");
        gemm_nt(o, wq, u, ssz, d, d, true);
        println!("    ok");

        // 3) lm_head-ish: n = vocab = 64 (multiple of 8, but small n)
        println!("[3] gemm_nt(hl, lm, lg, m=32, n=64, k=1536, acc=false)");
        gemm_nt(hl, lm, lg, ssz, vocab, d, false);
        println!("    ok");

        // 4) the exact A/B shape that already worked, for reference
        println!("[4] gemm_nt(hl, wq, q, m=128, n=1536, k=1536, acc=false)");
        let h128 = alloc_f32(128 * d);
        let q128 = alloc_f32(128 * d);
        gemm_nt(h128, wq, q128, 128, d, d, false);
        println!("    ok");

        println!("\nALL SHAPES OK - the fault is NOT in gemm_nt itself");
    }
}
