// PATH: B1
//
// DIAGNOSIS of the A1 failure introduced by landing S1 (vectorized gather).
//
// THE SYMPTOM
//   After landing S1 in core/src/b1_pure.rs:
//     vmlm-lead-b1.exe   (my probe):  A1 bit mismatches = 0 / 1536   PASS
//     vmlm-b1-verify.exe (b1-builder): A1 bad = 1536 / 1536          FAIL
//   Two probes, same function, opposite verdicts. That means at least one
//   probe is not measuring what it claims -- exactly the failure class this
//   project keeps hitting.
//
// THE CANDIDATE DIFFERENCE
//   The two probes use DIFFERENT references:
//     lead:  compute_act_mask(k=k0) then sparse_accum
//     builder: sparse_forward(k=k0)   -- which internally does mask + accum
//   and they build their SparseConnW with different layer ids / scales:
//     lead:    from_base(..., layer=3), init_sparse_conn_scaled(&sc, 0.1)
//     builder: from_base(..., layer=7), init_sparse_conn_scaled(&sc, 0.144)
//
//   A layer-id difference changes the PRNG-derived topology AND weights, but it
//   changes them for BOTH sides equally, so it cannot by itself cause a
//   functional disagreement.
//
// THE REAL HYPOTHESIS TO TEST (this is the important one)
//   `sparse_strength_row_avx2` computes |w*x| -- it takes the ABSOLUTE VALUE.
//   It was written for the STRENGTH ROW, where |.| is the intended semantic.
//   S1 reuses it to compute the PRODUCTS for the forward sum, where the SIGN
//   must be preserved. If the kernel really returns |w*x|, then every negative
//   term has had its sign destroyed and A1 must fail on ~all rows.
//
//   That would make the S1 landing FLATLY WRONG -- not a subtle precision
//   issue, a sign bug. This probe measures it directly instead of reasoning
//   about it.
//
// USAGE: vmlm-b1-s1diag.exe [--d 1536] [--k0 20]

extern "C" {
    fn malloc(n: usize) -> *mut u8;
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

fn main() {
    unsafe { vmlm_core::interp::init_cpu() };
    let d = arg_u32("--d", 1536) as usize;
    let k0 = arg_u32("--k0", 20) as usize;

    println!("===== S1 diagnosis: does the reused kernel return |w*x| or w*x? =====");
    println!("d={} k0={}", d, k0);
    println!("HAS_AVX2 = {}", unsafe { vmlm_core::interp::HAS_AVX2 });

    unsafe {
        let e = d * k0;

        // Build a real SparseConnW exactly as the acceptance drivers do.
        let base = malloc(vmlm_core::conn_train::SparseConnW::bytes(d as u32, k0 as u32));
        let sc = vmlm_core::conn_train::SparseConnW::from_base(base, d as u32, k0 as u32, 7);
        vmlm_core::conn_train::init_sparse_conn_scaled(&sc, 0.144);

        let x = malloc(d * 4) as *mut f32;
        for i in 0..d {
            *x.add(i) = ((i * 2654435761usize) % 1000) as f32 / 1000.0 - 0.5;
        }

        // ---- TEST 1: compare the kernel's output against the plain scalar product
        let mut prod = vec![0.0f32; k0];
        let mut n_sign_mismatch = 0usize;
        let mut n_value_mismatch = 0usize;
        let mut examples = 0usize;

        for i in 0..d {
            let basei = i * k0;

            // scalar reference
            let mut refp = vec![0.0f32; k0];
            for j in 0..k0 {
                let sid = *sc.src.add(basei + j) as usize;
                refp[j] = *sc.w.add(basei + j) * *x.add(sid);
            }

            // the kernel actually used by the landed S1
            if vmlm_core::interp::HAS_AVX2 {
                vmlm_core::kernels::sparse_strength_row_avx2(
                    sc.w.add(basei),
                    sc.src.add(basei),
                    x,
                    k0,
                    prod.as_mut_ptr(),
                );
            } else {
                for j in 0..k0 {
                    let sid = *sc.src.add(basei + j) as usize;
                    prod[j] = (*sc.w.add(basei + j) * *x.add(sid)).abs();
                }
            }

            for j in 0..k0 {
                if prod[j].to_bits() != refp[j].to_bits() {
                    n_value_mismatch += 1;
                    // is it specifically a SIGN flip? (|a| == |b| but sign differs)
                    if prod[j].abs().to_bits() == refp[j].abs().to_bits() {
                        n_sign_mismatch += 1;
                    }
                    if examples < 5 {
                        println!(
                            "  e.g. i={} j={}: kernel={:+.9}  scalar={:+.9}   |equal|={}",
                            i,
                            j,
                            prod[j],
                            refp[j],
                            prod[j].abs().to_bits() == refp[j].abs().to_bits()
                        );
                        examples += 1;
                    }
                }
            }
        }

        println!("\n--- kernel vs scalar product, over {} values ---", e);
        println!("  value mismatches          = {}", n_value_mismatch);
        println!("  of which PURE SIGN flips  = {}", n_sign_mismatch);

        if n_sign_mismatch == n_value_mismatch && n_value_mismatch > 0 {
            println!("\n  VERDICT: the kernel returns |w*x|. It is the STRENGTH kernel.");
            println!("  Reusing it for the forward SUM is WRONG -- the sign is destroyed.");
            println!("  => S1 must NOT call sparse_strength_row_avx2 for the products.");
            println!("     A separate signed-product kernel is required.");
        } else if n_value_mismatch == 0 {
            println!("\n  VERDICT: the kernel returns the SIGNED product. S1 is sound");
            println!("  as far as this property goes; the A1 failure has another cause.");
        } else {
            println!("\n  VERDICT: mismatches exist but are NOT all sign flips.");
            println!("  Mixed cause -- needs per-case inspection.");
        }

        // ---- TEST 2: confirm on the ACTUAL failing comparison ----
        // Reproduce exactly what vmlm-b1-verify does, and report where they diverge.
        println!("\n--- end-to-end: S1 b1_forward vs the operator reference ---");
        let yb = malloc(d * 4) as *mut f32;
        let yo = malloc(d * 4) as *mut f32;
        vmlm_core::b1_pure::b1_forward(&sc, x, yb);

        // read act as the operator path left it (init set it all to 1.0, then
        // b1_forward did not touch it)
        let mut act_ones = 0usize;
        for t in 0..e {
            if *sc.act.add(t) == 1.0 {
                act_ones += 1;
            }
        }
        println!("  act[] == 1.0 count = {} / {} (operator path needs all-active)", act_ones, e);

        vmlm_core::conn_train::compute_act_mask(&sc, x, k0);
        vmlm_core::conn_train::sparse_accum(&sc, x, yo, k0);

        let mut bad = 0usize;
        for i in 0..d {
            if (*yb.add(i)).to_bits() != (*yo.add(i)).to_bits() {
                bad += 1;
                if bad <= 3 {
                    println!("  i={}: b1={:+.9}  op={:+.9}", i, *yb.add(i), *yo.add(i));
                }
            }
        }
        println!("  bit mismatches vs operator = {} / {}", bad, d);
        println!();
        if bad > 0 {
            println!("  The operator reference is the trusted one (it is the verified");
            println!("  engine path used in production). So the landed S1 is WRONG and");
            println!("  must be reverted or fixed.");
        } else {
            println!("  S1 agrees with the operator reference.");
        }
    }
}
