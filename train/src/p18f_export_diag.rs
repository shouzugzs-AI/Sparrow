// P18-F: WHY did the export read-back mismatch on EVERY connection?
//
// Context: `export_sparse_artifact` wrote each SparseConnW via
// `conn_sparse::write_section`, then read it back and compared src+weight
// bit-for-bit. Result: mismatches = 12800 = 640 neurons x 20 = EVERY connection.
//
// Reading `write_section` shows it does NOT take the caller's arrays at all --
// it REGENERATES both from the deterministic PRNG:
//     src     = neuron_conn_src(layout.layer, i, j, n, k0)
//     weight  = neuron_conn_weight_bits(layout.layer, i, j)
//
// So two hypotheses, and they have different fixes:
//   H1: the export wrote the INITIAL (untrained) topology, so the weights
//       legitimately differ from the trained ones -- but `src` should MATCH
//       (training never changes the topology), so this alone cannot explain
//       src mismatching too.
//   H2: the `layer` id used by write_section differs from the one the
//       SparseConnW was initialised with, so src differs as well.
//
// This probe distinguishes them by comparing, for a real SparseConnW:
//   (a) its src[]  vs neuron_conn_src(layer, i, j, n, k0)
//   (b) its w[]    vs neuron_conn_weight_bits(layer, i, j)
// If (a) matches and (b) does not, the topology is stable and only the
// trained weights need a serializer. If (a) also differs, the layer id or
// the seed path is wrong.
use vmlm_core::conn_sparse::{neuron_conn_src, neuron_conn_weight_bits};
use vmlm_core::conn_train::*;

extern "C" {
    fn malloc(n: usize) -> *mut u8;
}

unsafe fn alloc_f32(n: usize) -> *mut f32 {
    let p = malloc(n * 4) as *mut f32;
    if p.is_null() {
        println!("OOM");
        std::process::exit(9);
    }
    p
}

fn main() {
    let n: u32 = 64;
    let k0: u32 = 20;
    let layer: u32 = 7;

    unsafe {
        let base = malloc(SparseConnW::bytes(n, k0));
        let sc = SparseConnW::from_base(base, n, k0, layer);
        init_sparse_conn(&sc);

        println!("===== P18-F: SparseConnW vs the PRNG that write_section re-derives =====");
        println!("n={} k0={} layer={} e={}", n, k0, layer, sc.e);

        let mut src_bad = 0usize;
        let mut w_bad = 0usize;
        let mut total = 0usize;
        let mut first_report = 0usize;

        for i in 0..n {
            for j in 0..k0 {
                let idx = (i as usize) * (k0 as usize) + (j as usize);
                let got_s = *sc.src.add(idx) as u32;
                let want_s = neuron_conn_src(layer, i, j, n, k0);
                let got_w = (*sc.w.add(idx)).to_bits();
                let want_w = neuron_conn_weight_bits(layer, i, j);
                total += 1;
                if got_s != want_s {
                    src_bad += 1;
                    if first_report < 5 {
                        first_report += 1;
                        println!(
                            "  src mismatch i={} j={}: sc.src={} prng={}",
                            i, j, got_s, want_s
                        );
                    }
                }
                if got_w != want_w {
                    w_bad += 1;
                }
            }
        }

        println!("\n  total connections        = {}", total);
        println!("  src    mismatches        = {}", src_bad);
        println!("  weight mismatches        = {}", w_bad);
        println!();
        if src_bad == 0 && w_bad == 0 {
            println!("  => the SparseConnW IS exactly the PRNG's output at init.");
            println!("     So write_section round-trips the INITIAL state correctly,");
            println!("     and any post-training mismatch is purely 'trained != initial'.");
        } else if src_bad == 0 {
            println!("  => TOPOLOGY MATCHES, ONLY WEIGHTS DIFFER.");
            println!("     write_section can therefore persist the topology but NOT");
            println!("     the trained weights: there is no weight serializer.");
        } else {
            println!("  => EVEN THE TOPOLOGY DIFFERS. The layer id or the init path");
            println!("     used here does not match what write_section re-derives.");
        }

        // Second half: does init_sparse_conn_scaled change the topology or only w?
        let base2 = malloc(SparseConnW::bytes(n, k0));
        let sc2 = SparseConnW::from_base(base2, n, k0, layer);
        init_sparse_conn_scaled(&sc2, 0.144);
        let mut s2_bad = 0usize;
        let mut w2_bad = 0usize;
        for i in 0..n {
            for j in 0..k0 {
                let idx = (i as usize) * (k0 as usize) + (j as usize);
                if *sc2.src.add(idx) as u32 != neuron_conn_src(layer, i, j, n, k0) {
                    s2_bad += 1;
                }
                if (*sc2.w.add(idx)).to_bits() != neuron_conn_weight_bits(layer, i, j) {
                    w2_bad += 1;
                }
            }
        }
        println!("\n  after init_sparse_conn_scaled(0.144):");
        println!("    src mismatches = {}   weight mismatches = {}", s2_bad, w2_bad);
        if s2_bad == 0 && w2_bad > 0 {
            println!("    => scaling changes only the WEIGHTS (as documented),");
            println!("       so topology is stable across init variants.");
        }
        let _ = alloc_f32(1);
    }
}
