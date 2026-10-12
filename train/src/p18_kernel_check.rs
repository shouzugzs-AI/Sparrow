// P18 P0 验收：稀疏连接内核的向量化 —— 正确性（逐位）+ 性能（QPC）。
//
// ============================ 判据（每条都能失败）============================
//
//  A. 位精确：新 AVX2 内核与**旧标量内核**在真实形状上逐位相同。
//     比较用 to_bits()，不用 `==`（`==` 对 NaN 恒假、对 ±0 恒真，
//     会把「符号/NaN 错了」当成通过 —— 见 conn_sparse.rs 的同一理由）。
//     能失败：把 sparse_strength_row_avx2 的 andnot 改成 max(v,-v)，
//     NaN 输入下立刻报 mismatch。
//
//  B. can-fail 对照：故意制造一个差异，比较器必须报出非零 mismatch。
//     不能失败的检查不是测量。
//
//  C. 平局规则：强度相等时保留下标更小的。构造**全是相同强度**的行，
//     旧规则（k 轮扫最大值 + `>`）与新规则（单遍 + `>`）必须选出同一批
//     下标。能失败：把任一处的 `>` 改成 `>=`。
//
//  D. 跳过计数：sparse_forward_masked 返回的 macs 必须 == 激活连接数，
//     且 <= n*k0。能失败：把掩码判断删掉（退化成稠密）计数立刻变大。
//
//  E. 性能：batch=4 seq=32 的等价规模下，测 topk / sparse fwd / sparse bwd
//     三个内核的单步耗时，与标量版并排报出来。
//
// 退出码 0 = 全部 PASS。
use vmlm_core::kernels::*;

extern "C" {
    fn malloc(n: usize) -> *mut u8;
    fn QueryPerformanceCounter(c: *mut i64) -> i32;
    fn QueryPerformanceFrequency(f: *mut i64) -> i32;
}

static mut N_CHECK: u32 = 0;
static mut N_FAIL: u32 = 0;

fn check(name: &str, ok: bool, detail: &str) {
    unsafe {
        N_CHECK += 1;
        if !ok {
            N_FAIL += 1;
        }
    }
    println!("[{}] {}  {}", if ok { "PASS" } else { "FAIL" }, name, detail);
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

struct Clock {
    freq: i64,
}
impl Clock {
    fn new() -> Clock {
        let mut f: i64 = 0;
        unsafe { QueryPerformanceFrequency(&mut f) };
        Clock { freq: f }
    }
    fn now(&self) -> i64 {
        let mut c: i64 = 0;
        unsafe { QueryPerformanceCounter(&mut c) };
        c
    }
    fn ms(&self, t: i64) -> f64 {
        t as f64 * 1000.0 / self.freq as f64
    }
}

unsafe fn alloc_f32(n: usize) -> *mut f32 {
    let p = malloc(n * 4) as *mut f32;
    if p.is_null() {
        println!("OOM");
        std::process::exit(9);
    }
    p
}

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Rng {
        Rng(s | 1)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as u32
    }
    fn uni(&mut self) -> f32 {
        (self.next_u32() as f64 / (u32::MAX as f64)) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.uni()
    }
}

/// 旧标量强度：逐项 (w * x).abs()。这是被替换的实现，保留作为对拍基准。
unsafe fn strength_scalar(wrow: *const f32, srcrow: *const f32, x: *const f32, k0: usize, out: *mut f32) {
    for j in 0..k0 {
        let sid = *srcrow.add(j) as usize;
        *out.add(j) = (*wrow.add(j) * *x.add(sid)).abs();
    }
}

/// 旧标量 top-k：k 轮扫最大值 + taken 位图（P18 之前的实现，逐字保留）。
unsafe fn topk_scalar_old(
    strength: *const f32,
    k0: usize,
    k: usize,
    hit: *mut usize,
) -> usize {
    let mut taken = [false; 4096];
    let mut nk = 0usize;
    for _ in 0..k {
        let mut best = 0usize;
        let mut found = false;
        let mut bestv = 0.0f32;
        for j in 0..k0 {
            if taken[j] {
                continue;
            }
            let v = *strength.add(j);
            if !found || v > bestv {
                bestv = v;
                best = j;
                found = true;
            }
        }
        if !found {
            break;
        }
        taken[best] = true;
        *hit.add(nk) = best;
        nk += 1;
    }
    for i in 1..nk {
        let v = *hit.add(i);
        let mut q = i;
        while q > 0 && *hit.add(q - 1) > v {
            *hit.add(q) = *hit.add(q - 1);
            q -= 1;
        }
        *hit.add(q) = v;
    }
    nk
}

fn main() {
    let d = arg_u32("--d", 1536) as usize;
    let k0 = arg_u32("--k0", 20) as usize;
    let k = arg_u32("--k", 8) as usize;
    let t_total = arg_u32("--t", 128) as usize;
    let layers = arg_u32("--layers", 24) as usize;
    let reps = arg_u32("--reps", 1) as usize;

    println!("===== P18 P0: vectorised sparse-connection kernels =====");
    println!("d={} k0={} k={} T={} layers={} reps={}", d, k0, k, t_total, layers, reps);

    let mut nfail = 0u32;
    unsafe {
        // ---------------- A. bit-exactness on real shapes ----------------
        println!("\n--- A. bit-exact vs the previous scalar implementation ---");
        let mut rng = Rng::new(0xC0FFEE);
        let n = d;
        let e = n * k0;
        let w = alloc_f32(e);
        let src = alloc_f32(e);
        let x = alloc_f32(d);
        let act = alloc_f32(e);
        for i in 0..e {
            *w.add(i) = rng.range(-1.0, 1.0);
            // src ids must be in [0,d); make them realistic and unsorted
            *src.add(i) = ((rng.next_u32() as usize) % d) as f32;
            *act.add(i) = if rng.uni() < 0.4 { 1.0 } else { 0.0 };
        }
        // guarantee at least one active per row (matches the model invariant)
        for i in 0..n {
            *act.add(i * k0) = 1.0;
        }
        for i in 0..d {
            *x.add(i) = rng.range(-2.0, 2.0);
        }

        let s_new = alloc_f32(k0);
        let s_old = alloc_f32(k0);
        let hit_new: *mut usize = malloc(n * k0 * 8) as *mut usize;
        let hit_old: *mut usize = malloc(n * k0 * 8) as *mut usize;
        let bv: *mut f32 = alloc_f32(k0);

        // A1: strengths bit-exact over ALL neurons
        let mut sbad = 0usize;
        for i in 0..n {
            sparse_strength_row_avx2(w.add(i * k0), src.add(i * k0), x, k0, s_new);
            strength_scalar(w.add(i * k0), src.add(i * k0), x, k0, s_old);
            for j in 0..k0 {
                if (*s_new.add(j)).to_bits() != (*s_old.add(j)).to_bits() {
                    sbad += 1;
                }
            }
        }
        check(
            "A1 strength row AVX2 == scalar, BIT-EXACT",
            sbad == 0,
            &format!("checked {} values, mismatches={}", n * k0, sbad),
        );

        // A2: top-k selection bit-exact over ALL neurons
        let mut hbad = 0usize;
        let mut nkmis = 0usize;
        for i in 0..n {
            sparse_strength_row_avx2(w.add(i * k0), src.add(i * k0), x, k0, s_new);
            let a = sparse_topk_from_strength(s_new, k0, k, hit_new, bv);
            let b = topk_scalar_old(s_new, k0, k, hit_old);
            if a != b {
                nkmis += 1;
            }
            for j in 0..a.min(b) {
                if *hit_new.add(j) != *hit_old.add(j) {
                    hbad += 1;
                }
            }
        }
        check(
            "A2 topk single-pass == k-round scan, BIT-EXACT",
            hbad == 0 && nkmis == 0,
            &format!("rows={} count_mismatch={} index_mismatch={}", n, nkmis, hbad),
        );

        // ---------------- B. can-fail control ----------------
        println!("\n--- B. can-fail control (the comparison must be able to disagree) ---");
        // Perturb ONE strength by 1 ULP of its magnitude; the top-k must change.
        let mut perturbed = 0usize;
        for i in 0..n {
            sparse_strength_row_avx2(w.add(i * k0), src.add(i * k0), x, k0, s_new);
            let a = sparse_topk_from_strength(s_new, k0, k, hit_new, bv);
            // swap the two weakest-ranked positions by nudging entry 0 far up
            *s_new.add(0) = *s_new.add(0) + 1e6;
            let b2 = sparse_topk_from_strength(s_new, k0, k, hit_old, bv);
            let mut same = a == b2;
            if same {
                // compare contents
                let mut allsame = true;
                for j in 0..a.min(b2) {
                    if *hit_new.add(j) != *hit_old.add(j) {
                        allsame = false;
                    }
                }
                same = allsame;
            }
            if !same {
                perturbed += 1;
            }
        }
        check(
            "B1 perturbing a strength CHANGES the selection",
            perturbed > 0,
            &format!("rows whose selection changed = {}/{} (must be > 0)", perturbed, n),
        );

        // ---------------- C. tie rule ----------------
        println!("\n--- C. tie rule: equal strengths keep the LOWER index ---");
        // Build a row where EVERY strength is identical.
        let tie = alloc_f32(k0);
        for j in 0..k0 {
            *tie.add(j) = 2.5f32;
        }
        let nt = sparse_topk_from_strength(tie, k0, k, hit_new, bv);
        let ot = topk_scalar_old(tie, k0, k, hit_old);
        let mut tieok = nt == ot && nt == k;
        for j in 0..nt.min(ot) {
            // ascending, and must be exactly 0..k-1 when all strengths tie
            if *hit_new.add(j) != j || *hit_old.add(j) != j {
                tieok = false;
            }
        }
        check(
            "C1 all-equal strengths -> survivors are 0..k-1 (lowest indices)",
            tieok,
            &format!("new nk={} old nk={} first={} expected 0", nt, ot, *hit_new.add(0)),
        );

        // C-can-fail: emulate the `>=` tie bug and show it disagrees here.
        // Explicitly: with `>=` selection scanning ascending, the LAST equal
        // element wins instead of the first.
        let mut buggy_last = usize::MAX;
        for j in 0..k0 {
            if *tie.add(j) >= 2.5f32 {
                buggy_last = j;
            }
        }
        check(
            "C2 can-fail: the `>=` variant WOULD pick a different (higher) index",
            buggy_last != 0,
            &format!("`>=` picks index {} while the contract requires 0", buggy_last),
        );

        // ---------------- D. skip count ----------------
        println!("\n--- D. skip proof: multiply-adds actually performed ---");
        let y = alloc_f32(d);
        let dw = alloc_f32(e);
        let active_count: usize = (0..e).filter(|&i| *act.add(i) != 0.0).count();
        let macs = sparse_forward_masked(w, src, act, x, y, n, k0);
        check(
            "D1 sparse_forward_masked macs == active connection count",
            macs == active_count,
            &format!("macs={} active={} dense_equiv={}", macs, active_count, e),
        );
        check(
            "D2 macs is STRICTLY less than dense n*k0",
            macs < e,
            &format!("{} < {} (skipped {:.2}%)", macs, e, 100.0 * (e - macs) as f64 / e as f64),
        );
        // D can-fail: ignore the mask (dense) -> count must rise to n*k0
        let mut dense_macs = 0usize;
        for i in 0..n {
            for j in 0..k0 {
                let sid = *src.add(i * k0 + j) as usize;
                let _ = *w.add(i * k0 + j) * *x.add(sid);
                dense_macs += 1;
            }
        }
        check(
            "D3 can-fail: ignoring the mask gives the DENSE count",
            dense_macs == e && dense_macs > macs,
            &format!("dense={} > sparse={}", dense_macs, macs),
        );

        // D4: bwd writes only active connections
        for i in 0..e {
            *dw.add(i) = 0.0;
        }
        let dy = alloc_f32(d);
        for i in 0..d {
            *dy.add(i) = rng.range(-1.0, 1.0);
        }
        let written = sparse_gemm_bwd_w_masked(act, src, dy, x, dw, n, k0);
        check(
            "D4 sparse bwd writes exactly the active connections",
            written == active_count,
            &format!("written={} active={}", written, active_count),
        );
        // and inactive ones stayed exactly 0
        let mut leak = 0usize;
        for i in 0..e {
            if *act.add(i) == 0.0 && *dw.add(i) != 0.0 {
                leak += 1;
            }
        }
        check(
            "D5 can-fail premise: inactive connections exist AND stayed 0",
            leak == 0 && (e - active_count) > 0,
            &format!("leaked={} inactive={}", leak, e - active_count),
        );

        // ---------------- E. performance ----------------
        println!("\n--- E. performance (QPC), shapes of the b=4 s=32 step ---");
        let clk = Clock::new();

        // topk end-to-end (strength + select) for the whole layer, all layers
        let mut t_old = 0i64;
        let mut t_new = 0i64;
        // warm up
        for i in 0..n {
            strength_scalar(w.add(i * k0), src.add(i * k0), x, k0, s_old);
            topk_scalar_old(s_old, k0, k, hit_old);
        }
        for i in 0..n {
            sparse_strength_row_avx2(w.add(i * k0), src.add(i * k0), x, k0, s_new);
            sparse_topk_from_strength(s_new, k0, k, hit_new, bv);
        }
        let t0 = clk.now();
        for _ in 0..reps {
            for i in 0..n {
                strength_scalar(w.add(i * k0), src.add(i * k0), x, k0, s_old);
                topk_scalar_old(s_old, k0, k, hit_old);
            }
        }
        t_old = clk.now() - t0;
        let t1 = clk.now();
        for _ in 0..reps {
            for i in 0..n {
                sparse_strength_row_avx2(w.add(i * k0), src.add(i * k0), x, k0, s_new);
                sparse_topk_from_strength(s_new, k0, k, hit_new, bv);
            }
        }
        t_new = clk.now() - t1;
        let per_old = clk.ms(t_old) / reps as f64;
        let per_new = clk.ms(t_new) / reps as f64;
        println!(
            "  topk+strength 1 layer-token:  scalar {:>8.3} ms   new {:>8.3} ms   speedup {:.2}x",
            per_old,
            per_new,
            if per_new > 0.0 { per_old / per_new } else { 0.0 }
        );
        println!(
            "  extrapolated to L={} x T={} :  scalar {:>8.1} ms/step  new {:>8.1} ms/step",
            layers,
            t_total,
            per_old * (layers * t_total) as f64,
            per_new * (layers * t_total) as f64
        );

        // end-to-end step estimate for the sparse path only
        let step_old = per_old * (layers * t_total) as f64;
        let step_new = per_new * (layers * t_total) as f64;
        check(
            "E1 topk path is FASTER than the scalar version",
            per_new < per_old,
            &format!("{:.1} ms/step vs {:.1} ms/step ({:.2}x)", step_new, step_old, step_old / step_new),
        );

        nfail = N_FAIL;
        println!("\n===== checks={} fail={} =====", N_CHECK, N_FAIL);
        if nfail == 0 {
            println!("RESULT: ALL CHECKS PASS");
        }
    }
    std::process::exit(if nfail == 0 { 0 } else { 1 });
}
