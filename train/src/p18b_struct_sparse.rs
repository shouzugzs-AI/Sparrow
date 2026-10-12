// P18-B：结构化稀�?vs 随机稀�?—�?同一形状下的 gather 性能对拍�?//
// ============================ 这一步要回答什�?============================
//
// 白皮�?§13.6 实测结论是「逐连接稀疏在对称形状下反而变慢（0.55x 未打包）」，
// 根因�?gather 开销。但那个结论是在**随机稀�?*下得到的�?//
//     「每算一�?8x8 输出块都要对 A �?W 各做 8 �?gather8，比连续 load �?2~3 倍，
//       gather 开销吃掉了全部收益�?//
// **结构化稀疏（聚类 + 块状打包）是另一回事，白皮书没有测过�?*
// 本探针就是补这个测量，因为它是「QKV/Wo 该不该改稀疏」的唯一判据�?//
// ============================ 三种连接布局 ============================
//
// 形状取模型里的真实投影：[T=128] x [d=1536] x [d=1536]，k0=20 连接/神经元�?//
//   (a) DENSE            稠密 GEMM（基�?1.00x），作为「不稀疏」的对照
//   (b) RANDOM-SPARSE    每神经元**随机**�?k0 个源神经元（白皮书测的那一种）
//   (c) CLUSTER-SPARSE   每神经元�?k0 �?*相似/相邻**的源神经元（结构化）
//   (d) CLUSTER-BLOCK    �?(c) 基础上，把源下标**打包成块**后走 packed 内核
//
// (c) 的构造不是「随机连 20 个」，而是让输出神经元 i 连源区间 [base_i, base_i+k0)�?// 其中 base_i �?i 缓变 —�?这是「相似源聚类」的最小可测模型：同一区域的输�?// 神经元读同一片源，于是列并集**真的收缩**（这正是白皮�?§13.6 里那�?// 「换掩码口径比写更快�?kernel 重要得多」所指的方向）�?//
// ============================ 为什么这个测试可以失�?============================
//
// 判据是「结构化稀疏的加速比 > 1x」。为了让这个判据**能失�?*，探针同时报�?// 随机稀疏的加速比；如果两者都 < 1x，就说明结构化也救不回来，结论是
// 「此形状下连接稀疏不成立」—�?那是一个合法且必须如实报告的结果�?//
// 另：本探�?*�?*用「稀疏省了多�?FLOPs」作为证据。省 FLOPs 不等于省时间
// （本项目�?sparse_dyn 的「稠�?110ms vs 稀�?148ms」坑过）。只�?QPC 时间�?//
// USAGE: vmlm-p18b.exe [--t 128] [--d 1536] [--k0 20] [--reps 20]

extern "C" {
    fn malloc(n: usize) -> *mut u8;
    fn QueryPerformanceCounter(c: *mut i64) -> i32;
    fn QueryPerformanceFrequency(f: *mut i64) -> i32;
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
        println!("OOM n={}", n);
        std::process::exit(9);
    }
    p
}

unsafe fn alloc_u32(n: usize) -> *mut u32 {
    let p = malloc(n * 4) as *mut u32;
    if p.is_null() {
        println!("OOM n={}", n);
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

static mut N_FAIL: u32 = 0;
fn check(name: &str, ok: bool, detail: &str) {
    unsafe {
        if !ok {
            N_FAIL += 1;
        }
    }
    println!("[{}] {}  {}", if ok { "PASS" } else { "FAIL" }, name, detail);
}

fn main() {
    let t = arg_u32("--t", 128) as usize;
    let d = arg_u32("--d", 1536) as usize;
    let k0 = arg_u32("--k0", 20) as usize;
    let reps = arg_u32("--reps", 20) as usize;

    unsafe {
        vmlm_core::interp::init_cpu();
    }
    println!("===== P18-B: structured vs random sparsity, SAME shape =====");
    println!(
        "T={} d={} k0={} reps={}  HAS_AVX2={}",
        t,
        d,
        k0,
        reps,
        unsafe { vmlm_core::interp::HAS_AVX2 }
    );
    println!(
        "dense connections/position = {} ; sparse = {} ({}x fewer)\n",
        d * d,
        d * k0,
        d / k0
    );

    let clk = Clock::new();

    // Everything below walks raw pointers, so the whole measurement body is one
    // unsafe block. (The allocations and kernels are all `unsafe fn`.)
    unsafe {
    // ---- data: A[T,d] and W[d,d] ----
    let a = alloc_f32(t * d);
    let w = alloc_f32(d * d);
    let mut rng = Rng::new(0xBEEF);
    for i in 0..t * d {
        *a.add(i) = rng.uni() * 0.2 - 0.1;
    }
    for i in 0..d * d {
        *w.add(i) = rng.uni() * 0.2 - 0.1;
    }

    // ===================== (a) DENSE baseline =====================
    let c_dense = alloc_f32(t * d);
    let mut t_dense = 0.0f64;
    {
        // warm-up
        vmlm_core::kernels::gemm_nt(a, w, c_dense, t, d, d, false);
        let t0 = clk.now();
        for _ in 0..reps {
            vmlm_core::kernels::gemm_nt(a, w, c_dense, t, d, d, false);
        }
        t_dense = clk.ms(clk.now() - t0) / reps as f64;
    }
    println!("--- (a) DENSE: gemm_nt(A[T,d], W[d,d]) ---");
    println!("    {:.3} ms   = 1.00x baseline\n", t_dense);

    // ===================== column sets =====================
    // For each of comparison we need, for output-neuron-independent sparsity:
    // The architecture says "each output NEURON keeps k0 of its d inputs".
    // That is a per-ROW choice for C = A x W^T: row i of C needs k0 columns of W.
    // We test via the "cols_selected" kernels, which take ONE shared column set.
    //
    // ⚠️ This is the exact distinction the brief asked to measure:
    //   RANDOM  -> per-row choices differ, shared union == all d columns
    //   CLUSTER -> per-row choices overlap, shared union is SMALL
    // So we BUILD the union for each layout and report its size, then time the
    // kernel with that union. The union size IS the structural result.

    let n_sel_rows = d; // one column-set per output neuron (its k0 inputs' sources)
    // (b) RANDOM: each output neuron picks k0 random source columns
    let rand_cols = alloc_u32(d * k0);
    for i in 0..d {
        for j in 0..k0 {
            *rand_cols.add(i * k0 + j) = rng.next_u32() % (d as u32);
        }
    }
    // (c) CLUSTER: output neuron i reads source window starting at a slowly
    // varying base -> neighbouring output neurons share sources.
    let clus_cols = alloc_u32(d * k0);
    for i in 0..d {
        let base = (i * k0) % d;
        for j in 0..k0 {
            *clus_cols.add(i * k0 + j) = ((base + j) % d) as u32;
        }
    }

    // union of a per-row column set (dedup via a bitmap)
    let mut union_size = |cols: *mut u32, rows: usize| -> usize {
        let seen = alloc_f32(d);
        for i in 0..d {
            *seen.add(i) = 0.0;
        }
        for i in 0..rows {
            for j in 0..k0 {
                let c = *cols.add(i * k0 + j) as usize;
                *seen.add(c) = 1.0;
            }
        }
        let mut n = 0usize;
        for i in 0..d {
            if *seen.add(i) != 0.0 {
                n += 1;
            }
        }
        n
    };
    let un_rand = union_size(rand_cols, n_sel_rows);
    let un_clus = union_size(clus_cols, n_sel_rows);

    println!("--- STRUCTURAL RESULT: size of the source-column UNION ---");
    println!(
        "    (b) RANDOM  per-neuron top-{} : union = {}/{} columns  ({:.1}% savable)",
        k0,
        un_rand,
        d,
        100.0 * (d - un_rand) as f64 / d as f64
    );
    println!(
        "    (c) CLUSTER per-neuron top-{} : union = {}/{} columns  ({:.1}% savable)",
        k0,
        un_clus,
        d,
        100.0 * (d - un_clus) as f64 / d as f64
    );
    println!();

    // ===================== (b) RANDOM sparse, unpacked =====================
    // Use the real sparse kernel with the union column set.
    // We build the union list as a u32 array for the kernel.
    let build_union = |cols: *mut u32| -> (*mut u32, usize) {
        let seen = alloc_u32(d);
        for i in 0..d {
            *seen.add(i) = 0;
        }
        for i in 0..n_sel_rows {
            for j in 0..k0 {
                let c = *cols.add(i * k0 + j) as usize;
                *seen.add(c) = 1;
            }
        }
        let out = alloc_u32(d);
        let mut n = 0usize;
        for i in 0..d {
            if *seen.add(i) != 0 {
                *out.add(n) = i as u32;
                n += 1;
            }
        }
        (out, n)
    };

    let c_sp = alloc_f32(t * d);

    // (b) random -> union is (nearly) full, so this measures the gather cost
    //     with no structural saving. Reported for contrast.
    let (u_rand, n_rand) = build_union(rand_cols);
    let mut t_rand = 0.0f64;
    {
        vmlm_core::sparse::gemm_nt_cols_selected(a, w, c_sp, t, d, d, u_rand, n_rand);
        let t0 = clk.now();
        for _ in 0..reps {
            vmlm_core::sparse::gemm_nt_cols_selected(a, w, c_sp, t, d, d, u_rand, n_rand);
        }
        t_rand = clk.ms(clk.now() - t0) / reps as f64;
    }
    println!("--- (b) RANDOM-SPARSE (unpacked cols_selected) ---");
    println!(
        "    {:.3} ms   = {:.2}x vs dense   [union {}/{} -> gather, no saving]",
        t_rand,
        t_dense / t_rand,
        n_rand,
        d
    );

    // (c) cluster -> union is small, so this is the structured case
    let (u_clus, n_clus) = build_union(clus_cols);
    let mut t_clus = 0.0f64;
    {
        vmlm_core::sparse::gemm_nt_cols_selected(a, w, c_sp, t, d, d, u_clus, n_clus);
        let t0 = clk.now();
        for _ in 0..reps {
            vmlm_core::sparse::gemm_nt_cols_selected(a, w, c_sp, t, d, d, u_clus, n_clus);
        }
        t_clus = clk.ms(clk.now() - t0) / reps as f64;
    }
    println!("--- (c) CLUSTER-SPARSE (unpacked cols_selected) ---");
    println!(
        "    {:.3} ms   = {:.2}x vs dense   [union {}/{}]",
        t_clus,
        t_dense / t_clus,
        n_clus,
        d
    );

    // ===================== (d) CLUSTER + PACKED =====================
    // This is the path the whitepaper says fixes the gather overhead
    // (0.55x -> 1.00x at 0% sparsity, 4.04x at 75%).
    //
    // ⚠️ BOTH operands must be packed to the same column set: the kernel
    //    multiplies A_packed[m,kk] by W_packed[n,kk]. The first version of this
    //    probe packed only W and read A with the "packed" kernel, which would
    //    read A columns 0..kk (not the selected ones) -- a wrong-answer bug that
    //    still times the same. Correctness is checked below against dense.
    let apack = alloc_f32(t * n_clus);
    let wpack = alloc_f32(d * n_clus.max(1));
    let mut t_clus_p = 0.0f64;
    {
        vmlm_core::sparse::pack_cols(a, apack, t, d, u_clus, n_clus);
        vmlm_core::sparse::pack_cols(w, wpack, d, d, u_clus, n_clus);
        vmlm_core::sparse::gemm_nt_cols_packed(apack, wpack, c_sp, t, d, n_clus);
        let t0 = clk.now();
        for _ in 0..reps {
            vmlm_core::sparse::gemm_nt_cols_packed(apack, wpack, c_sp, t, d, n_clus);
        }
        t_clus_p = clk.ms(clk.now() - t0) / reps as f64;
    }
    println!("--- (d) CLUSTER-SPARSE + PACKED (pack_cols + cols_packed) ---");
    println!(
        "    {:.3} ms   = {:.2}x vs dense   [union {}/{}]",
        t_clus_p,
        t_dense / t_clus_p,
        n_clus,
        d
    );

    // ---- correctness of the sparse paths (a speed claim on wrong numbers is worthless)
    // The cluster layout keeps only n_clus of the d source columns, so the
    // sparse result must equal a DENSE result computed on the SAME reduced
    // column set. That is the honest reference: it is not "sparse == dense".
    println!("\n--- correctness: sparse result vs dense-on-the-same-columns ---");
    let cols_f = alloc_f32(d * d);
    for i in 0..d * d {
        *cols_f.add(i) = *w.add(i);
    }
    // Build the reduced dense reference: C_ref[t,j] = sum_{p in cols} A[t,p] * W[j,p]
    let c_ref = alloc_f32(t * d);
    for i in 0..t * d {
        *c_ref.add(i) = 0.0;
    }
    for ti in 0..t {
        for j in 0..d {
            let mut acc = 0.0f32;
            for q in 0..n_clus {
                let p = *u_clus.add(q) as usize;
                acc += *a.add(ti * d + p) * *w.add(j * d + p);
            }
            *c_ref.add(ti * d + j) = acc;
        }
    }
    let mut bad = 0usize;
    let mut maxd = 0.0f32;
    for i in 0..t * d {
        let x = *c_sp.add(i);
        let y = *c_ref.add(i);
        let dd = (x - y).abs();
        if dd > maxd {
            maxd = dd;
        }
        if dd > 1e-4 {
            bad += 1;
        }
    }
    check(
        "sparse+packed result matches dense-on-same-columns",
        bad == 0,
        &format!("bad={}/{} max|diff|={:.3e}", bad, t * d, maxd),
    );

    // ===================== (e) SHARED-POOL sparse =====================
    //
    // THE FIRST RUN OF THIS PROBE PROVED MY OWN "CLUSTER" WAS NOT STRUCTURED:
    //   (b) random  per-neuron top-20 : union = 1536/1536  (0.0% savable)
    //   (c) cluster per-neuron top-20 : union = 1536/1536  (0.0% savable)
    // Clustering each row did NOT shrink the union, because the union is taken
    // over ALL 1536 output neurons: as long as every source is used by SOME
    // neuron, the union is the full set. That is the §13.6 geometric fact, and
    // my first construction fell into it.
    //
    // The ONLY way the union shrinks is to restrict every neuron to a SHARED,
    // SMALLER source pool. So that is what (e) measures: all neurons draw their
    // k0 connections from a shared pool of `pool` sources. This is the honest
    // form of "structured sparsity" for this architecture, and it is the
    // configuration that would justify making QKV/Wo sparse.
    //
    // We sweep the pool size to find where sparse overtakes dense.
    println!("\n--- (e) SHARED-POOL sparse: all neurons read from a pool of P sources ---");
    println!("    {:>6}  {:>7}  {:>10}  {:>10}  {:>10}  {:>8}", "P", "union", "unpacked", "packed", "best", "vs dense");
    let mut best_overall = 0.0f64;
    let mut best_pool = 0usize;
    let mut pool_results: [(usize, f64, f64); 6] = [(0, 0.0, 0.0); 6];
    let pools = [1536usize, 1024, 768, 512, 384, 256];
    for (pi, &pool) in pools.iter().enumerate() {
        // each output neuron picks k0 DISTINCT sources from [0, pool)
        let cols = alloc_u32(d * k0);
        for i in 0..d {
            // deterministic stride so the pool is evenly covered (worst case
            // for the union, i.e. the union will BE the pool)
            let base = (i * 7) % pool;
            for j in 0..k0 {
                *cols.add(i * k0 + j) = ((base + j) % pool) as u32;
            }
        }
        let un = union_size(cols, n_sel_rows);
        let (uc, nc) = build_union(cols);

        // unpacked
        vmlm_core::sparse::gemm_nt_cols_selected(a, w, c_sp, t, d, d, uc, nc);
        let t0 = clk.now();
        for _ in 0..reps {
            vmlm_core::sparse::gemm_nt_cols_selected(a, w, c_sp, t, d, d, uc, nc);
        }
        let tp_un = clk.ms(clk.now() - t0) / reps as f64;

        // packed (both operands restricted to the shared pool)
        let ap = alloc_f32(t * nc.max(1));
        let wp = alloc_f32(d * nc.max(1));
        vmlm_core::sparse::pack_cols(a, ap, t, d, uc, nc);
        vmlm_core::sparse::pack_cols(w, wp, d, d, uc, nc);
        vmlm_core::sparse::gemm_nt_cols_packed(ap, wp, c_sp, t, d, nc);
        let t1 = clk.now();
        for _ in 0..reps {
            vmlm_core::sparse::gemm_nt_cols_packed(ap, wp, c_sp, t, d, nc);
        }
        let tp_p = clk.ms(clk.now() - t1) / reps as f64;

        let best = if tp_un < tp_p { tp_un } else { tp_p };
        let sp = t_dense / best;
        pool_results[pi] = (pool, sp, tp_p);
        if sp > best_overall {
            best_overall = sp;
            best_pool = pool;
        }
        println!(
            "    {:>6}  {:>7}  {:>9.3}ms  {:>9.3}ms  {:>9.3}ms  {:>7.2}x",
            pool, un, tp_un, tp_p, best, sp
        );
    }
    println!(
        "\n    best: pool={} -> {:.2}x vs dense",
        best_pool, best_overall
    );
    // ===================== THE CRITERION =====================
    println!("\n===== VERDICT =====");
    let sp_rand = t_dense / t_rand;
    let sp_clus = t_dense / t_clus;
    let sp_clus_p = t_dense / t_clus_p;
    println!("  random  unpacked : {:.2}x", sp_rand);
    println!("  cluster unpacked : {:.2}x", sp_clus);
    println!("  cluster packed   : {:.2}x", sp_clus_p);

    check(
        "structured sparsity beats RANDOM sparsity (the actual hypothesis)",
        sp_clus_p > sp_rand,
        &format!("cluster+packed {:.2}x vs random {:.2}x", sp_clus_p, sp_rand),
    );
    check(
        "structured sparsity is faster than DENSE (>1x => architecture stands)",
        sp_clus_p > 1.0,
        &format!(
            "{:.2}x ({:.3} ms vs {:.3} ms)",
            sp_clus_p, t_clus_p, t_dense
        ),
    );
    // The decisive criterion for "should QKV/Wo be sparse": a SHARED-POOL sparse
    // projection must beat dense GEMM by a real margin. The full-union case
    // (b)/(c) cannot, and that is the honest reference point.
    check(
        "SHARED-POOL sparse beats dense (the real 'should we go sparse' test)",
        best_overall > 1.0,
        &format!("best {:.2}x at pool={} (vs full-union packed {:.2}x)", best_overall, best_pool, sp_clus_p),
    );
    } // end unsafe

    let nfail = unsafe { N_FAIL };
    println!("\n  fail={}", nfail);
    std::process::exit(if nfail == 0 { 0 } else { 1 });
}
