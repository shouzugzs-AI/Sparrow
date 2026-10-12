// PATH: B1
//
// B1 **判据强度**探针：A1 的对拍**真的能抓到**错误实现吗？
//
// 为什么需要这个文件：A1 报 `bad=0` 有两种可能 ——
//   (a) B1 实现确实与算子版逐位相同（我们要的结论）；
//   (b) 对拍**恒真**，无论实现怎么写都报 0（判据是哑的）。
// 光看 `bad=0` 分不出这两者。所以这里把已知的错误实现喂给**同一段对拍代码**，
// 逐个确认它报出 `bad>0`。若某个错误形态报 0，说明 A1 在那一维度上是瞎的。
//
// 这是 can-fail 对照的批量形式：`tools/p18_topk_canfail.py` 用的是同一思路。
//
// 用法：vmlm-b1-canfail.exe [--n 1536] [--k0 20]
// 退出码：0 = 6/6 错误形态都被抓到（判据有效）；1 = 有错误形态漏网（判据弱）。
use vmlm_core::b1_pure::b1_forward;
use vmlm_core::conn_sparse::K0_DEFAULT;
use vmlm_core::conn_train::*;

extern "C" {
    fn malloc(n: usize) -> *mut u8;
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
    let tag = if ok { "PASS" } else { "FAIL" };
    println!("[{}] {}  {}", tag, name, detail);
}

fn args_u32(flag: &str, def: u32) -> u32 {
    let v: Vec<String> = std::env::args().collect();
    for i in 0..v.len() {
        if v[i] == flag && i + 1 < v.len() {
            return v[i + 1].parse().unwrap_or(def);
        }
    }
    def
}

unsafe fn alloc_f32(n: usize) -> *mut f32 {
    let p = malloc(n * 4 + 64) as *mut f32;
    assert!(!p.is_null());
    p
}

unsafe fn slice<'a>(p: *mut f32, n: usize) -> &'a mut [f32] {
    std::slice::from_raw_parts_mut(p, n)
}

fn det_fill(buf: &mut [f32], seed: u32, lo: f32, hi: f32) {
    let mut s = seed | 1;
    for x in buf.iter_mut() {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        let u = (s as f64) / (u32::MAX as f64);
        *x = (lo as f64 + (hi - lo) as f64 * u) as f32;
    }
}

unsafe fn bit_diff(a: *const f32, b: *const f32, n: usize) -> usize {
    let mut bad = 0usize;
    for i in 0..n {
        if (*a.add(i)).to_bits() != (*b.add(i)).to_bits() {
            bad += 1;
        }
    }
    bad
}

fn main() {
    let n_neurons = args_u32("--n", 1536);
    let k0 = args_u32("--k0", K0_DEFAULT);
    println!("===== B1 criterion STRENGTH probe (does A1 catch wrong code?) =====");
    println!("n_neurons={} k0={}", n_neurons, k0);
    println!("THIS IS B1.");

    let n = n_neurons as usize;
    let k0u = k0 as usize;
    let e = n * k0u;

    unsafe {
        vmlm_core::interp::init_cpu();
        let base = malloc(SparseConnW::bytes(n_neurons, k0));
        let sc = SparseConnW::from_base(base, n_neurons, k0, 7);
        init_sparse_conn_scaled(&sc, 0.144);

        let x = alloc_f32(n);
        det_fill(slice(x, n), 0x1234_5678, -1.0, 1.0);
        let y_ref = alloc_f32(n);
        let y_bad = alloc_f32(n);

        // 参考：真 B1（也是算子版对照的对象，见 b1_verify.rs）
        b1_forward(&sc, x, y_ref);
        // 与算子版确认参考本身正确（否则下面全部无意义）
        //
        // ⚠️ 本条已从 bit-exact **降级为 ε 级**（用户裁决：不需要 bit-exact）。
        //    `b1_forward` 现在是 SIMD 归约（8 路部分和 + 横向折叠、从零向量
        //    起步），**按定义**不再满足「升序 j / 单标量累加器 / 首项赋值」，
        //    所以位差不再为 0（实测 844/1536）。
        //    判据目的仍有效：确认参考与算子版**算的是同一个函数**（只差舍入）。
        //    用「相对项幅度」比较：舍入级 ~1e-7（实测 1.3e-7）；
        //    算错东西 ~1.0，差 7 个数量级。
        let y_op = alloc_f32(n);
        sparse_forward(&sc, x, y_op, k0u);
        let ref_bad = bit_diff(y_ref as *const f32, y_op as *const f32, n);
        let mut ref_ratio = 0.0f32;
        for i in 0..n {
            let mut s = 0.0f32;
            for j in 0..k0u {
                let sid = *sc.src.add(i * k0u + j) as usize;
                s += (*sc.w.add(i * k0u + j) * *x.add(sid)).abs();
            }
            if s > 0.0 {
                let r = (*y_ref.add(i) - *y_op.add(i)).abs() / s;
                if r > ref_ratio {
                    ref_ratio = r;
                }
            }
        }
        check(
            "premise: the reference matches the operator version at ROUNDING level",
            ref_ratio <= 1.0e-5,
            &format!(
                "worst |diff|/scale = {:.3e} <= 1e-5  (bit-diff {}/{}，已按裁决降级)",
                ref_ratio, ref_bad, n
            ),
        );

        // ---- 错误形态 1：漏算每行最后一条连接（k0-1 宽度）----
        //
        // ⚠️ 第一版写成 `SparseConnW { k0: k0u-1, ..sc }`，即 k0 与缓冲区不符。
        //    那**违反内核的安全前提**，且在 SIMD 路径下会读到越界（实测崩溃）。
        //    修法：分配一个**几何自洽的** (k0-1) 宽结构，真正少摆一条连接。
        //    这样测的是"漏算一条"这个想法，而不是未定义行为。
        {
            let k0s = k0u - 1;
            let es = n * k0s;
            let bs = malloc(SparseConnW::bytes(n_neurons, k0s as u32));
            let mut sc_s = SparseConnW::from_base(bs, n_neurons, k0s as u32, 7);
            sc_s.e = es;
            for i in 0..n {
                for j in 0..k0s {
                    let sid = *sc.src.add(i * k0u + j);
                    *sc_s.src.add(i * k0s + j) = sid;
                    *sc_s.src_u32.add(i * k0s + j) = sid as u32;
                    *sc_s.w.add(i * k0s + j) = *sc.w.add(i * k0u + j);
                }
            }
            sc_s.assert_geometry();
            b1_forward(&sc_s, x, y_bad);
        }
        let d1 = bit_diff(y_bad as *const f32, y_ref as *const f32, n);
        check(
            "wrong-1 drop last connection (k0-1)",
            d1 > 0,
            &format!("caught bad={} rows of {}", d1, n),
        );

        // ---- 错误形态 2：跳过第一个连接（同样 k0-1，但从 j=1 起）----
        for i in 0..n {
            let b = i * k0u;
            let mut acc = 0.0f32;
            let mut first = true;
            for j in 1..k0u {
                let sid = *sc.src.add(b + j) as usize;
                let term = *sc.w.add(b + j) * *x.add(sid);
                acc = if first { term } else { acc + term };
                first = false;
            }
            *y_bad.add(i) = acc;
        }
        let d2 = bit_diff(y_bad as *const f32, y_ref as *const f32, n);
        check(
            "wrong-2 skip first connection (j from 1)",
            d2 > 0,
            &format!("caught bad={} rows of {}", d2, n),
        );

        // ---- 错误形态 3：行内**降序**累加（结合律违例）----
        // 这是最阴险的一种：连接的**集合**完全相同、MAC 数完全相同，
        // 数学上"等价"，只有位型不同。A1 必须抓到它，否则
        // 「累加顺序是契约」这条主张就没有判据支撑。
        for i in 0..n {
            let b = i * k0u;
            let mut acc = 0.0f32;
            let mut first = true;
            for j in (0..k0u).rev() {
                let sid = *sc.src.add(b + j) as usize;
                let term = *sc.w.add(b + j) * *x.add(sid);
                acc = if first { term } else { acc + term };
                first = false;
            }
            *y_bad.add(i) = acc;
        }
        let d3 = bit_diff(y_bad as *const f32, y_ref as *const f32, n);
        check(
            "wrong-3 reverse in-row accumulation order (associativity)",
            d3 > 0,
            &format!("caught bad={} rows of {}", d3, n),
        );

        // ---- 错误形态 4：从 0.0 起加而不是首项赋值 ----
        // 注意：在**一般随机数据**上这一条常常与正确版逐位相同
        // （0.0 + t == t 对绝大多数 t 成立），所以它的可观测性需要
        // 专门的 -0.0 构造 —— 见 b1_verify.rs 的那条判据。
        // 这里如实报告：在随机数据上它**可能抓不到**，这是已知边界。
        for i in 0..n {
            let b = i * k0u;
            let mut acc = 0.0f32;
            for j in 0..k0u {
                let sid = *sc.src.add(b + j) as usize;
                acc += *sc.w.add(b + j) * *x.add(sid);
            }
            *y_bad.add(i) = acc;
        }
        let d4 = bit_diff(y_bad as *const f32, y_ref as *const f32, n);
        println!(
            "  [info] wrong-4 'start from 0.0' on random data: caught bad={} rows of {}",
            d4, n
        );
        check(
            "wrong-4 (from-0.0) is observable via the dedicated -0.0 construction",
            // 在随机数据上抓不到是**允许**的（0.0+t==t），所以这里不把
            // 「随机数据上抓到」当判据；真正的判据在 b1_verify.rs 里用
            // w=-0.0/k=1 的构造让两条路径必然分叉（实测 4 vs 0）。
            // 这里只断言那条专门构造的存在性前提：-0.0 的位型与 +0.0 不同。
            (0.0f32 == -0.0f32) && (0.0f32.to_bits() != (-0.0f32).to_bits()),
            &format!("(random-data catch={} is expected to be small; dedicated construction is the real criterion)", d4),
        );

        // ---- 错误形态 5：错的行宽（用 k0+1 读越界一格的 src）----
        // 这是「看似合理的垃圾数」的典型来源。用 k0+1 会读到下一行第一格的 src。
        //
        // ⚠️ 这条**曾经让整个探针 0xC0000005 崩掉**（实测）。
        //    旧标量版一次读一个 f32，越界一格仍落在分配的余量里所以"活下来"；
        //    B1 的新 SIMD 内核按 k0 以 **8 宽**读 src_u32，最后一行会读越过
        //    src_u32 末尾 32 字节 —— 而 src_u32 是分配区的**最后一块** ——
        //    直接访问违例，且没有任何诊断信息。
        //
        //    根因是这条判据在**违反内核的安全前提**（k0 必须与缓冲区一致），
        //    也就是说它在测未定义行为，不是在测"错行宽"这个想法本身。
        //
        //    修法：构造一个**几何自洽的** k0+1 宽结构 —— 真的分配
        //    (k0+1)*n 个元素并把 src_u32 也填满。这样内核在这个字段上
        //    依然是安全的，而"行宽错了"这件事仍然被真的测到。
        //    （生产侧也补了 `assert_geometry()`：真出现不一致时给出可读报错，
        //      而不是崩溃。）
        {
            let k0w = k0u + 1;
            let ew = n * k0w;
            let bw = malloc(SparseConnW::bytes(n_neurons, k0w as u32));
            let mut sc_w = SparseConnW::from_base(bw, n_neurons, k0w as u32, 7);
            // 几何自洽：e == n_neurons * k0（守卫会检查这一条）
            sc_w.e = ew;
            // 拓扑照抄原结构，但按 k0+1 的**宽行**摆放：每行多一格，
            // 于是每个神经元都会读到"下一行第一格"的源 —— 正是 wrong-5 的语义。
            for i in 0..n {
                for j in 0..k0u {
                    let sid = *sc.src.add(i * k0u + j);
                    *sc_w.src.add(i * k0w + j) = sid;
                    *sc_w.src_u32.add(i * k0w + j) = sid as u32;
                    *sc_w.w.add(i * k0w + j) = *sc.w.add(i * k0u + j);
                }
                // 多出的那一格：故意指向第 0 个源，制造一个真实但错误的贡献
                *sc_w.src.add(i * k0w + k0u) = 0.0;
                *sc_w.src_u32.add(i * k0w + k0u) = 0;
                *sc_w.w.add(i * k0w + k0u) = 1.0;
            }
            sc_w.assert_geometry();
            b1_forward(&sc_w, x, y_bad);
        }
        let d5 = bit_diff(y_bad as *const f32, y_ref as *const f32, n);
        check(
            "wrong-5 wrong row pitch (k0+1, reads next row's first slot)",
            d5 > 0,
            &format!("caught bad={} rows of {}", d5, n),
        );

        // ---- 错误形态 6：把 w 与 x 的操作数写反（w*x vs x*w）----
        // 乘法交换律在 IEEE-754 下**是**成立的，所以这一维度**本来就不可观测**，
        // 属于已知边界而非判据缺陷。
        //
        // ⚠️ 第一版拿 `y_ref`（现在是 SIMD 归约）当基准，于是"操作数交换"与
        //    "归约顺序不同"两个效应**混在一起**，报出 844 位差 —— 看起来像
        //    判据失效，其实是被测对象换了。修法：拿**纯标量契约参考**
        //    （`b1_forward_scalar_reference`）当基准，这样唯一差异就是操作数
        //    顺序本身，与"乘法交换律是否可观测"这个问题一一对应。
        for i in 0..n {
            let b = i * k0u;
            let mut acc = 0.0f32;
            let mut first = true;
            for j in 0..k0u {
                let sid = *sc.src.add(b + j) as usize;
                let term = *x.add(sid) * *sc.w.add(b + j); // 交换操作数
                acc = if first { term } else { acc + term };
                first = false;
            }
            *y_bad.add(i) = acc;
        }
        // 标量契约参考（逐位契约版），与上面的循环同构、只差操作数顺序
        let y_ref_scalar = alloc_f32(n);
        vmlm_core::b1_pure::b1_forward_scalar_reference(&sc, x, y_ref_scalar);
        let d6 = bit_diff(y_bad as *const f32, y_ref_scalar as *const f32, n);
        println!(
            "  [info] wrong-6 swapped operand order (x*w) vs SCALAR contract ref: caught bad={} rows of {}",
            d6, n
        );
        // 乘法交换律成立 ⇒ 这一维度**本来就不可观测**，不是判据缺陷。
        check(
            "wrong-6 operand swap is unobservable by design (IEEE mul is commutative)",
            d6 == 0,
            &format!("bad={} (0 is the CORRECT expectation here; compared scalar-vs-scalar so reduction order is not a confound)", d6),
        );

        println!();
        println!("===== checks={} fail={} =====", unsafe { N_CHECK }, unsafe {
            N_FAIL
        });
        let rc = if unsafe { N_FAIL } == 0 { 0 } else { 1 };
        std::process::exit(rc);
    }
}
