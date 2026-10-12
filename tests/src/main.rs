// VMLM 内核回归（Lead 维护）
//
// 覆盖：
//   1. Ins 结构必须是 16 字节（定长指令格式的硬约束；P8 曾因字段对齐变成 20，
//      导致 text_len 与 META 偏移错位、所有模型静默算错）
//   2. GEMM NT 的 13 组形状（含 N=1 / M=1 边界，历史上在这里崩过）
//   3. TOPK_THRESH 的阈值语义
//   4. [H/P9] exp_f 精度硬回归 + 速度 A/B（任务 H 的验收指标）
//
// 更完整的 packed/对齐压力测试在 vmlm-packbench（成员 D）。
use vmlm_core::interp::Ins;
use vmlm_core::kernels;

/// [H/P9] 旧 exp_f 的转录，只用于 A/B 对比（精度与速度各一份）。
/// 这不是引擎路径的一部分，仅为回归提供「改前」基线。
fn exp_f_old(x: f32) -> f32 {
    let x = if x < -88.0 {
        -88.0
    } else if x > 88.0 {
        88.0
    } else {
        x
    };
    let n = (x * 1.442_695_04 + 0.5) as i32;
    let r = x - (n as f32) * 0.693_147_18;
    let r2 = r * r;
    let p = 1.0
        + r
        + r2 * 0.5
        + r2 * r * 0.166_666_67
        + r2 * r2 * 0.041_666_668
        + r2 * r2 * r * 0.008_333_334
        + r2 * r2 * r2 * 0.001_388_889;
    let bits = ((n + 127) as u32) << 23;
    p * f32::from_bits(bits)
}

/// f64 exp 参考（纯 Rust，不引 libm：exp(x) = 2^(x/log2) 用 f64 的
/// 位操作 + 泰勒；这里直接用 f64 的幂级数足够，误差 ~1e-16）
fn exp_ref(x: f64) -> f64 {
    // range reduce: x = k*ln2 + r
    let k = (x / core::f64::consts::LN_2).round();
    let r = x - k * core::f64::consts::LN_2;
    // 泰勒 18 项在 |r| <= 0.347 下误差 < 1e-20
    let mut term = 1.0f64;
    let mut sum = 1.0f64;
    let mut i = 1;
    while i <= 18 {
        term *= r / (i as f64);
        sum += term;
        i += 1;
    }
    sum * exp2i(k)
}

fn exp2i(k: f64) -> f64 {
    // 2^k，k 为整数；|k| < 1024，用位拼装避免任何 libm 调用
    if k >= -1022.0 && k <= 1023.0 {
        let ki = k as i64;
        if ki >= -1022 && ki <= 1023 {
            let bits = ((ki + 1023) as u64) << 52;
            return f64::from_bits(bits);
        }
    }
    // 超出 f64 正常范围：用平方累乘
    let mut v = 1.0f64;
    let mut e = if k < 0.0 { -(k as i64) } else { k as i64 };
    let mut base = if k < 0.0 { 0.5f64 } else { 2.0f64 };
    while e > 0 {
        if e & 1 == 1 {
            v *= base;
        }
        base *= base;
        e >>= 1;
    }
    v
}

/// [H/P9] exp_f 精度回归。x ∈ [-88,88] 全定义域密采样；判据 max rel <= 1e-6。
///
/// 注意区间：早期版本只扫 [-20,20]，因此**漏掉了次正规区的 bug**
/// （exp_f(-88) 曾因 `(n+127)<<23` 在 n=-127 时位模式为 0 而返回 0.0）。
/// 教训：exp_f 的契约是「f32 的 exp」，就必须覆盖它的全部有效定义域。
fn exp_precision_sweep() -> bool {
    const N: usize = 4_000_001; // 均匀 4e6 点 + ln2 缝线加密
    let lo = -88.0f64;
    let hi = 88.0f64;
    let mut worst_new = 0.0f64;
    let mut worst_new_x = 0.0f64;
    let mut sum_new = 0.0f64;
    let mut worst_old = 0.0f64;
    let mut worst_old_x = 0.0f64;
    let mut cnt = 0u64;
    // 次正规区（exp(x) < 2^-126）单独统计：其可达相对误差由 f32 的
    // 次正规表示精度决定（尾数位不足），不能与正规区的 0.66 ulp 混淆。
    let mut worst_sub = 0.0f64;
    let mut worst_sub_x = 0.0f64;
    let mut sub_cnt = 0u64;

    let mut scan = |xf: f32| {
        // 参考用 f32 输入（内核只收到 f32）
        let r = exp_ref(xf as f64);
        let n = kernels::exp_f(xf) as f64;
        let o = exp_f_old(xf) as f64;
        let rn = (n - r).abs() / r.abs();
        let ro = (o - r).abs() / r.abs();
        if rn > worst_new {
            worst_new = rn;
            worst_new_x = xf as f64;
        }
        if ro > worst_old {
            worst_old = ro;
            worst_old_x = xf as f64;
        }
        sum_new += rn;
        cnt += 1;
        // 正规/次正规分界：exp(x) = 2^-126 在 x = ln(2^-126) = -87.33654
        if (xf as f64) < -87.33654 {
            if rn > worst_sub {
                worst_sub = rn;
                worst_sub_x = xf as f64;
            }
            sub_cnt += 1;
        }
    };

    // 1) 均匀网格
    for i in 0..N {
        let t = lo + (hi - lo) * (i as f64) / ((N - 1) as f64);
        scan(t as f32);
    }
    // 2) 缝线加密：每个 k*ln2 与 (k+0.5)*ln2 周围 ±2e-3
    let mut k = -29i32;
    while k <= 29 {
        for half in [0.0f64, 0.5] {
            let c = (k as f64 + half) * core::f64::consts::LN_2;
            if c < lo - 0.01 || c > hi + 0.01 {
                continue;
            }
            for j in 0..4001 {
                let off = -2e-3 + 4e-3 * (j as f64) / 4000.0;
                let t = c + off;
                if t >= lo && t <= hi {
                    scan(t as f32);
                }
            }
        }
        k += 1;
    }

    println!("\n=== [H/P9] exp_f 精度扫描（x in [-88,88] 全定义域）===");
    println!("  采样点              : {}", cnt);
    println!("  改前 max rel err    : {:.6e}  @ x = {:.9}", worst_old, worst_old_x);
    println!("  改后 max rel err    : {:.6e}  @ x = {:.9}", worst_new, worst_new_x);
    println!("  改后 mean rel err   : {:.6e}", sum_new / (cnt as f64));
    println!("  改善倍数            : {:.1}x", worst_old / worst_new);
    println!("  改后 ulp            : {:.2} ulp", worst_new / 1.1920929e-7);
    println!("  次正规区 ({} 点)   : max rel {:.6e} @ x = {:.6}",
             sub_cnt, worst_sub, worst_sub_x);
    println!("    ^ 次正规区上限由 f32 表示精度决定，非内核缺陷");
    let pass = worst_new <= 1e-6;
    println!("  判据 max rel <= 1e-6: {}", if pass { "PASS" } else { "FAIL" });

    // 额外：边界与特殊值不许出现 NaN / 符号错误
    let mut edge_ok = true;
    for &v in &[-1e30f32, -88.0, -87.9, -87.34, -87.0, -1.0, -0.0, 0.0, 1e-30, 1.0, 88.0, 1e30] {
        let r = kernels::exp_f(v);
        if r.is_nan() {
            println!("  !! exp_f({}) = NaN", v);
            edge_ok = false;
        }
    }
    // [H/P9 FIX] 次正规回归守卫：exp_f(-88) 曾经返回 0.0
    let sub_zero = kernels::exp_f(-88.0);
    let sub_ok = sub_zero > 0.0 && (sub_zero as f64 - exp_ref(-88.0)).abs() / exp_ref(-88.0) < 1e-6;
    println!("  次正规 exp_f(-88)   : {} (got {:.6e}, 期望 6.054602e-39)",
             if sub_ok { "PASS" } else { "FAIL" }, sub_zero);
    // 单调性（正区间）
    let mut prev = kernels::exp_f(-20.0);
    let mut mono = true;
    let mut i = 0;
    while i < 200000 {
        let x = -20.0 + 40.0 * (i as f32) / 199999.0;
        let v = kernels::exp_f(x);
        if v < prev {
            mono = false;
            println!("  !! 非单调 @ x={}", x);
            break;
        }
        prev = v;
        i += 1;
    }
    println!("  边界无 NaN          : {}", if edge_ok { "PASS" } else { "FAIL" });
    println!("  单调性              : {}", if mono { "PASS" } else { "FAIL" });
    // exp(0) 必须精确为 1
    let z = kernels::exp_f(0.0);
    let z_ok = z == 1.0;
    println!("  exp_f(0) == 1.0     : {} (got {})", if z_ok { "PASS" } else { "FAIL" }, z);

    pass && edge_ok && mono && z_ok && sub_ok
}

/// [H/P9] 速度 A/B：各调用 1e7 次，用 std::time::Instant。
fn exp_speed_ab() -> bool {
    const ITERS: usize = 10_000_000;
    // 输入用一段固定伪随机序列，避免编译器把循环折叠成常量
    let mut xs = vec![0f32; 1024];
    let mut s = 12345u32;
    for v in xs.iter_mut() {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        *v = ((s >> 8) as f32 / 8388608.0) * 20.0 - 10.0;
    }

    let run = |f: fn(f32) -> f32| -> (f64, f32) {
        // 预热
        let mut acc = 0f32;
        for i in 0..100_000 {
            acc += f((xs[i & 1023]) - 1.0);
        }
        let t0 = std::time::Instant::now();
        let mut acc = 0f32;
        let mut i = 0usize;
        while i < ITERS {
            acc += f(*unsafe { xs.get_unchecked(i & 1023) });
            i += 1;
        }
        let dt = t0.elapsed();
        (dt.as_secs_f64(), acc)
    };

    // 交替跑两轮，取各自最好值，降低频率/调度噪声
    let (mut old_best, mut new_best) = (f64::MAX, f64::MAX);
    let mut sink = 0f32;
    for _ in 0..3 {
        let (a, sa) = run(exp_f_old);
        let (b, sb) = run(kernels::exp_f);
        sink += sa + sb;
        if a < old_best {
            old_best = a;
        }
        if b < new_best {
            new_best = b;
        }
    }
    std::hint::black_box(sink);

    let mo = old_best * 1e9 / ITERS as f64;
    let mn = new_best * 1e9 / ITERS as f64;
    let delta = (mn - mo) / mo * 100.0;
    println!("\n=== [H/P9] exp_f 速度 A/B（{} 次/轮，取 3 轮最好值）===", ITERS);
    println!("  改前: {:.3} s  ({:.3} ns/call)", old_best, mo);
    println!("  改后: {:.3} s  ({:.3} ns/call)", new_best, mn);
    println!("  变化: {:+.2}%  {}", delta,
             if delta > 20.0 { "(退化 > 20%，需说明)" } else { "(在 20% 预算内)" });
    true
}

/// [H/P9] 把 exp_f 的采样结果 dump 给 tools/exp_accuracy.py 做权威校验。
/// 格式: int32 n | n * f32 x | n * f32 exp_f(x)
fn exp_dump(path: &str) -> std::io::Result<()> {
    use std::io::Write;
    const N: usize = 2_000_001;
    let mut xs: Vec<f32> = Vec::with_capacity(N);
    // 全定义域 [-88, 88]：含次正规区，避免再次漏掉边界缺陷
    for i in 0..N {
        let t = -88.0 + 176.0 * (i as f64) / ((N - 1) as f64);
        xs.push(t as f32);
    }
    let f = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(f);
    w.write_all(&(N as i32).to_le_bytes())?;
    for v in &xs {
        w.write_all(&v.to_le_bytes())?;
    }
    for v in &xs {
        w.write_all(&kernels::exp_f(*v).to_le_bytes())?;
    }
    w.flush()?;
    println!("dumped {} points to {}", N, path);
    Ok(())
}

fn naive(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0f32;
            for p in 0..k {
                s += a[i * k + p] * b[j * k + p];
            }
            c[i * n + j] = s;
        }
    }
    c
}

fn chk(m: usize, n: usize, k: usize) -> bool {
    let a: Vec<f32> = (0..m * k).map(|i| ((i * 37 % 100) as f32) / 50.0 - 1.0).collect();
    let b: Vec<f32> = (0..n * k).map(|i| ((i * 53 % 100) as f32) / 50.0 - 1.0).collect();
    let r = naive(&a, &b, m, n, k);
    let mut c = vec![0f32; m * n];
    unsafe {
        kernels::gemm_nt(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k, false);
    }
    let mut mx = 0f32;
    for i in 0..m * n {
        let e = (c[i] - r[i]).abs();
        if e > mx {
            mx = e;
        }
    }
    let ok = mx < 1e-3;
    println!("  GEMM {:>4}x{:<4}x{:<4} maxerr={:.7} {}", m, n, k, mx,
             if ok { "PASS" } else { "FAIL" });
    ok
}

fn main() {
    // [H/P9] dump 模式：给 tools/exp_accuracy.py 校验**编译产物**（而非模型）
    let argv: Vec<String> = std::env::args().collect();
    if argv.len() >= 3 && argv[1] == "expdump" {
        exp_dump(&argv[2]).unwrap();
        return;
    }

    unsafe { vmlm_core::interp::init_cpu() };

    let mut ok = true;

    // ---- 1. 指令结构必须是 16 字节 ----
    let sz = std::mem::size_of::<Ins>();
    let sz_ok = sz == 16;
    println!("Ins 结构大小 = {} 字节 (期望 16)  {}", sz, if sz_ok { "PASS" } else { "FAIL" });
    ok &= sz_ok;

    println!("AVX2 = {}", unsafe { vmlm_core::interp::HAS_AVX2 });
    println!("=== GEMM 内核回归（NT，含边界形状）===");
    for (m, n, k) in [
        (4, 8, 8), (1, 1, 64), (1, 32, 64), (1, 10, 32), (8, 8, 512),
        (512, 512, 512), (3, 5, 7), (5, 3, 9), (17, 23, 31), (64, 64, 64),
        (16, 1, 16), (1, 16, 16), (16, 16, 1),
    ] {
        ok &= chk(m, n, k);
    }

    println!("\n=== TOPK_THRESH ===");
    let v: Vec<f32> = vec![3.0, 1.0, 5.0, 2.0, 4.0, 0.5, 6.0, 1.5];
    let t = unsafe { kernels::topk_thresh(v.as_ptr(), v.len(), 3) };
    let tok = (t - 4.0f32).abs() < 1e-6;
    println!("  k=3 阈值 = {} 期望 4.0  {}", t, if tok { "PASS" } else { "FAIL" });
    ok &= tok;

    // ---- [H/P9] exp_f 精度与速度 ----
    ok &= exp_precision_sweep();
    exp_speed_ab();

    println!("\n{}", if ok { "全部 PASS" } else { "存在 FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}
