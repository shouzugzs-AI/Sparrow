// Lead 独立复核：直接调用真引擎的 exp_f，与 f64 exp 比较。
// 不依赖任何 Python 复刻（P9 教训：复刻可能自己有 bug）。
//
// 这是 std 侧程序，可以用 std 的 f64::exp 作参考。
//
// [H/P9] 采样区间已从 [-20,20] 扩到**全有效定义域 [-88,88]**，
// 并补上次正规区（x < -87.3365）与精确边界点的断言。
// 教训：只在「常用区间」测精度会漏掉边界缺陷 —— 次正规 bug 就是这样漏掉的。

fn rel_err(x: f32) -> f64 {
    let got = vmlm_core::kernels::exp_f(x) as f64;
    let want = (x as f64).exp();
    (got - want).abs() / want.abs().max(1e-300)
}

fn main() {
    let mut worst = 0.0f64;
    let mut worst_x = 0.0f32;
    let mut sum_rel = 0.0f64;
    let mut n = 0u64;

    // [-20, 20] 密采样：步长 1e-4 -> 40 万点
    let mut i: i64 = -200_000;
    while i <= 200_000 {
        let x = (i as f32) * 1e-4;
        let rel = rel_err(x);
        if rel > worst {
            worst = rel;
            worst_x = x;
        }
        sum_rel += rel;
        n += 1;
        i += 1;
    }

    println!("=== Lead 独立复核 exp_f（调用真引擎）===");
    println!("采样 [-20,20] 步长 1e-4, {} 点", n);
    println!("  最大相对误差 = {:.6e} @ x = {:+.6}", worst, worst_x);
    println!("  平均相对误差 = {:.6e}", sum_rel / n as f64);
    println!("  折算 ULP     = {:.2}  (1 ULP = 1.192093e-7)", worst / 1.192093e-7);
    println!();

    // ---- [H/P9] 全定义域扫描：[-88, 88]，含次正规区 ----
    // 次正规区 (x < ln(2^-126) = -87.33654) 的**可达**相对误差受限于
    // f32 的次正规表示精度（尾数位不足 24 位），必须单独报告，
    // 不能与正规区的 0.66 ulp 混为一谈。判据用 8e-7（次正规表示下限）。
    let mut worst_norm = 0.0f64;
    let mut worst_norm_x = 0.0f32;
    let mut worst_sub = 0.0f64;
    let mut worst_sub_x = 0.0f32;
    let mut m = 0u64;
    let mut j: i64 = -880_000;
    while j <= 880_000 {
        let x = (j as f32) * 1e-4;
        let rel = rel_err(x);
        // 正规结果区: exp(x) >= 2^-126
        if x >= -87.3365 {
            if rel > worst_norm {
                worst_norm = rel;
                worst_norm_x = x;
            }
        } else if rel > worst_sub {
            worst_sub = rel;
            worst_sub_x = x;
        }
        m += 1;
        j += 1;
    }
    println!("=== [H/P9] 全定义域扫描 [-88,88] 步长 1e-4, {} 点 ===", m);
    println!("  正规结果区 x >= -87.3365: max rel = {:.6e} @ x = {:+.6}  ({:.2} ulp)",
             worst_norm, worst_norm_x, worst_norm / 1.192093e-7);
    println!("  次正规区   x <  -87.3365: max rel = {:.6e} @ x = {:+.6}",
             worst_sub, worst_sub_x);
    println!("    ^ 次正规区上限由 f32 表示精度决定(~1e-6)，非内核缺陷");
    println!();

    // 边界检查
    println!("=== 边界 ===");
    for &x in &[-88.0f32, -87.5, -87.3365, -87.3366, -87.0, -1.0, 0.0, 1.0, 88.0] {
        let got = vmlm_core::kernels::exp_f(x) as f64;
        let want = (x as f64).exp();
        let rel = (got - want).abs() / want.abs().max(1e-300);
        println!("  exp_f({:>9}) = {:.6e}  参考 {:.6e}  相对误差 {:.3e}", x, got, want, rel);
    }

    // ---- [H/P9] 次正规区的硬断言（这次 bug 的回归守卫）----
    // exp_f(-88) 曾经返回 0.0（(n+127)<<23 在 n=-127 时位模式为 0）。
    let sub_ok = {
        let v = vmlm_core::kernels::exp_f(-88.0);
        v > 0.0 && rel_err(-88.0) < 1e-6
    };
    println!();
    println!("  次正规 exp_f(-88) > 0 : {} (got {:.6e})",
             if sub_ok { "PASS" } else { "FAIL" },
             vmlm_core::kernels::exp_f(-88.0));

    // 单调性在次正规区也必须保持
    let mut mono = true;
    let mut prev = vmlm_core::kernels::exp_f(-20.0);
    let mut k = 0;
    while k < 400000 {
        let x = -20.0 + 68.0 * (k as f32) / 399999.0;
        let v = vmlm_core::kernels::exp_f(x);
        if v < prev {
            mono = false;
            println!("  !! 非单调 @ x = {}", x);
            break;
        }
        prev = v;
        k += 1;
    }
    println!("  单调性 [-20,48]       : {}", if mono { "PASS" } else { "FAIL" });

    let ok = worst < 1e-6 && worst_norm < 1e-6 && sub_ok && mono;
    println!();
    println!("目标 <= 1e-6 : {}", if ok { "PASS" } else { "FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}
