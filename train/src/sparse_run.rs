// P6: 动态稀疏验证
//
// 两个必须同时成立的判据：
//   1. 输出与 Python 参考一致（掩码选对了）
//   2. **稀疏率恒等于 top_ratio**（每行恰好 k 个非零）—— sparse_dyn 的硬保证
use vmlm_core::interp::Vm;

extern "C" {
    fn malloc(n: usize) -> *mut u8;
}

unsafe fn load_vmb(path: &str) -> *mut u8 {
    let data = std::fs::read(path).unwrap_or_else(|_| panic!("读不到 {}", path));
    let n = data.len();
    let p = malloc(n + 64);
    std::ptr::copy_nonoverlapping(data.as_ptr(), p, n);
    for i in n..n + 64 {
        *p.add(i) = 0;
    }
    p
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).cloned().unwrap_or_else(|| "examples/sparse".into());
    unsafe { vmlm_core::interp::init_cpu() };

    let (t, d, f, k) = (4usize, 16usize, 32usize, 8usize);

    let rd = |n: &str| -> Vec<f32> {
        std::fs::read(format!("{}/{}.bin", dir, n))
            .unwrap_or_else(|_| panic!("读不到 {}.bin", n))
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let _ = rd("x");

    let ref_y_txt = std::fs::read_to_string(format!("{}/ref_y.txt", dir)).unwrap();
    let ref_y: Vec<Vec<f32>> = ref_y_txt
        .lines()
        .map(|l| l.split_whitespace().map(|s| s.parse().unwrap()).collect())
        .collect();

    println!("=== P6: 动态稀疏验证（sparse_dyn 神经元级 top-k）===");
    println!("  配置: T={} D={} F={} k={} top_ratio={:.2}", t, d, f, k, k as f32 / f as f32);

    unsafe {
        let buf = load_vmb(&format!("{}/ffn.vmb", dir));
        let mut vm = Vm::load(buf);
        vm.run();

        // ---- 1. 输出对比 ----
        let yi = vm.find_slot("y");
        let yp = vm.slot_mut(yi);
        let mut maxerr = 0.0f32;
        for i in 0..t {
            for j in 0..d {
                let e = (*yp.add(i * d + j) - ref_y[i][j]).abs();
                if e > maxerr {
                    maxerr = e;
                }
            }
        }

        // ---- 2. 稀疏率不变量 ----
        // hs = h ⊙ mask，逐行统计非零个数
        let hi = vm.find_slot("hs");
        let hp = vm.slot_mut(hi);
        let mut counts = Vec::new();
        let mut total_nz = 0usize;
        for i in 0..t {
            let mut nz = 0usize;
            for j in 0..f {
                if (*hp.add(i * f + j)).abs() > 1e-12 {
                    nz += 1;
                }
            }
            counts.push(nz);
            total_nz += nz;
        }
        let ratio = total_nz as f32 / (t * f) as f32;
        let exact = counts.iter().all(|&c| c == k);

        // 容差取 1e-3：GELU 用 tanh 近似，与 numpy 的双精度 tanh 在
        // 数值边缘会有 ~1e-4 量级的差异，这是实现口径差异不是错误。
        let out_ok = maxerr < 1e-3;
        println!("\n  输出最大绝对误差 = {:.3e}  {}", maxerr, if out_ok { "PASS" } else { "FAIL" });
        println!("  每行非零个数   = {:?}（期望全为 {}）", counts, k);
        println!("  实际稀疏率     = {:.4}（期望 {:.4}）", ratio, k as f32 / f as f32);
        println!(
            "  稀疏率不变量   = {}",
            if exact && (ratio - k as f32 / f as f32).abs() < 1e-6 {
                "PASS（恒等于 top_ratio，无浮动）"
            } else {
                "FAIL（稀疏率未恒定 —— 说明用了阈值比较而非「取第 k 名」）"
            }
        );

        let ok = out_ok && exact;
        println!("\n  判定: {}", if ok { "PASS" } else { "FAIL" });
        std::process::exit(if ok { 0 } else { 1 });
    }
}
