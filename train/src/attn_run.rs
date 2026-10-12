// P5: Transformer 前向验证（多头注意力 + 因果掩码）
//
// 验证点：
//   1. 因果掩码：att 的上三角必须严格为 0
//   2. 逐位置 logits 与 Python 参考一致
//   3. 整块指令流一次跑完（无需状态槽）
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
    let dir = args.get(1).cloned().unwrap_or_else(|| "examples/attn".into());
    unsafe { vmlm_core::interp::init_cpu() };

    let t = 8usize;
    let v = 4usize;

    // Python 参考
    let ref_txt = std::fs::read_to_string(format!("{}/ref_logits.txt", dir)).unwrap();
    let ref_logits: Vec<Vec<f32>> = ref_txt
        .lines()
        .map(|l| l.split_whitespace().map(|s| s.parse().unwrap()).collect())
        .collect();

    println!("=== P5: Transformer 前向验证（多头注意力 + 因果掩码）===");

    unsafe {
        let buf = load_vmb(&format!("{}/block.vmb", dir));
        let mut vm = Vm::load(buf);
        vm.run();

        let li = vm.find_slot("logits");
        let lp = vm.slot_mut(li);
        let mut maxerr = 0.0f32;
        println!("  位置   VMLM logits                    参考 logits                    最大差");
        for i in 0..t {
            let got: Vec<f32> = (0..v).map(|j| *lp.add(i * v + j)).collect();
            let mut e = 0.0f32;
            for j in 0..v {
                let d = (got[j] - ref_logits[i][j]).abs();
                if d > e {
                    e = d;
                }
            }
            if e > maxerr {
                maxerr = e;
            }
            println!(
                "   {}     [{}]  [{}]  {:.3e}",
                i,
                got.iter().map(|x| format!("{:>8.4}", x)).collect::<Vec<_>>().join(","),
                ref_logits[i]
                    .iter()
                    .map(|x| format!("{:>8.4}", x))
                    .collect::<Vec<_>>()
                    .join(","),
                e
            );
        }

        // 因果性硬检查：att 的上三角必须为 0
        let ai = vm.find_slot("att");
        let ap = vm.slot_mut(ai);
        let mut causal_ok = true;
        for i in 0..t {
            for j in (i + 1)..t {
                if (*ap.add(i * t + j)).abs() > 1e-6 {
                    causal_ok = false;
                }
            }
        }
        println!(
            "\n  因果掩码检查（att 上三角必须为 0）: {}",
            if causal_ok { "PASS" } else { "FAIL —— 存在未来信息泄漏" }
        );

        println!("\n  全局最大绝对误差 = {:.3e}", maxerr);
        let ok = maxerr < 1e-4 && causal_ok;
        println!(
            "  判定: {}",
            if ok {
                "PASS（注意力 + 因果掩码 + FFN 全链路与 Python 参考一致）"
            } else {
                "FAIL"
            }
        );
        std::process::exit(if ok { 0 } else { 1 });
    }
}
