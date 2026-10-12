// P4: RNN 驱动 —— 逐步喂序列，验证递归态（状态槽）跨 step 正确
//
// 关键验证点：
//   1. 状态槽 st_S / st_xprev 跨 step 保持（不重置）
//   2. 与 Python 逐步参考实现逐位一致
//   3. 同一份 .vmb 跑 T 步不需要重新编译（BPTT 截断由驱动控制）
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

unsafe fn set_slot(vm: &mut Vm, name: &str, vals: &[f32]) {
    let si = vm.find_slot(name);
    let p = vm.slot_mut(si);
    for (i, v) in vals.iter().enumerate() {
        *p.add(i) = *v;
    }
}

unsafe fn get_slot(vm: &mut Vm, name: &str, n: usize) -> Vec<f32> {
    let si = vm.find_slot(name);
    let p = vm.slot_mut(si);
    (0..n).map(|i| *p.add(i)).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).cloned().unwrap_or_else(|| "examples/rnn".into());
    unsafe { vmlm_core::interp::init_cpu() };

    let rd = |n: &str| -> Vec<f32> {
        let b = std::fs::read(format!("{}/{}.bin", dir, n))
            .unwrap_or_else(|_| panic!("读不到 {}.bin", n));
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };

    let (h, d, v, t_steps) = (32usize, 16usize, 4usize, 8usize);
    let xseq = rd("x"); // [T, H]

    // Python 参考
    let ref_txt = std::fs::read_to_string(format!("{}/ref_logits.txt", dir)).unwrap();
    let ref_logits: Vec<Vec<f32>> = ref_txt
        .lines()
        .map(|l| l.split_whitespace().map(|s| s.parse().unwrap()).collect())
        .collect();

    println!("=== P4: RNN 递归态验证 ===");
    unsafe {
        let buf = load_vmb(&format!("{}/ese_step.vmb", dir));
        let mut vm = Vm::load(buf);

        let mut maxerr = 0.0f32;
        let mut all_logits = Vec::new();

        for t in 0..t_steps {
            // 喂第 t 个 token 的嵌入
            let xs = &xseq[t * h..(t + 1) * h];
            set_slot(&mut vm, "x", xs);

            // 执行一步（状态槽跨 step 保持，不需要重置）
            vm.run();

            let lg = get_slot(&mut vm, "logits", v);
            all_logits.push(lg.clone());

            let mut e = 0.0f32;
            for k in 0..v {
                let de = (lg[k] - ref_logits[t][k]).abs();
                if de > e {
                    e = de;
                }
            }
            if e > maxerr {
                maxerr = e;
            }
            println!(
                "  step {}: VMLM {:?}\n          参考 {:?}   最大差 {:.3e}",
                t,
                lg.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>(),
                ref_logits[t]
                    .iter()
                    .map(|x| format!("{:.4}", x))
                    .collect::<Vec<_>>(),
                e
            );
        }

        // 递归态最终值
        let s = get_slot(&mut vm, "st_S", d * d);
        println!("\n末步递归态 st_S[0,:4] = {:?}", &s[..4]);

        println!("\n全局最大绝对误差 = {:.3e}", maxerr);
        let ok = maxerr < 1e-3;  // 8 步递归的 float32 累积误差上限
        println!(
            "判定: {}",
            if ok {
                "PASS（VMLM 递归态与 Python 参考一致，状态槽跨 step 正确保持）"
            } else {
                "FAIL"
            }
        );
        std::process::exit(if ok { 0 } else { 1 });
    }
}
