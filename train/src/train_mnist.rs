// VMLM 训练驱动（std 侧，体积不受限）
//
// 职责：加载 .vmb，按训练步循环执行，打印 loss / 准确率 / CPU 占用。
// 训练逻辑全部在 .vml 指令流里（引擎不做自动微分）。
use std::time::Instant;
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

// 覆盖常量池里某个张量的内容（把数据集/标签/超参喂进去）
unsafe fn poke(vm: &mut Vm, name: &str, vals: &[f32]) {
    let si = vm.find_slot(name);
    let p = vm.slot_mut(si);
    for (i, v) in vals.iter().enumerate() {
        *p.add(i) = *v;
    }
}

unsafe fn peek(vm: &mut Vm, name: &str, n: usize) -> Vec<f32> {
    let si = vm.find_slot(name);
    let p = vm.slot_mut(si);
    (0..n).map(|i| *p.add(i)).collect()
}

unsafe fn scalar(vm: &mut Vm, name: &str) -> f32 {
    let si = vm.find_slot(name);
    *vm.slot_mut(si)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).cloned().unwrap_or_else(|| "examples/mnist".into());
    let steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);

    println!("=== VMLM 训练 (P3) ===");
    println!("目录: {}", dir);
    unsafe { vmlm_core::interp::init_cpu() };

    // 读数据集
    let rd = |n: &str| -> Vec<f32> {
        let b = std::fs::read(format!("{}/{}.bin", dir, n))
            .unwrap_or_else(|_| panic!("读不到 {}.bin", n));
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let meta = std::fs::read_to_string(format!("{}/meta.txt", dir)).unwrap();
    let m: Vec<usize> = meta.split_whitespace().map(|s| s.parse().unwrap()).collect();
    let (ntr, nte, in_dim, hid, ncls) = (m[0], m[1], m[2], m[3], m[4]);
    println!("数据: 训练 {} 测试 {} | 结构 {}->{}->{}", ntr, nte, in_dim, hid, ncls);

    let x_all = rd("X");
    let y_all = rd("Y");
    let xte_all = rd("Xte");
    let yte_all = rd("Yte");

    unsafe {
        let buf = load_vmb(&format!("{}/train_step.vmb", dir));
        let mut vm = Vm::load(buf);

        // 超参槽
        let hp = vm.find_slot("hparams");
        let p = vm.slot_mut(hp);
        *p = 1e-3;        // lr
        *p.add(1) = 0.9;  // b1
        *p.add(2) = 0.999;// b2
        *p.add(3) = 1e-8; // eps
        let cn = vm.find_slot("clip_norm");
        *vm.slot_mut(cn) = 5.0;

        let mut losses = Vec::with_capacity(steps);
        let t0 = Instant::now();
        let mut correct = 0usize;
        let mut seen = 0usize;

        for step in 0..steps {
            let i = step % ntr;
            // 喂一个样本
            let xs = &x_all[i * in_dim..(i + 1) * in_dim];
            let ys = [y_all[i]];
            poke(&mut vm, "x", xs);
            poke(&mut vm, "y", &ys);

            vm.run();

            let loss = vm.loss;
            losses.push(loss);

            // 训练准确率（用当前 logits）
            let lg = peek(&mut vm, "logits", ncls);
            let mut best = 0;
            for k in 1..ncls {
                if lg[k] > lg[best] {
                    best = k;
                }
            }
            if best == y_all[i] as usize {
                correct += 1;
            }
            seen += 1;

            if step % 20 == 0 || step + 1 == steps {
                let w = 50.min(step + 1);
                let avg: f32 = losses[step + 1 - w..].iter().sum::<f32>() / w as f32;
                println!(
                    "  step {:>4}  loss {:>8.4}  最近{}步均值 {:>8.4}  训练acc {:.3}  ({:.0} ms)",
                    step,
                    loss,
                    w,
                    avg,
                    correct as f32 / seen as f32,
                    t0.elapsed().as_secs_f64() * 1000.0
                );
            }
        }

        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let first10: f32 = losses[..10.min(losses.len())].iter().sum::<f32>() / 10.0_f32.min(losses.len() as f32);
        let last10: f32 = losses[losses.len().saturating_sub(10)..].iter().sum::<f32>()
            / 10.0_f32.min(losses.len() as f32);
        println!("\n训练: {} 步, {:.1} ms, {:.3} ms/step", steps, train_ms, train_ms / steps as f64);
        println!("loss: 首10步均值 {:.4} -> 末10步均值 {:.4}", first10, last10);

        // 测试集推理：用只前向的 vmb，并把训练好的权重拷进去
        // （训练只改常量池 —— 这正是「不重新编译算子逻辑」的体现，
        //   换了指令流依然复用同一份权重内存布局）
        let ebuf = load_vmb(&format!("{}/eval_step.vmb", dir));
        let mut ev = Vm::load(ebuf);
        for name in ["W1", "b1", "W2", "b2"] {
            let src = vm.find_slot(name);
            let dst = ev.find_slot(name);
            let (r, c) = vm.slot_shape(src);
            std::ptr::copy_nonoverlapping(vm.slot_mut(src), ev.slot_mut(dst), r * c);
        }

        let t1 = Instant::now();
        let mut ok = 0usize;
        for i in 0..nte {
            let xs = &xte_all[i * in_dim..(i + 1) * in_dim];
            poke(&mut ev, "x", xs);
            ev.run();
            let lg = peek(&mut ev, "logits", ncls);
            let mut best = 0;
            for k in 1..ncls {
                if lg[k] > lg[best] {
                    best = k;
                }
            }
            if best == yte_all[i] as usize {
                ok += 1;
            }
        }
        let eval_ms = t1.elapsed().as_secs_f64() * 1000.0;
        println!(
            "测试: {}/{} = {:.2}%  ({:.1} ms, {:.4} ms/样本)",
            ok,
            nte,
            ok as f32 / nte as f32 * 100.0,
            eval_ms,
            eval_ms / nte as f64
        );
    }
}
