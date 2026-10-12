// 独立的 std 侧探针（成员 G，P8 任务 3 验收用）
//
// 目的：给出「读文件 + std 驱动」路径下每步 loss 的**原始 f32**，
// 与 AOT 侧 `--dump-loss=` 的输出做 SHA256 逐位比对。
//
// 为什么不直接用 vmlm-train：它只打印 4/6 位小数（train_mnist.rs:135），
// 而验收要的是逐位一致 —— 打印会把真实差异四舍五入掉，也会把纯粹的
// 舍入差误报成不一致。train/ 是别人的作用域，我不改它，改为独立探针。
//
// 语义**逐条复刻** train_mnist.rs 的训练循环（89..128 行）：
//   i = step % ntr;  x = X[i*196..]; y = Y[i];  vm.run();  loss = vm.loss
//
// 构建（从 workspace 根，std crate 不能用 runtime 的 config）：
//   cd <repo>
//   cargo build -p vmlm-g-stdprobe --release
// 运行：
//   vmlm-g-stdprobe.exe examples/mnist 200 <out.bin>

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

unsafe fn poke(vm: &mut Vm, name: &str, vals: &[f32]) {
    let si = vm.find_slot(name);
    let p = vm.slot_mut(si);
    for (i, v) in vals.iter().enumerate() {
        *p.add(i) = *v;
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).cloned().unwrap_or_else(|| "examples/mnist".into());
    let steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let out = args.get(3).cloned();

    unsafe { vmlm_core::interp::init_cpu() };

    let rd = |n: &str| -> Vec<f32> {
        let b = std::fs::read(format!("{}/{}.bin", dir, n))
            .unwrap_or_else(|_| panic!("读不到 {}.bin", n));
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let meta = std::fs::read_to_string(format!("{}/meta.txt", dir)).unwrap();
    let m: Vec<usize> = meta.split_whitespace().map(|s| s.parse().unwrap()).collect();
    let (ntr, in_dim) = (m[0], m[2]);

    let x_all = rd("X");
    let y_all = rd("Y");

    unsafe {
        let buf = load_vmb(&format!("{}/train_step.vmb", dir));
        let mut vm = Vm::load(buf);

        let hp = vm.find_slot("hparams");
        let p = vm.slot_mut(hp);
        *p = 1e-3;
        *p.add(1) = 0.9;
        *p.add(2) = 0.999;
        *p.add(3) = 1e-8;
        let cn = vm.find_slot("clip_norm");
        *vm.slot_mut(cn) = 5.0;

        let mut losses: Vec<f32> = Vec::with_capacity(steps);
        for step in 0..steps {
            let i = step % ntr;
            let xs = &x_all[i * in_dim..(i + 1) * in_dim];
            let ys = [y_all[i]];
            poke(&mut vm, "x", xs);
            poke(&mut vm, "y", &ys);
            vm.run();
            losses.push(vm.loss);
        }

        println!(
            "probe: {} steps, first10 avg {:.4}, last10 avg {:.4}",
            steps,
            losses[..10.min(losses.len())].iter().sum::<f32>() / 10.0_f32.min(losses.len() as f32),
            losses[losses.len().saturating_sub(10)..].iter().sum::<f32>()
                / 10.0_f32.min(losses.len() as f32)
        );

        if let Some(o) = out {
            let mut bytes = Vec::with_capacity(losses.len() * 4);
            for v in &losses {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(&o, &bytes).unwrap();
            println!("wrote {} B -> {}", bytes.len(), o);
        }
    }
}
