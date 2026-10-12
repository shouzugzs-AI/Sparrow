// 跑一次反向（lr=0 冻结参数），把解析梯度写成 _vmlm_grads.bin
// 供 tools/gradcheck_vs_torch.py 与 PyTorch autograd 对比。
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
    unsafe { vmlm_core::interp::init_cpu() };

    let rd = |n: &str| -> Vec<f32> {
        let b = std::fs::read(format!("{}/{}.bin", dir, n)).unwrap();
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let meta = std::fs::read_to_string(format!("{}/meta.txt", dir)).unwrap();
    let m: Vec<usize> = meta.split_whitespace().map(|s| s.parse().unwrap()).collect();
    let (in_dim, ncls) = (m[2], m[4]);

    let x_all = rd("X");
    let y_all = rd("Y");
    let i = 0usize;
    let xs = &x_all[i * in_dim..(i + 1) * in_dim];
    let ys = [y_all[i]];

    unsafe {
        let buf = load_vmb(&format!("{}/train_step.vmb", dir));
        let mut vm = Vm::load(buf);

        // lr=0 → ADAM 不改变权重（纯反向），且关掉梯度裁剪
        let hp = vm.find_slot("hparams");
        let p = vm.slot_mut(hp);
        *p = 0.0;
        *p.add(1) = 0.9;
        *p.add(2) = 0.999;
        *p.add(3) = 1e-8;
        let cn = vm.find_slot("clip_norm");
        *vm.slot_mut(cn) = 1e9;

        poke(&mut vm, "x", xs);
        poke(&mut vm, "y", &ys);
        vm.run();

        println!("VMLM loss = {:.6}", vm.loss);

        // 按 W1, b1, W2, b2 顺序拼出梯度
        let mut out: Vec<f32> = Vec::new();
        for gn in ["g_W1", "g_b1", "g_W2", "g_b2"] {
            let si = vm.find_slot(gn);
            let (r, c) = vm.slot_shape(si);
            let pg = vm.slot_mut(si);
            for k in 0..r * c {
                out.push(*pg.add(k));
            }
        }
        let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(format!("{}/_vmlm_grads.bin", dir), bytes).unwrap();
        println!("写出 {} 个梯度值 -> {}/_vmlm_grads.bin", out.len(), dir);
        let _ = ncls;
    }
}
