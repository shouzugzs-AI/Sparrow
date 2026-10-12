// P3 梯度检查：解析梯度 vs 数值梯度（白皮书 §6.4 的可判定验收项）
//
// 做法：
//   1. 用同一个 .vmb 前向，取 loss
//   2. 数值梯度：对每个参数做中心差分 (f(w+h) - f(w-h)) / 2h
//   3. 解析梯度：跑反向指令流，读梯度槽
//   4. 比较相对误差
//
// 这是唯一可信的梯度正确性判据 —— 不依赖 PyTorch，自证正确。
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

unsafe fn grad_of(vm: &mut Vm, name: &str) -> *mut f32 {
    let si = vm.find_slot(name);
    vm.slot_mut(si)
}

unsafe fn param_of(vm: &mut Vm, name: &str) -> *mut f32 {
    let si = vm.find_slot(name);
    vm.slot_mut(si)
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
    let (in_dim, hid, ncls) = (m[2], m[3], m[4]);

    let x_all = rd("X");
    let y_all = rd("Y");

    println!("=== P3 梯度检查（解析 vs 数值）===");
    unsafe {
        // 用固定一个样本，避免随机性干扰
        let i = 0usize;
        let xs = &x_all[i * in_dim..(i + 1) * in_dim];
        let ys = [y_all[i]];

        // 记录初始权重（数值梯度会扰动它们，最后要还原）
        let w_names = ["W1", "b1", "W2", "b2"];
        let g_names = ["g_W1", "g_b1", "g_W2", "g_b2"];

        let buf = load_vmb(&format!("{}/train_step.vmb", dir));
        let mut vm = Vm::load(buf);

        let hp = vm.find_slot("hparams");
        let p = vm.slot_mut(hp);
        *p = 1e-3;
        *p.add(1) = 0.9;
        *p.add(2) = 0.999;
        *p.add(3) = 1e-8;
        let cn = vm.find_slot("clip_norm");
        *vm.slot_mut(cn) = 1e9; // 关掉裁剪，否则梯度被缩放

        // ---------- 1. 解析梯度 ----------
        // 需要「只跑到反向、不更新参数」的路径。
        // train_step.vml 会 ADAM 更新，所以先把 lr 设为 0 来冻结参数。
        *vm.slot_mut(hp) = 0.0;
        poke(&mut vm, "x", xs);
        poke(&mut vm, "y", &ys);
        vm.run();

        // 冻结前保存权重快照（ADAM lr=0 时不变，但仍留一份做还原）
        let mut snap: Vec<(&str, Vec<f32>, usize)> = Vec::new();
        for n in w_names {
            let si = vm.find_slot(n);
            let (r, c) = vm.slot_shape(si);
            let pn = vm.slot_mut(si);
            let v: Vec<f32> = (0..r * c).map(|k| *pn.add(k)).collect();
            snap.push((n, v, r * c));
        }

        let mut analytic: Vec<(&str, Vec<f32>)> = Vec::new();
        for (wn, gn) in w_names.iter().zip(g_names.iter()) {
            let si = vm.find_slot(gn);
            let (r, c) = vm.slot_shape(si);
            let pg = vm.slot_mut(si);
            let v: Vec<f32> = (0..r * c).map(|k| *pg.add(k)).collect();
            analytic.push((wn, v));
        }

        // ---------- 2. 数值梯度 ----------
        // f() 只跑前向。用 eval_step.vmb 更干净，但它没有 y/CE。
        // 这里直接用 train_step 但把 lr 保持 0，前向后取 loss。
        let h: f32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1e-3);
        let mut max_rel = 0.0f32;
        let mut total_checked = 0usize;
        let mut worst = String::new();

        for (wi, (wn, gv)) in analytic.iter().enumerate() {
            let si = vm.find_slot(wn);
            let (r, c) = vm.slot_shape(si);
            let n = r * c;
            let pp = vm.slot_mut(si);
            // 大张量抽样检查（数值梯度 O(n) 次前向，太慢）
            let stride = if n > 64 { n / 64 } else { 1 };

            let mut worst_this = 0.0f32;
            let mut k = 0;
            while k < n {
                let orig = *pp.add(k);
                *pp.add(k) = orig + h;
                poke(&mut vm, "x", xs);
                poke(&mut vm, "y", &ys);
                vm.run();
                let lp = vm.loss;
                *pp.add(k) = orig - h;
                poke(&mut vm, "x", xs);
                poke(&mut vm, "y", &ys);
                vm.run();
                let lm = vm.loss;
                *pp.add(k) = orig;

                let num = (lp - lm) / (2.0 * h);
                let ana = gv[k];
                // 标准梯度检查判据（cs231n 风格）：
                //   rel_err = |num - ana| / max(|num| + |ana|, 1e-8)
                // 注意分母必须用「数值梯度量级」而非固定 1e-6：
                // 当梯度本身接近 0 时，中心差分的截断/舍入误差会主导，
                // 此时绝对量级极小，不应判为错误。
                let denom = num.abs() + ana.abs();
                let rel = if denom < 1e-6 {
                    0.0 // 两侧都≈0，视为一致
                } else {
                    (num - ana).abs() / denom
                };
                let absdiff = (num - ana).abs();
                if rel > worst_this {
                    worst_this = rel;
                }
                if rel > max_rel {
                    max_rel = rel;
                    worst = format!("{}[{}] num={:.6} ana={:.6}", wn, k, num, ana);
                }
                total_checked += 1;
                k += stride;
            }
            println!(
                "  {:<4} n={:<6} 抽样 {:<4} 最大相对误差 = {:.3e}  {}",
                wn,
                n,
                (n + stride - 1) / stride,
                worst_this,
                if worst_this < 1e-2 { "PASS" } else { "FAIL" }
            );
            let _ = wi;
        }

        // 还原权重
        for (n, v, cnt) in &snap {
            let si = vm.find_slot(n);
            let pp = vm.slot_mut(si);
            for k in 0..*cnt {
                *pp.add(k) = v[k];
            }
        }

        println!("\n合计检查 {} 个参数元素", total_checked);
        println!("全局最大相对误差 = {:.3e}  ({})", max_rel, worst);
        let ok = max_rel < 1e-2;
        println!("判定: {}", if ok { "PASS（解析梯度与数值梯度一致）" } else { "FAIL" });
        let _ = (hid, ncls, param_of, grad_of);
        std::process::exit(if ok { 0 } else { 1 });
    }
}
