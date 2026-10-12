// AOT 训练闭环：`--features "aot train"` 下的**真正**训练驱动（P8 成员 G）
//
// 背景：P7-B 遗留问题 3 —— 「未测 `--features "aot train"` 下的完整训练闭环」。
// 实测下来 AOT 构建跑 `train_step.vmb` 只是「执行一遍指令流然后退出」：
//   * 不崩（exit=0），但**没有任何输出、也没有训练发生**
//   * 因为 .vmb 里只有「一个训练步」，而真正训练需要外层循环：
//     喂样本 -> vm.run() -> 读 loss -> 下一个样本
//
// 这个「外层循环 + 喂数据」在原设计里是 std 侧驱动的活儿
// （train/src/train_mnist.rs 的 poke()/peek()/循环）。AOT 的意义是
// **不读任何外部文件**，所以数据集也必须一起烧进 exe。
//
// 做法：把「数据集 + 训练循环」也用同一套常量内嵌机制烧进来。
//   * 模型：复用 aot.rs 的单模型路径（或 aot-multi 的 --model=）
//   * 数据：本模块按名字从内嵌块里取
//   * 循环：完全照抄 std 驱动的语义（训练步 N 次 + 测试集评估）
//
// 与 std 驱动的**逐位一致性要求**：
//   同样 200 步、同样样本顺序（step % ntr）、同样超参，loss 序列必须一致。
//   train_mnist.rs 的循环语义（见该文件 89..128 行）：
//       i = step % ntr
//       poke("x", X[i*196..]); poke("y", Y[i]); vm.run();
//       loss = vm.loss
//   这里逐条复刻。
//
// 无 CRT 约束：不 println!，只用 vmlm_core::print。

use crate::args;
use crate::aot;
use crate::aot_multi;
use vmlm_core::interp::Vm;

// 内嵌数据集：由 build.rs 生成到 OUT_DIR（约 19 MB 数据 -> 128 MB 源码，
// 不适合当源码提交，所以构建期生成）。加 `--features aot-offline` 时
// 回落到 tools/aot_embed.py 离线生成的 src/data_aot.rs。
#[cfg(not(feature = "aot-offline"))]
include!(concat!(env!("OUT_DIR"), "/data_aot.rs"));

#[cfg(feature = "aot-offline")]
include!("data_aot.rs");

/// 装载一个内嵌模型（按名字，走 aot-multi 的名字表）。
///
/// **必须走 `aot::copy_to_aligned_writable`**：`static` 落在只读 .rdata，
/// 而 `Vm::load` 把常量池当可写区（OP_ADAM 就地更新参数）——直接喂会
/// 0xC0000005（P7-B 踩过的坑，训练路径**必然**触发，纯推理模型不会）。
unsafe fn load_model(name: &[u8]) -> Vm {
    match aot_multi::model_bytes(name) {
        Some(b) => Vm::load(aot::copy_to_aligned_writable(b.as_ptr(), b.len())),
        None => {
            vmlm_core::print(b"embedded model not found: ");
            vmlm_core::print(name);
            vmlm_core::print(b"\n");
            vmlm_core::exit(9)
        }
    }
}

unsafe fn print_u64(mut v: u64) {
    let mut d = [0u8; 20];
    let mut n = 0;
    if v == 0 {
        vmlm_core::print(b"0");
        return;
    }
    while v > 0 {
        d[n] = b'0' + (v % 10) as u8;
        v /= 10;
        n += 1;
    }
    let mut out = [0u8; 20];
    let mut i = 0;
    while i < n {
        out[i] = d[n - 1 - i];
        i += 1;
    }
    vmlm_core::print(&out[..n]);
}

/// 打印 f32，定点 4 位小数（不用 core::fmt）。
unsafe fn print_f32(v: f32) {
    let mut buf = [0u8; 32];
    let mut len = 0usize;
    let neg = v < 0.0;
    let av = if neg { -v } else { v };
    let ip = av as u64;
    let mut d = [0u8; 20];
    let mut dn = 0;
    let mut t = ip;
    if t == 0 {
        d[0] = b'0';
        dn = 1;
    }
    while t > 0 {
        d[dn] = b'0' + (t % 10) as u8;
        t /= 10;
        dn += 1;
    }
    if neg {
        buf[len] = b'-';
        len += 1;
    }
    let mut k = dn;
    while k > 0 {
        k -= 1;
        buf[len] = d[k];
        len += 1;
    }
    buf[len] = b'.';
    len += 1;
    let mut frac = av - (ip as f32);
    for _ in 0..4 {
        frac *= 10.0;
        let dig = frac as u32;
        buf[len] = b'0' + (dig % 10) as u8;
        len += 1;
        frac -= dig as f32;
    }
    vmlm_core::print(&buf[..len]);
}

/// 按名字取内嵌数据块，返回 (指针, f32 元素个数)。
///
/// 注意：`name` 传**不带** NUL 的字节串（如 `b"META"`）。表项里存的是
/// `b"META\0"`，这里剥掉结尾 NUL 再比 —— 早先我在调用点传了 `b"META\0"`，
/// 结果长度不等永远匹配不上（实测 "data block not found: META"）。
unsafe fn data(name: &[u8]) -> (*const f32, usize) {
    let mut i = 0;
    while i < AOT_DATA.len() {
        let e = &AOT_DATA[i];
        let n = e.name.len() - 1;
        if n == name.len() && args::eq(&e.name[..n], name) {
            return (e.bytes.as_ptr() as *const f32, e.bytes.len() / 4);
        }
        i += 1;
    }
    vmlm_core::print(b"data block not found: ");
    vmlm_core::print(name);
    vmlm_core::print(b"\n");
    vmlm_core::exit(9)
}

unsafe fn poke(vm: &mut Vm, name: &str, vals: *const f32, n: usize) {
    let si = vm.find_slot(name);
    let p = vm.slot_mut(si);
    let mut i = 0;
    while i < n {
        *p.add(i) = *vals.add(i);
        i += 1;
    }
}

unsafe fn peek_argmax(vm: &mut Vm, name: &str, n: usize) -> usize {
    let si = vm.find_slot(name);
    let p = vm.slot_mut(si);
    let mut best = 0usize;
    let mut k = 1usize;
    while k < n {
        if *p.add(k) > *p.add(best) {
            best = k;
        }
        k += 1;
    }
    best
}

/// 跑完整训练闭环，返回退出码。
///
/// 用**两条**内嵌指令流，与 std 驱动（train_mnist.rs）完全一致：
///   * `TRAIN_MODEL`（默认 mnist_train，含反向 + ADAM）用于训练
///   * `EVAL_MODEL`（默认 mnist_eval，只前向）用于测试集评估，
///     评估前把训练好的权重从 train vm 拷进 eval vm
///
/// 为什么必须两条：`train_step.vmb` 跑完前向还会继续跑 CE_BWD / CLIP_GRAD /
/// ADAM —— 如果拿它做评估，每次 `vm.run()` 都会**继续训练**（而且测试样本
/// 没设 y，用的是上一个训练样本的标签）。实测这个错误会让测试准确率
/// 从 94.5% 掉到 22.8%，但**不报任何错**。
///
/// 名字可用 `--train-model=` / `--eval-model=` 覆盖。
///
/// # Safety
/// 只应在启动时调用一次。
pub unsafe fn run_training() -> u32 {
    // 内嵌的元信息：meta.txt = "ntr nte in_dim hid ncls"
    // 注意下标：ncls 是**第 5 个**（add(4)），第 4 个 add(3) 是 hid。
    // 早先我误把 add(3) 当 ncls，于是按 64 类取 argmax（实际只有 4 类），
    // 训练准确率恒为 0 —— 又是一个不报错的静默错误。
    let (mp, _) = data(b"META");
    let ntr = *mp as usize;
    let nte = *mp.add(1) as usize;
    let in_dim = *mp.add(2) as usize;
    let ncls = *mp.add(4) as usize;

    let (xp, _) = data(b"X");
    let (yp, _) = data(b"Y");
    let (xtep, _) = data(b"XTE");
    let (ytep, _) = data(b"YTE");

    // 模型名（可覆盖）。
    // 注意：`b"..."` 字面量长度不同，类型是 `&[u8; N]`，直接
    // `unwrap_or(b"mnist_train")` 会 E0308（N 不一致）——要显式切片成 `&[u8]`。
    let default_train: &[u8] = b"mnist_train";
    let default_eval: &[u8] = b"mnist_eval";
    let mut mbuf = [0u8; 512];
    let train_name: &[u8] = match args::find_flag(b"train-model", &mut mbuf) {
        Some(v) => v,
        None => default_train,
    };
    let mut ebuf = [0u8; 512];
    let eval_name: &[u8] = match args::find_flag(b"eval-model", &mut ebuf) {
        Some(v) => v,
        None => default_eval,
    };

    let mut vm = load_model(train_name);

    // 步数：--steps=N，默认 200（与 std 驱动默认一致）
    let mut sbuf = [0u8; 512];
    let steps = match args::find_flag(b"steps", &mut sbuf) {
        Some(v) => {
            let mut n = 0usize;
            let mut i = 0;
            while i < v.len() && v[i] >= b'0' && v[i] <= b'9' {
                n = n * 10 + (v[i] - b'0') as usize;
                i += 1;
            }
            if n == 0 {
                200
            } else {
                n
            }
        }
        None => 200,
    };

    vmlm_core::print(b"=== VMLM AOT training (no external files) ===\n");
    vmlm_core::print(b"data: train ");
    print_u64(ntr as u64);
    vmlm_core::print(b" test ");
    print_u64(nte as u64);
    vmlm_core::print(b" | in_dim ");
    print_u64(in_dim as u64);
    vmlm_core::print(b" hid ");
    print_u64(*mp.add(3) as u64);
    vmlm_core::print(b" ncls ");
    print_u64(ncls as u64);
    vmlm_core::print(b" | steps ");
    print_u64(steps as u64);
    vmlm_core::print(b"\nmodels: train=");
    vmlm_core::print(train_name);
    vmlm_core::print(b" eval=");
    vmlm_core::print(eval_name);
    vmlm_core::print(b"\n");

    // 超参（与 train_mnist.rs 逐条一致）
    let hp = vm.find_slot("hparams");
    let p = vm.slot_mut(hp);
    *p = 1e-3;
    *p.add(1) = 0.9;
    *p.add(2) = 0.999;
    *p.add(3) = 1e-8;
    let cn = vm.find_slot("clip_norm");
    *vm.slot_mut(cn) = 5.0;
    let mut correct = 0usize;
    let mut seen = 0usize;
    let mut first10 = 0.0f32; // 首 10 步 loss 和
    let mut nfirst = 0usize;
    let mut last10 = 0.0f32; // 末 10 步 loss 和（滚动窗口）
    let mut ring = [0.0f32; 10];
    let mut ring_n = 0usize;

    let mut step = 0usize;

    // `--dump-loss=<path>`：把每步 loss 的**原始 f32** 顺序写文件，
    // 供「AOT vs std 驱动逐位一致」的验收比对（比打印的 4 位小数强得多：
    // 打印会把真实差异四舍五入掉，也会把纯粹的舍入差误报成不一致）。
    //
    // 注意：`find_flag` 会往传进来的 buffer 里写，所以这里**不能**把
    // 它的返回切片再拿去做别的事；先拷贝到自己的缓冲里再用。
    let mut dbuf = [0u8; 512];
    let mut path_buf = [0u8; 512];
    let mut path_len = 0usize;
    let mut has_dump = false;
    if let Some(v) = args::find_flag(b"dump-loss", &mut dbuf) {
        path_len = v.len().min(510);
        let mut k = 0;
        while k < path_len {
            path_buf[k] = v[k];
            k += 1;
        }
        has_dump = path_len > 0;
    }
    let mut loss_dump = [0.0f32; 512];
    let mut ndump = 0usize;

    while step < steps {
        let i = step % ntr;
        poke(&mut vm, "x", xp.add(i * in_dim), in_dim);
        poke(&mut vm, "y", yp.add(i), 1);
        vm.run();

        let loss = vm.loss;
        if ndump < loss_dump.len() {
            loss_dump[ndump] = loss;
            ndump += 1;
        }
        if nfirst < 10 {
            first10 += loss;
            nfirst += 1;
        }
        ring[step % 10] = loss;
        if ring_n < 10 {
            ring_n += 1;
        }

        let best = peek_argmax(&mut vm, "logits", ncls);
        if best == *yp.add(i) as usize {
            correct += 1;
        }
        seen += 1;

        // 每 20 步（与 std 驱动同频）报一行
        if step % 20 == 0 || step + 1 == steps {
            vmlm_core::print(b"  step ");
            print_u64(step as u64);
            vmlm_core::print(b"  loss ");
            print_f32(loss);
            vmlm_core::print(b"  train_acc ");
            print_f32(correct as f32 / seen as f32);
            vmlm_core::print(b"\n");
        }
        step += 1;
    }

    last10 = 0.0;
    let mut i2 = 0usize;
    while i2 < ring_n {
        last10 += ring[i2];
        i2 += 1;
    }

    // 落盘 loss 序列（原始 f32）
    if has_dump {
        path_buf[path_len] = 0;
        vmlm_core::dump_f32(path_buf.as_ptr(), loss_dump.as_ptr(), ndump);
    }

    vmlm_core::print(b"\ntrain: ");
    print_u64(steps as u64);
    vmlm_core::print(b" steps\n");
    vmlm_core::print(b"loss: first10 avg ");
    print_f32(first10 / nfirst as f32);
    vmlm_core::print(b" -> last10 avg ");
    print_f32(last10 / ring_n as f32);
    vmlm_core::print(b"\n");

    // ---- 测试集评估 ----
    // 装载**只前向**的 eval 指令流，并把训练好的权重拷过去。
    // 这与 std 驱动 train_mnist.rs(140..147) 完全一致。
    // 绝不能复用 train vm：train_step 在 HALT 前还有 CE_BWD/CLIP_GRAD/ADAM，
    // 再 run() 会继续训练（且测试样本没设 y）——实测准确率 94.5% -> 22.8%，
    // 且不报任何错。
    let mut ev = load_model(eval_name);
    let names: [&[u8]; 4] = [b"W1", b"b1", b"W2", b"b2"];
    let mut ni = 0usize;
    while ni < names.len() {
        // find_slot 收 &str；这里只有 ASCII 名字，手工按字节查更省事：
        // 直接借用 Vm::find_slot 的 &str 要求，把字节名转成 str。
        let nm = core::str::from_utf8_unchecked(names[ni]);
        let src = vm.find_slot(nm);
        let dst = ev.find_slot(nm);
        let (r, c) = vm.slot_shape(src);
        core::ptr::copy_nonoverlapping(vm.slot_mut(src), ev.slot_mut(dst), r * c);
        ni += 1;
    }

    let mut ok = 0usize;
    let mut i = 0usize;
    while i < nte {
        poke(&mut ev, "x", xtep.add(i * in_dim), in_dim);
        ev.run();
        let best = peek_argmax(&mut ev, "logits", ncls);
        if best == *ytep.add(i) as usize {
            ok += 1;
        }
        i += 1;
    }
    vmlm_core::print(b"test: ");
    print_u64(ok as u64);
    vmlm_core::print(b"/");
    print_u64(nte as u64);
    vmlm_core::print(b" = ");
    print_f32(ok as f32 / nte as f32 * 100.0);
    vmlm_core::print(b"%\n");

    // 判定：loss 必须显著下降（可判定）
    let f = first10 / nfirst as f32;
    let l = last10 / ring_n as f32;
    if l < f * 0.6 {
        vmlm_core::print(b"VERDICT: PASS (loss dropped >40%)\n");
        0
    } else {
        vmlm_core::print(b"VERDICT: FAIL (loss did not drop enough)\n");
        1
    }
}
