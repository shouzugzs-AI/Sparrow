// 基准/调试入口：argv 解析 + 计时 + 结果 dump
// 只在 --features bench 时编译，因此不计入 30KB 引擎体积
use vmlm_core::interp::Vm;

#[link(name = "kernel32")]
extern "system" {
    fn GetCommandLineA() -> *const u8;
    fn QueryPerformanceCounter(out: *mut i64) -> i32;
    fn QueryPerformanceFrequency(out: *mut i64) -> i32;
}

unsafe fn arg(n: usize, dst: *mut u8) {
    let cl = GetCommandLineA();
    let mut i = 0usize;
    let mut idx = 0usize;
    loop {
        while *cl.add(i) == b' ' || *cl.add(i) == b'\t' {
            i += 1;
        }
        if *cl.add(i) == 0 {
            *dst = 0;
            return;
        }
        let start = i;
        let quoted = *cl.add(i) == b'"';
        if quoted {
            i += 1;
            let s = i;
            while *cl.add(i) != b'"' && *cl.add(i) != 0 {
                i += 1;
            }
            if idx == n {
                let mut k = 0;
                while s + k < i {
                    *dst.add(k) = *cl.add(s + k);
                    k += 1;
                }
                *dst.add(k) = 0;
                return;
            }
            if *cl.add(i) == b'"' {
                i += 1;
            }
        } else {
            while *cl.add(i) != b' ' && *cl.add(i) != b'\t' && *cl.add(i) != 0 {
                i += 1;
            }
            if idx == n {
                let mut k = 0;
                while start + k < i {
                    *dst.add(k) = *cl.add(start + k);
                    k += 1;
                }
                *dst.add(k) = 0;
                return;
            }
        }
        idx += 1;
    }
}

fn parse_u32(b: &[u8]) -> u32 {
    let mut v = 0u32;
    let mut i = 0;
    while i < b.len() && b[i] >= b'0' && b[i] <= b'9' {
        v = v * 10 + (b[i] - b'0') as u32;
        i += 1;
    }
    v
}

fn find_byte(b: &[u8], c: u8) -> Option<usize> {
    let mut i = 0;
    while i < b.len() {
        if b[i] == c {
            return Some(i);
        }
        i += 1;
    }
    None
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
    for i in 0..n {
        out[i] = d[n - 1 - i];
    }
    vmlm_core::print(&out[..n]);
}

// 用法: vmlm-runtime <model.vmb> [reps[:dumpSlot]]
pub unsafe fn run() -> u32 {
    let mut path = [0u8; 512];
    arg(1, path.as_mut_ptr());
    if path[0] == 0 {
        vmlm_core::print(b"usage: vmlm-runtime <model.vmb> [reps[:dumpSlot]]\n");
        return 1;
    }
    let mut i = 0;
    while i < 511 && path[i] != 0 {
        i += 1;
    }
    path[i] = 0;

    let mut reps_buf = [0u8; 32];
    arg(2, reps_buf.as_mut_ptr());

    // "N:slot" 语法：N 遍计时，并把 slot 内容 dump 到 out.bin（数值校验）
    let mut dump_slot: Option<usize> = None;
    let reps = if reps_buf[0] == 0 {
        1
    } else if let Some(c) = find_byte(&reps_buf, b':') {
        dump_slot = Some(parse_u32(&reps_buf[c + 1..]) as usize);
        parse_u32(&reps_buf)
    } else {
        parse_u32(&reps_buf)
    };

    let (buf, _len) = vmlm_core::read_file(path.as_ptr());
    let mut vm = Vm::load(buf);

    if reps <= 1 {
        let r = vm.run();
        if let Some(si) = dump_slot {
            let (rr, cc) = vm.slot_shape(si);
            vmlm_core::dump_f32(b"out.bin\0".as_ptr(), vm.slot_mut(si), rr * cc);
        }
        return r;
    }

    vm.run(); // 预热 + 正确性
    if let Some(si) = dump_slot {
        let (rr, cc) = vm.slot_shape(si);
        vmlm_core::dump_f32(b"out.bin\0".as_ptr(), vm.slot_mut(si), rr * cc);
    }

    let mut f = 0i64;
    QueryPerformanceFrequency(&mut f);
    let mut t0 = 0i64;
    let mut t1 = 0i64;
    QueryPerformanceCounter(&mut t0);
    for _ in 0..reps {
        vm.run();
    }
    QueryPerformanceCounter(&mut t1);

    let us = (t1 - t0) as f64 * 1e6 / f as f64 / reps as f64;
    vmlm_core::print(b"avg us/iter = ");
    print_u64(us as u64);
    vmlm_core::print(b"\n");
    0
}
