// 共享的命令行参数解析（no_std 无 CRT，没有 argv）
//
// 为什么需要单独一个模块：
//   `bench` feature 里原本有一份私有的 `arg()`；AOT 的多模型切换同样需要
//   「按 `--model=<name>` 选模型」。与其复制两份（迟早会漂移），这里抽成
//   一处，`bench` 与 `aot` 共用。
//
// 代价说明（体积）：
//   `vmlm_core::arg()` 已经存在（core/src/lib.rs），但引擎核心本体不调用它
//   —— 只有本模块用。把本模块放在 runtime 侧而不是 core 侧，是为了让
//   **默认推理构建完全不编译它**（`#[cfg(any(feature="bench", feature="aot"))]`），
//   从而不挤压 48KB 体积预算。
//
// 无 CRT 约束：只用 kernel32 的 GetCommandLineA，不碰 libc。

#[link(name = "kernel32")]
extern "system" {
    fn GetCommandLineA() -> *const u8;
}

/// 取第 n 个命令行参数（n=0 是程序名），写成 NUL 结尾到 `dst`。
/// `dst` 必须至少有 512 字节。找不到参数时写空串。
///
/// # Safety
/// `dst` 必须指向至少 512 个可写字节。
pub unsafe fn arg(n: usize, dst: *mut u8) {
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
                copy_out(cl.add(s), dst, i - s);
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
                copy_out(cl.add(start), dst, i - start);
                return;
            }
        }
        idx += 1;
    }
}

unsafe fn copy_out(src: *const u8, dst: *mut u8, n: usize) {
    let mut k = 0;
    while k < n {
        *dst.add(k) = *src.add(k);
        k += 1;
    }
    *dst.add(k) = 0;
}

/// 算 NUL 结尾串的长度。
///
/// 注意（**实测踩到的坑**）：不要写成标准的
/// `let mut n = 0; while *p.add(n) != 0 { n += 1 } n` 循环 —— LLVM 会把那个
/// 形状（哪怕加了循环携带的累加状态、哪怕函数标了 #[inline(never)]）
/// 识别成 `strlen` 惯用法，直接生成对 libc `strlen` 的调用；本引擎是
/// `/NODEFAULTLIB` 无 CRT，于是 LNK2019。
/// 同族的坑：切片相等 `&a[..2] == b"--"` 会生成 `memcmp`（core 只提供了
/// memcpy/memset/memmove，见 core/src/lib.rs）。两者都已实测触发过。
///
/// 规避办法：**每次迭代用 `read_volatile` 读**。volatile 读不可被合并/向量化，
/// 循环也就无法被降成对 libc 的批量调用 —— 这是唯一实测有效的写法。
///
/// # Safety
/// `p` 必须指向 NUL 结尾的串。
pub unsafe fn scan_nul(p: *const u8) -> usize {
    let mut n = 0usize;
    loop {
        if core::ptr::read_volatile(p.add(n)) == 0 {
            return n;
        }
        n += 1;
    }
}

/// `arg(n)` 的切片化封装：解析进调用方给的缓冲，返回已写入的 `&[u8]`（不含 NUL）。
///
/// # Safety
/// `buf` 必须至少有 512 字节。
pub unsafe fn arg_slice<'a>(n: usize, buf: &'a mut [u8; 512]) -> &'a [u8] {
    arg(n, buf.as_mut_ptr());
    let l = scan_nul(buf.as_ptr());
    &buf[..l]
}

/// 在整个命令行里找 `--key=value` 或 `--key value`，返回 value 的切片。
///
/// 支持两种写法是因为 Windows 命令行习惯两者都有；`--model=attn` 更省一次
/// 解析，`--model attn` 更符合直觉。两者都收。
///
/// 实现注意（踩过的坑）：**不要用 `&a[..2] == b"--"` 这类切片比较**。
/// 那会生成对 `memcmp` 的调用，而本引擎是 `/NODEFAULTLIB` 的 no_std 无 CRT，
/// core 只提供了 memcpy/memset/memmove（见 core/src/lib.rs），没有 memcmp /
/// strlen —— 实测直接 LNK2019。这里全部改成逐字节手写比较。
///
/// # Safety
/// 只读 kernel32 的命令行缓冲，无额外前提。
pub unsafe fn find_flag<'a>(key: &[u8], buf: &'a mut [u8; 512]) -> Option<&'a [u8]> {
    // 逐个参数扫（跳过 argv[0]），避免自己处理引号/转义。
    //
    // 借用注意：不能在下一次 `arg_slice(.., buf)`（可变借用）还活着时
    // 持有上一步返回的切片。所以每一轮只做**判定**（返回偏移量），
    // 真正构造返回切片只在命中那一次做。
    let mut n = 1usize;
    loop {
        let mut cur = [0u8; 512];
        let a = arg_slice(n, &mut cur);
        if a.is_empty() {
            return None;
        }
        // 只支持 `--key=value`。裸开关 `--key` 用 has_flag() 查 ——
        // 早先这里还想兼容 `--key value`，结果 `--list` 会被走成
        // 「取下一个参数当值」，而下一个参数不存在 → 误返回 None（实测踩过）。
        if let Some(off) = eq_form_off(a, key) {
            let l = a.len() - off;
            let mut i = 0;
            while i < l {
                buf[i] = cur[off + i];
                i += 1;
            }
            return Some(core::slice::from_raw_parts(buf.as_ptr(), l));
        }
        n += 1;
    }
}

/// 是否存在裸开关 `--key`（无值）。用于 `--list` 这类纯开关。
///
/// 单独提供是因为 `find_flag` 返回 0 长度切片时，调用方无法区分
/// 「开关存在但无值」与「开关不存在」。这里给一个明确的布尔查询。
pub unsafe fn has_flag(key: &[u8]) -> bool {
    let mut cur = [0u8; 512];
    let mut n = 1usize;
    loop {
        let a = arg_slice(n, &mut cur);
        if a.is_empty() {
            return false;
        }
        if has_flag_form(a, key) {
            return true;
        }
        n += 1;
    }
}

/// `a` 形如 `--key=value` 时返回 value 在 `a` 中的起始偏移。
/// 手写前缀比较，不引入 memcmp。
fn eq_form_off(a: &[u8], key: &[u8]) -> Option<usize> {
    // 至少 "--" + key + "="
    if a.len() < key.len() + 3 {
        return None;
    }
    if a[0] != b'-' || a[1] != b'-' {
        return None;
    }
    let mut i = 0;
    while i < key.len() {
        if a[2 + i] != key[i] {
            return None;
        }
        i += 1;
    }
    if a[2 + key.len()] != b'=' {
        return None;
    }
    Some(3 + key.len())
}

/// `a` 恰好形如 `--key` 时返回 true。手写比较，不引入 memcmp。
fn has_flag_form(a: &[u8], key: &[u8]) -> bool {
    if a.len() != key.len() + 2 {
        return false;
    }
    if a[0] != b'-' || a[1] != b'-' {
        return false;
    }
    let mut i = 0;
    while i < key.len() {
        if a[2 + i] != key[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// 字节切片相等（手写，不引 memcmp —— 见 find_flag 的说明）。
pub fn eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}
