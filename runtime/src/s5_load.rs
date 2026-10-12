// S5-W2 装载期入口：`neuron_kv` 的真实调用者（引擎侧）。
//
// ============================ 这个文件只做装载 ============================
//
// 本文件是**引擎的一部分**，受体积预算约束。所以它只包含真实的装载路径：
//
//     找 section -> 校验 -> 推紧凑区几何 -> 分配 [n_neurons][stride]
//       -> 重排 -> 自证非空 -> 暴露寻址
//
// **演示/探针脚手架不在引擎里**：极性演示、内存自造 artifact、argv 扫描器、
// 十六进制打印都已移出（见 tools/p15_s5_load_probe.py 的 A/B/C 字节归因）。
// 引擎侧保留**每一条**硬失败守卫及其**各自的具体消息**，以及十进制打印
// `print_num`（守卫要报出「声明 vs 需要」这类数字，那属于失败路径）。
//
// ============================ 入口怎么被选中 ============================
//
// 由**固定的文件名** `s5_kv.vmb` 触发，只比较 argv[1] 是否恰为
// `--s5-load`（见 `wants_load`）。不做 `=path` 解析：那需要一整个 argv
// 扫描器，而本入口只需要「证明 S5 接上了」。这**没有**牺牲任何守卫。

use vmlm_core::neuron_kv::{self, NeuronId, NeuronKvLayout, NkvLog, LOG_LOAD_ENTER};
use vmlm_core::{alloc, arg, exit, print, read_file};

/// 装载的逐位日志（`load_validate` 写入）。
static mut LOG: NkvLog = NkvLog { slots: [0.0; 12] };

/// 装载的固定输入文件名（相对 CWD）。
pub const DEFAULT_PATH: &[u8] = b"s5_kv.vmb\0";

/// `vmlm_core::arg()` 需要的可写缓冲；用具名静态槽让位置固定。
static mut ARG_BUF: [u8; 32] = [0u8; 32];

/// 唯一一条 argv 逻辑：argv[1] 是否恰为 `--s5-load`。
///
/// 认错的代价：写成 `--s5-load=/path` 时这里返回 false，程序走默认推理
/// 路径（去读 model.vmb）。那是**可见的**失败（报 model.vmb 打不开），
/// 不是静默的数据损坏。
pub unsafe fn wants_load() -> bool {
    let buf = core::ptr::addr_of_mut!(ARG_BUF) as *mut u8;
    arg(1, buf);
    const FLAG: &[u8] = b"--s5-load";
    let mut i = 0usize;
    while i < FLAG.len() {
        if *buf.add(i) != FLAG[i] {
            return false;
        }
        i += 1;
    }
    *buf.add(FLAG.len()) == 0
}

/// 装载产物。
pub struct NkvBlock {
    /// `[n_neurons][stride]` 行主序，已重排并清零。
    pub base: *mut f32,
    pub layout: NeuronKvLayout,
    /// 紧凑区在文件里的字节偏移。
    pub compact_off: usize,
    /// 从**文件长度**推出的位置数（必须 > 0）。
    pub n_pos: u32,
    pub file: *const u8,
    pub file_len: usize,
}

/// 无 CRT 的十进制打印。保留在引擎侧的理由见文件头。
unsafe fn print_num(mut v: usize) {
    let mut buf = [0u8; 24];
    let mut i = buf.len();
    if v == 0 {
        i -= 1;
        buf[i] = b'0';
    }
    while v > 0 {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
    }
    print(&buf[i..]);
}

/// 装载期真实路径：读文件 -> 校验 -> 推紧凑区几何 -> 分配 -> 重排。
///
/// 所有守卫都在 core 里；本函数只负责按正确顺序串起来，不重复实现校验。
pub unsafe fn load_file(path: *const u8) -> NkvBlock {
    let (f, flen) = read_file(path);

    // 1. 校验。buf_elems = 0 表示「只校验、先不分配」：非法输入不该先
    //    吃掉 25 MB 再报错。含 S8 的长度精确匹配与头部/section 一致性。
    let (sec, layout) = neuron_kv::load_validate(f, flen, 0, core::ptr::addr_of_mut!(LOG));

    // 2. 紧凑区几何：n_pos 由**文件**推出，不是硬编码。
    let sec_off = neuron_kv::section_file_off(f);
    let (compact_off, n_pos) = neuron_kv::load_compact_geometry(f, &layout, sec_off, flen);
    let compact = f.add(compact_off) as *const f32;

    // 3. 分配 + 重排（真的搬 n_pos 个位置的数据）。
    let base = neuron_kv::load_reorder(&layout, sec, compact, n_pos);

    NkvBlock {
        base,
        layout,
        compact_off,
        n_pos,
        file: f,
        file_len: flen,
    }
}

/// 自证：重排 -> 读回 -> 与文件里的紧凑区逐位比对，并报告可失败的性质计数。
///
/// ============================ D1：非空 ============================
///
/// 判据是「实际搬运并比对了多少个字节」，不是「代码跑到了结尾」。
/// 三条硬失败：`n_pos == 0`、`compared == 0`、`diff != 0`。
///
/// ============================ D3：方向判据的诚实边界 ============================
///
/// ⚠️ 被替换掉的旧判据（记录在此，因为它的失败方式有教育意义）：
///
///     旧代码：want = compact[p*n + i]；got = per_neuron[i*stride + p]；
///             要求两者相等。注释还声称「反向重排会让它大面积不等」。
///
/// 那两格正是写入语句 `*drow.add(p) = *src.add(p*n+i)` 的两端，所以判据是
/// **把赋值重述成等式** —— 写入在上一句让等式成立，因此它对**任何**输入
/// 都成立。
///
/// 旧注释的「方向敏感」说法是**假的**（实测）：判据从未解引用
/// `compact[i*n_pos+p]`，它描述的那个比较**根本不在代码里**。
/// 这正是本项目最忌讳的「未测量断言」——与之前被纠正的「无 gather」
/// 夸大属于同一类。
///
/// ---- 关键区分（我自己也曾经搞错，用 tools/_d3probe 实测过）----
///
/// 「反向」是**两件不同的事**，混为一谈会得出错误结论：
///   (A) 代码反过来：dst[i*stride+p] = src[i*npos+p]
///       -> 旧判据**能**抓（实测 22/24 格不符）；
///          现役的 xref 判据**也能**抓（两条地址算术分叉）。
///   (B) 文件的数据是转置的：代码不变，compact 按转置约定写。
///       -> 旧判据与 xref 判据**都抓不到**（实测 0 格不符）。
/// (B) 才是验证者测的那个。它从格式上看**完全合法且自洽**，文件里没有
/// 任何冗余能区分两种约定 —— 要发现它需要**本格式不存在**的外部参照。
///
/// ---- 因此这里做什么 ----
///
/// 1) `pos_varying > 0`：紧凑区确实随位置变化。否则往返判据恒真。
/// 2) `nonconstant_rows > 0`：每个神经元的前 n_pos 个槽位**不是**常量，
///    即确实按位置分散地搬了东西（能抓「把同一个值广播进所有槽位」、
///    漏搬、以及数据退化）。这两条都能失败。
/// 3) `xref_ok == n*npos`：逐格交叉核对**跑满了**（覆盖计数）。
///    ⚠️ 注意 `xref_bad` **恒为 0**，它不是判别式 —— 详见函数体内那段
///    被独立验证者（D5）更正的注释。这里只把「跑满」当证据。
///    1)、2)、3) 合起来都**不能**发现内容损坏，也不能发现方向写反。
///
/// ---- UNVERIFIED（如实记录，不包装）----
///
/// 「文件内容被改坏、但结构依然自洽」**当前检测不到**。实测：把紧凑区
/// 5 个 f32 改掉，本函数全部判据照常通过、exit=0。原因是所有判据的两端
/// 都来自同一个文件，没有任何独立信息。该能力需要一个本格式不存在的
/// 外部参照（校验和 / 期望内容 / 独立第二份副本），**记为 UNVERIFIED**。
/// 不为其硬造一个假能过的检查。
///
/// ---- 诚实结论（写死，不要含糊）----
///
/// **引擎侧无法用一条判据证明方向正确。** 转置的文件是一个合法的
/// [n_neurons][stride] 缓冲区，在同一条代码路径下与正确排布同样自洽。
/// 任何声称「引擎自检证明了方向」的说法都是**夸大**。
///
/// **方向由测试侧用独立实现核对**：tools/p15_s5_load_probe.py 的
/// REVERSED 用例在**引擎之外**用 Python 造转置文件并比对期望排布 ——
/// 那才是对方向的真判据，因为它的期望不来自引擎。这与验证者记录的限制
/// 是同一条（reversed artifact 不是格式上可检测的非法输入）。
pub unsafe fn verify_bitexact(block: &NkvBlock) -> usize {
    let layout = &block.layout;
    let n = layout.n_neurons();
    let npos = block.n_pos as usize;
    let stride = layout.stride as usize;
    let compact = block.file.add(block.compact_off) as *const f32;
    let per_neuron = block.base;

    if npos == 0 {
        // **纯索引 artifact**（真实生产 `.vmb` 属于这一类）：section 是
        // 元数据，文件里**没有**紧跟其后的紧凑 payload。这不是「空操作
        // 却报成功」——它是**另一种 artifact**，本条路径只装载布局与索引，
        // 不搬运任何元素。
        //
        // ⚠️ 与 D1 的区别必须说清楚：D1 是「有 payload 却搬 0 个」，
        // 那是缺陷；这里是「本来就没有 payload」。为了不让两者混淆，
        // 本路径**不打印** "S5LOAD ok"，而是打印一条**明确说明零元素被搬**
        // 的独立行，并且**不**声称完成过重排。
        print(b"S5LOAD index-only (no compact payload; zero elements moved)\n");
        return 0;
    }

    // 读回：per-neuron -> compact，回到文件里紧凑区的排布。
    let back = alloc(npos * n * 4) as *mut f32;
    core::ptr::write_bytes(back as *mut u8, 0, npos * n * 4);
    neuron_kv::reorder_neurons_to_compact(layout, per_neuron, back, block.n_pos);

    let compared = npos * n * 4;
    let diff = neuron_kv::bitwise_equal(compact, back, npos * n);

    // 可失败性质 1：紧凑区随位置变化（否则往返判据恒真）。
    //
    // ⚠️ 必须用 `.to_bits()` 取位型，**不能**写 `as u32`：`as u32` 是
    // **数值截断**（0.0 -> 0、2.5 -> 2、-1.0 -> 0），拿它去和 `to_bits()`
    // 比是拿两种不同的表示对比 —— 实测症状是本判据在**完全正确**的
    // 重排上误报。这正是「按位型比，不按值比」那条规矩的又一实例。
    let mut pos_varying = 0usize;
    // 可失败性质 2：神经元行内非常量（确实按位置分散搬运）。
    let mut nonconstant_rows = 0usize;
    // 可失败性质 3：重排确实把 compact[0*n+i] 放进了神经元 i 的槽位 0。
    let mut slot0_ok = 0usize;

    // ---- 交叉核对：一条**独立地址算术**的对照（Lead 加入，实测结论见下）----
    //
    // ⚠️ 上面所有判据加在一起**仍然抓不到文件被改坏** —— 两个独立调查者
    // 各自测出了同一件事，本函数第一版就是这么错的：
    //   * `diff`（往返）比较的是
    //     `reorder_neurons_to_compact(reorder_compact_to_neurons(X))` 与 `X`。
    //     两个重排函数是**严格互逆**的，所以对**任意** X 都恒等 ——
    //     `diff==0` 是构造出来的，不是测出来的。它验证的是「两个重排函数
    //     互逆」，**不是**「文件读对了」。
    //   * `slot0_ok` 比较 `per_neuron[i*stride]` 与 `compact[i]`，而重排
    //     正是**刚刚**把前者写成后者的 —— 同义反复，比较的是自己写入的格。
    //   * `pos_varying` / `nonconstant_rows` 是**数据的形状**性质，
    //     不是「读对了」的性质。
    //   ⇒ 实测（Lead）：把紧凑区 5 个 f32 改掉，以上全部照常通过、exit=0。
    //
    // ---- 这条 xref 判据的**真实**判别力（不许夸大）----
    //
    // ⚠️ 独立验证者（task-4，D5）实测推翻了本段**第一版**的说法，已更正：
    //   第一版声称它「能抓 stride 写错」和「能抓方向写反」。**两条都是假的**：
    //     * 它比较的 `compact[xp*n+xi]` 与 `per[xi*stride+xp]`，正是重排
    //       自己那条赋值语句的两端 —— 同义反复。实测：20 万组随机 compact，
    //       `xref_bad` 恒为 0；引擎上「正确」与「反序」两种输入都得
    //       `xref_bad=0`、exit=0。
    //     * stride 也不可能被抓：重排与核对**读的是同一个** `layout.stride`，
    //       两者共享同一个错值（即 S5-A/§4.4 的自洽陷阱）。
    //   所以 `xref_bad` **恒等于 0**，它不是判别式，只是把赋值重读了一遍。
    //   保留它的唯一价值是**覆盖计数**（`xref_ok == n*npos`）：它证明这段
    //   逐格扫描真的跑满了，而不是空测量。这一点也必须说清楚，不能当成
    //   「方向已验证」。
    //
    // 它同样**不能**发现内容损坏。实测（Lead）：改坏 5 个 f32 后仍 exit=0。
    //
    // ⚠️ 要发现「文件内容被改坏但结构自洽」或「方向写反」，需要一个
    //    **本格式不存在**的外部参照（校验和 / 期望内容 / 独立的第二份副本）：
    //    对同一 artifact 而言，正确与反序两份数据的**长度相同、差异全在
    //    compact 区内**，任何读者都无法仅凭文件把它们分辨开（验证者的 C2）。
    //    **该能力不存在**，在报告里如实记为 UNVERIFIED，绝不为它硬造一个
    //    假能过的检查 —— 那正是本项目最忌讳的形态。
    //
    // 覆盖计数必须恰好 n*npos：少扫一格就是「空测量」，必须自己说出来。
    let mut xref_ok = 0usize;
    let mut xref_bad = 0usize;
    {
        let mut xi = 0usize;
        while xi < n {
            let base_elem = xi * stride;
            let mut xp = 0usize;
            while xp < npos {
                let expect = (*compact.add(xp * n + xi)).to_bits();
                let got = (*per_neuron.add(base_elem + xp)).to_bits();
                if expect == got {
                    xref_ok += 1;
                } else {
                    xref_bad += 1;
                }
                xp += 1;
            }
            xi += 1;
        }
    }

    let mut i = 0usize;
    while i < n {
        let a = (*compact.add(i)).to_bits();
        let b = (*compact.add(n + i)).to_bits();
        if a != b {
            pos_varying += 1;
        }
        if (*per_neuron.add(i * stride)).to_bits() == a {
            slot0_ok += 1;
        }
        let first = (*per_neuron.add(i * stride)).to_bits();
        let mut p = 1usize;
        while p < npos {
            if (*per_neuron.add(i * stride + p)).to_bits() != first {
                nonconstant_rows += 1;
                break;
            }
            p += 1;
        }
        i += 1;
    }

    // ---- 证据行。字段名刻意把「验证的是什么」写在名字里，避免被误读 ----
    //
    // relabel（任务 2）：`inverse_ok` 与 `slot0_ok` 都**不是**内容判据，
    // 名字里直接说明它们验证的是「两个重排函数互逆」和「自己刚写的格」。
    print(b"S5LOAD file compact_off=");
    print_num(block.compact_off);
    print(b" n_pos=");
    print_num(npos);
    print(b" inverse_bytes_compared=");
    print_num(compared);
    print(b" inverse_first_differing_byte=");
    print_num(diff);
    print(b" pos_varying_neurons=");
    print_num(pos_varying);
    print(b" nonconstant_rows=");
    print_num(nonconstant_rows);
    print(b" slot0_ok_selfconsistent=");
    print_num(slot0_ok);
    print(b" xref_ok=");
    print_num(xref_ok);
    print(b" xref_bad_tautological_always0=");
    print_num(xref_bad);
    print(b"\n");

    if compared == 0 {
        print(b"NKV: nothing was compared\n");
        exit(9);
    }
    // ⚠️ 往返判据（inverse pair）：验证 `reorder_neurons_to_compact` 与
    // `reorder_compact_to_neurons` **互逆**，**不是**「文件读对了」。
    // 两个函数严格互逆 ⇒ 该判据对任意输入恒成立，无法发现内容损坏。
    // 保留它是因为它仍能抓「两个函数之一被改坏导致不再互逆」。
    if diff != 0 {
        print(b"NKV: reorder and its inverse are not inverses (round trip)\n");
        exit(9);
    }
    if pos_varying == 0 {
        // 判据恒真 = 没有证据。必须说出来，不能当通过。
        print(b"NKV: round trip is vacuous (positions are indistinguishable)\n");
        exit(9);
    }
    if nonconstant_rows == 0 {
        // 所有神经元的槽位都相同 -> 无法区分「按位置分散搬运」与
        // 「把同一个值广播进所有槽位」。
        print(b"NKV: reorder moved no per-position variation (constant rows)\n");
        exit(9);
    }
    // ⚠️ slot0_ok 比较的是重排**刚刚写入**的格，属自洽判据，不是内容判据。
    if slot0_ok != n {
        print(b"NKV: slot 0 of some neuron does not hold position 0\n");
        exit(9);
    }
    // 交叉核对。
    // ⚠️ 诚实说明（验证者 D5 更正）：`xref_bad` **恒为 0**，因为本判据比较的
    // 两端就是重排那条赋值语句的两端 —— 同义反复，**抓不到** stride 写错
    // （重排与核对共享同一个 layout.stride），也**抓不到**方向写反。
    // 保留此分支只作防御性兜底；真正的证据是下面的**覆盖计数**。
    // **不能**发现内容损坏（实测：改坏 5 个 f32 仍 exit=0）。
    if xref_bad != 0 {
        print(b"NKV: reorder disagrees with the independent index prediction\n");
        exit(9);
    }
    if xref_ok != n * npos {
        // 覆盖不足 = 空测量。不许静默通过。
        print(b"NKV: cross-reference covered fewer cells than expected\n");
        exit(9);
    }
    compared
}

/// 装载 + 自证，打印证据行后返回 0。
///
/// "S5LOAD ok" 只在 verify_bitexact 通过之后才打印 —— 一个搬了 0 个元素
/// 或数据退化的运行必然已经 `exit(9)`，所以「ok」不可能出现在空操作上。
pub unsafe fn run_file(path: *const u8) -> u32 {
    let block = load_file(path);

    print(b"S5LOAD file n_neurons=");
    print_num(block.layout.n_neurons());
    print(b" n_layers=");
    print_num(block.layout.n_layers as usize);
    print(b" n_kv_heads=");
    print_num(block.layout.n_kv_heads as usize);
    print(b" head_dim=");
    print_num(block.layout.head_dim as usize);
    print(b" max_seq_per_neuron=");
    print_num(block.layout.max_seq_per_neuron as usize);
    print(b" stride=");
    print_num(block.layout.stride as usize);
    print(b" file_bytes=");
    print_num(block.file_len);
    print(b"\n");

    let ev = block.layout.n_neurons() * block.layout.stride as usize;
    print(b"S5LOAD file alloc_elems=");
    print_num(ev);
    print(b" alloc_bytes=");
    print_num(ev * 4);
    print(b"\n");

    let moved = verify_bitexact(&block);

    // 寻址式真的用一次：addr = base + idx*stride + slot。
    // (0,0,0) 号神经元的槽位 0 偏移必须是 0。
    let id = NeuronId {
        layer: 0,
        kv_head: 0,
        dim: 0,
    };
    let addr = block.layout.slot_addr(id, 0);
    if addr != 0 {
        print(b"NKV: addressing check failed\n");
        exit(9);
    }
    print(b"S5LOAD addr_check slot_addr=");
    print_num(addr);
    print(b"\n");

    // 只有**真的搬过元素**时才核对槽位内容 —— 纯索引 artifact 没有
    // payload，拿它比会把「本来没有 payload」误报成「读错元素」。
    if moved != 0 {
        let v0 = *neuron_kv::nkv_slot_ptr(&block.layout, block.base, id, 0);
        let want0 = *(block.file.add(block.compact_off) as *const f32);
        print(b"S5LOAD addr_check value_bits_ok=");
        print_num(if v0.to_bits() == want0.to_bits() { 1 } else { 0 });
        print(b"\n");
        if v0.to_bits() != want0.to_bits() {
            print(b"NKV: addressing reads the wrong element\n");
            exit(9);
        }
    }

    if neuron_kv::log_get(core::ptr::addr_of!(LOG), 0) != LOG_LOAD_ENTER {
        print(b"NKV: load entered flag missing\n");
        exit(9);
    }
    // ⚠️ 纯索引 artifact **不打印** "S5LOAD ok"：本行是「重排完成」的
    // 成功标志，而那条路径一个元素都没搬。混淆两者正是 D1 的形态。
    if moved == 0 {
        return 0;
    }
    print(b"S5LOAD ok\n");
    0
}
