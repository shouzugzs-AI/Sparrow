// PATH: B1
// 最小转译器：B1 稀疏前向的**单神经元单行** x86-64 内联机器码生成。
//
// =========================== 这个模块是什么 ===========================
//
// 它是 core/src/conn_train.rs 里 `sparse_accum`（第 700-726 行）那个标量
// 循环的**逐行展开版**。一个神经元 i 编译成一段独立可调用的机器码：
//
//     (每次运行开头)  mov dword [acc], 0
//     ---- 对 j = 0..k0 逐条展开，无循环 ----
//        movss  xmm1, [act + 4*j]      ; 这条连接激活吗 (1.0 / 0.0)
//                                      ; ⚠️ 必须是 movss + 4*j，不是 movsd+8*j
//                                      ;    （踩过：读成了第 2j 个元素）
//        comisd xmm1, xmm2             ; xmm2 == +0.0
//        jbe    <跳过本连接>             ; == 0.0（或 NaN）-> 跳过
//        movss  xmm1, [src + 4*j]      ; src id（**u32-as-f32 的浮点数值**）
//        cvttss2si rax, xmm1           ; f32 -> 整数**转换**（不是按位搬！）
//        movss  xmm0, [r11 + rax*4]    ; load x[src]  <- 真·间接寻址
//        mulss  xmm0, [w + 4*j]        ; * w[j]（权重地址烘入 disp32）
//        addss  xmm0, [rcx]            ; += acc；中间值始终在寄存器
//        movss  [rcx], xmm0
//     <跳过>:
//     ...（k0 条重复）
//        movss  xmm0, [rcx]            ; 读回累加器
//        movss  [y_i], xmm0            ; 写回 y[i]
//        ret
//
// ⚠️ src 的转换必须用 cvttss2si。src 存的是「u32-as-f32」的**浮点数值**
//   （如 63.0），不是索引的整数位型（63.0f32 的位型是 0x427C0000）。
//   引擎的做法是 `*srcrow.add(j) as usize`，AVX2 路径是
//   `_mm256_cvttps_epi32`。本转译器最初写成 `mov eax,[src]`（按位搬），
//   实测立刻访问违例（把 0x427C0000 当索引）。这条注释是那次实测的记录。
//
// ---- 任务书四条要求逐条对照 ----
//   * 无算子分发：没有指令巨 switch、没有内核函数表。一行就是一段直线代码。
//   * 无函数调用：生成的代码里**一条 call 都没有**（只有结尾 ret）；
//     而且内核不是"每个连接 call 一次"—— 整行 k0 条被展开进一段直线代码。
//   * 中间值在寄存器：累加器全程在 xmm0（写回时从槽位读回，见下）；
//     因子在 xmm0/xmm1；比较常量在 xmm2。
//   * 权重地址烘入：w / src / act / x / y_i / acc 的绝对地址都编成
//     mov r64, imm64；src 的偏移逐连接烘入；w[j] 直接折进寻址的 disp32。
//     x[src] 的**源地址不能烘入**（src id 是数据，这就是"稀疏"本身），
//     但它也不需要 call：由本行自己的 3 条地址计算指令算出。
//
// =========================== 这个模块不是什么 ===========================
//
// **不是 core/src/jit.rs 的扩展，没有复用它的 emit 架构。**
// 依据 ref_sparse_a/AUDIT_PATH_ISOLATION.md §7：
//   - jit.rs 的 emit() **没有** OP_TOPK_BY_STRENGTH / OP_SPARSE_MATVEC 臂，
//     未实现的 opcode 走 `_ =>` 在 jit.rs:297 硬失败（exit_i(6)）。
//   - jit.rs:8-9 自述的设计意图与本模块**相反**：
//     「JIT 生成『胶水』而不是『数学』……call 预编译的 AVX2 内核」。
//   本模块只复用了 jit.rs 的**编码原语字节形式**（mov_imm64 / sub_rsp）
//   与它的 RWX 两段式分配用法（jit.rs:36-52）。没有复用任何 emit 架构。
//   jit.rs 本身**未做任何修改**。
//
// =========================== 一处曾经写错、现已被实测纠正的语义 ===========================
//
// 【原以为的边界 1 —— 已被实测推翻，保留记录以免后人重犯】
//   我最初写：「算子版用 `act == 0.0`，NaN == 0.0 为 false ⇒ 会乘；
//   转译版用 comisd + jbe，jbe 在 unordered(NaN) 时为真 ⇒ 会跳过。
//   故 NaN 掩码上两版不等价。」
//
//   **这个推理是错的。** 实测（vmlm-trans-iso 的分支探针，直接执行生成的
//   比较+分支序列）：
//       act = QNaN(0x7FC00000) -> JBE **不跳**（= 照乘，与算子一致）
//       act = +0.0             -> JBE 跳（= 跳过，与算子一致）
//       act = 1.0              -> JBE 不跳（= 照乘，与算子一致）
//   即：`comisd` + `jbe` 与 `== 0.0` 在 **NaN、±0、正常值** 上三者同向。
//   原因是 `comisd` 是**信号型**比较（QNaN 触发 #IA），在 Windows 的默认
//   FP 环境下标志位不是"经典 unordered=ZF|PF|CF"那么简单；无论机理如何，
//   可观测行为就是上表，而**可观测行为才是契约**。
//
//   ⇒ 结论：在 act ∈ {0.0, 1.0} **以及含 NaN** 的域上，两版**逐位等价**。
//     驱动里的 [3] 判据因此改为"逐条断言两版在同一 NaN 输入上同值"，
//     而不是断言"必须不同"。**这一条是本验证唯一一次"注释里的推理被
//     实测纠正"的记录** —— 保留它，因为本项目的失败模式正是"看起来合理
//     的推理没被跑过"。
//
// 【边界 2】累加器按 +0.0 起加。
//   算子版写 `acc = if first { term } else { acc + term }`，即首个命中**直接
//   赋值**；转译版写 `acc = 0.0; acc = acc + term`。两者逐位相同，因为
//   int_add(+0.0, t) == t 对**所有** f32 位型 t 成立，含 ±0、次正规、NaN
//   （IEEE 规定 +0 + x = x，唯一例外是 x = -0 时得 +0）。
//   例外路径在 act ∈ {0.0,1.0} 下不可达：0.0 的连接已被跳过，故 acc 只在
//   已有命中后才可能为 ±0，此后必然发生真实加法。
//   若整行零命中，两版都保持 +0.0（转译版更确定：显式清零）。
//   ⇒ 在 act ∈ {0.0, 1.0} 的输入域上两版**逐位等价**（已由 [1] mismatch=0 实测）。
//
// =========================== 关于 k0 的历史包袱 ===========================
//
// 本模块**刻意不**链接 core::conn_train。理由：那份 conn_train.rs 里存在
// 「B1」被用作**配置名**的遗留用法，而本任务的路径契约（AUDIT §0）要求
// B1 是**生产路径**名。为让本验证口径独立于那份文件的重命名进度，
// trans_verify 自己构造同布局的缓冲；布局与 SparseConnW **逐字段相同**
// （w / src / act 三段连续 f32，src 元素是 u32-as-f32），并且
// trans_verify **会用 conn_train::sparse_accum 做对拍** —— 所以
// "同布局"这件事是被测过的，不是声称的。
#![allow(dead_code)]

// ============================================================================
// 0. 分配（复刻 core/src/jit.rs:36-52 的两段式 RWX 用法）
// ============================================================================

extern "system" {
    fn VirtualAlloc(a: *mut u8, n: usize, t: u32, p: u32) -> *mut u8;
    fn VirtualProtect(a: *mut u8, n: usize, new: u32, old: *mut u32) -> i32;
    fn GetCurrentProcess() -> *mut u8;
    fn FlushInstructionCache(h: *mut u8, base: *const u8, n: usize) -> i32;
}

/// 先 READWRITE(0x04) 写码，写完再切 EXECUTE_READ(0x20)。
/// 与 jit.rs:36-46 同：**不能**一开始就申请成 exec，那样会丢掉写权限，
/// 后续写码立即访问违例。
pub unsafe fn alloc_rw(n: usize) -> *mut u8 {
    VirtualAlloc(core::ptr::null_mut(), n, 0x3000, 0x04)
}

/// 切到 PAGE_EXECUTE_READ，然后**必须** FlushInstructionCache。
///
/// 为什么这一步不能省（本次实测踩到）：直接写完码就 VirtualProtect 再
/// 跳进去执行，在 Windows 上会出现**访问违例 0xC0000005**——因为 CPU 的
/// 指令缓存里可能仍是该地址的旧内容（此页此前从没被当作代码取过），
/// 而 x86 的指令取指不保证与数据写一致。`FlushInstructionCache` 是
/// 微软文档明确要求的配套动作；`VirtualProtect` 只在**权限变化**时做
/// 隐式刷新，不承担"我刚写进去的字节一定可见"的保证。
/// 这是本项目"看起来正常的代码在运行期崩"的又一例：反汇编完全正确，
/// 崩的是取指一致性。
pub unsafe fn make_exec(p: *mut u8, n: usize) -> bool {
    let mut old = 0u32;
    if VirtualProtect(p, n, 0x20, &mut old) == 0 {
        return false;
    }
    FlushInstructionCache(GetCurrentProcess(), p as *const u8, n) != 0
}

// ============================================================================
// 1. 编码原语（字节形式与 jit.rs 的 mov_imm64 / sub_rsp 一致）
// ============================================================================

pub const RAX: u8 = 0;
pub const RCX: u8 = 1;
pub const R8: u8 = 8;
pub const R9: u8 = 9;
pub const R10: u8 = 10;
pub const R11: u8 = 11;

pub struct Emitter {
    pub p: *mut u8,
    pub len: usize,
    pub cap: usize,
}

impl Emitter {
    pub unsafe fn new(p: *mut u8, cap: usize) -> Emitter {
        Emitter { p, len: 0, cap }
    }
    #[inline]
    pub unsafe fn b(&mut self, v: u8) {
        assert!(self.len < self.cap, "trans: emit overflow");
        *self.p.add(self.len) = v;
        self.len += 1;
    }
    #[inline]
    pub unsafe fn d32(&mut self, v: i32) {
        assert!(self.len + 4 <= self.cap, "trans: emit overflow");
        core::ptr::write_unaligned(self.p.add(self.len) as *mut i32, v);
        self.len += 4;
    }
    #[inline]
    pub unsafe fn d64(&mut self, v: u64) {
        assert!(self.len + 8 <= self.cap, "trans: emit overflow");
        core::ptr::write_unaligned(self.p.add(self.len) as *mut u64, v);
        self.len += 8;
    }
    /// 当前偏移（供回填分支位移）
    #[inline]
    pub fn here(&self) -> usize {
        self.len
    }

    // ---- 前缀 / 修饰字节 ----
    #[inline]
    unsafe fn rex(&mut self, w: u8, reg: u8, index: u8, base: u8) {
        let r = 0x40 | (w << 3) | ((reg >> 3) << 2) | ((index >> 3) << 1) | (base >> 3);
        if r != 0x40 {
            self.b(r);
        }
    }
    /// mod=10 + disp32 的 ModRM；base 低 3 位为 100(rsp/r12) 时补 SIB
    #[inline]
    unsafe fn modrm_d32(&mut self, reg: u8, base: u8) {
        self.b(0x80 | ((reg & 7) << 3) | (base & 7));
        if base & 7 == 4 {
            self.b(0x24); // SIB: scale=0, index=100(none), base=rsp
        }
    }

    // ---- 通用寄存器 ----
    /// mov r64, imm64   [48|49] [B8+r] imm64  —— 同 jit.rs:86
    pub unsafe fn mov_imm64(&mut self, reg: u8, imm: u64) {
        self.b(0x48 | (reg >> 3));
        self.b(0xB8 + (reg & 7));
        self.d64(imm);
    }
    /// mov r32, [base + disp32]   8B /r  (mod=10)
    /// 用来**按整数**搬 src[j] 的位型 —— 绝不能走浮点。
    pub unsafe fn mov_r32_mem(&mut self, dst: u8, base: u8, disp: i32) {
        self.rex(0, dst, 0, base);
        self.b(0x8B);
        self.modrm_d32(dst, base);
        self.d32(disp);
    }
    /// movsxd r64, r32   48 63 /r  (mod=11)
    pub unsafe fn movsxd(&mut self, dst: u8, src: u8) {
        self.b(0x48);
        self.rex(0, dst, 0, src);
        self.b(0x63);
        self.b(0xC0 | ((dst & 7) << 3) | (src & 7));
    }
    /// mov dword [base + disp32], imm32   C7 /0  (mod=10)
    pub unsafe fn mov_mem_imm32(&mut self, base: u8, disp: i32, imm: i32) {
        self.rex(0, 0, 0, base);
        self.b(0xC7);
        self.modrm_d32(0, base);
        self.d32(disp);
        self.d32(imm);
    }
    /// sub rsp, imm —— 字节形式同 jit.rs:92
    pub unsafe fn sub_rsp(&mut self, n: u32) {
        if n <= 127 {
            self.b(0x48);
            self.b(0x83);
            self.b(0xEC);
            self.b(n as u8);
        } else {
            self.b(0x48);
            self.b(0x81);
            self.b(0xEC);
            self.d32(n as i32);
        }
    }
    /// add rsp, imm —— 字节形式同 jit.rs:105
    pub unsafe fn add_rsp(&mut self, n: u32) {
        if n <= 127 {
            self.b(0x48);
            self.b(0x83);
            self.b(0xC4);
            self.b(n as u8);
        } else {
            self.b(0x48);
            self.b(0x81);
            self.b(0xC4);
            self.d32(n as i32);
        }
    }
    pub unsafe fn ret(&mut self) {
        self.b(0xC3);
    }

    // ---- SSE ----
    /// pxor xmm2, xmm2  (66 0F EF D2) —— 得到 +0.0 比较常量
    pub unsafe fn pxor_xmm2_self(&mut self) {
        self.b(0x66);
        self.b(0x0F);
        self.b(0xEF);
        self.b(0xD2);
    }
    /// cvttss2si r64, xmm   F3 [48] 0F 2C /r   (mod=11)
    ///
    /// 这是**必须**用的一条，理由是本次实测踩到的真 bug：
    /// `src` 数组存的是「u32-as-f32」的**浮点数值**（如 63.0），
    /// 不是索引的整数位型（63.0f32 的位型是 0x427C0000 = 1111482368）。
    /// 引擎自己的做法是 `*srcrow.add(j) as usize`（f32 -> 整数**转换**），
    /// AVX2 路径同样是 `_mm256_cvttps_epi32`。用 `mov eax,[..]` 去做
    /// "按位搬"会把 0x427C0000 当成索引 -> x[1111482368] -> 访问违例。
    /// cvttss2si 的截断语义与 Rust 的 `as usize`（向零取整）一致。
    /// ⚠️ 与 Rust `as` 的差异：NaN / 溢出时 cvttss2si 返回 0x8000_0000，
    /// 而 Rust 饱和。合法 src ∈ [0, n_neurons) 时两者相同。
    pub unsafe fn cvttss2si_r64(&mut self, dst: u8, xmm: u8) {
        self.b(0xF3);
        self.b(0x48 | (dst >> 3));
        self.b(0x0F);
        self.b(0x2C);
        self.b(0xC0 | ((dst & 7) << 3) | (xmm & 7));
    }
    /// movss xmm, [base + disp32]   F3 0F 10 /r  (mod=10)
    ///
    /// ⚠️ 读 act[j] 必须用 **movss（32 位）**，偏移是 **4*j**。
    /// 本转译器最初用的是 `movsd [act + 8*j]`（64 位载入），那是错的：
    /// act_row 的元素是 f32，act[j] 在字节偏移 4*j；写 8*j 实际读到的是
    /// **第 2j 个**元素（并且 j 大时越界到下一行）。因为 act ∈ {0,1}，
    /// 低 32 位恒为 0，于是"读到的 double"仍然与某个 0/1 值比较得上，
    /// **数值上看不出崩**，但选中的是错的连接 —— 实测表现为
    /// mismatch 1536/1536 而进程不崩。这类"看起来合理的垃圾数"正是
    /// 本项目要消灭的失败模式，所以这里用 movss 并把 4*j 写死。
    pub unsafe fn movss_load(&mut self, xmm: u8, base: u8, disp: i32) {
        self.b(0xF3);
        self.rex(0, xmm, 0, base);
        self.b(0x0F);
        self.b(0x10);
        self.modrm_d32(xmm, base);
        self.d32(disp);
    }
    /// movss xmm, [base + index*4]  F3 0F 10 /r  (mod=00 + SIB, scale=2)
    /// 这就是 `x[src]` 的那一次真·间接取数。
    /// ⚠️ index 不能是 rsp(4)，否则 SIB 里 index=100 表示"无 index"。
    /// 这里 index 恒为 rax(0)，安全。
    pub unsafe fn movss_load_idx4(&mut self, xmm: u8, base: u8, index: u8) {
        self.b(0xF3);
        self.rex(0, xmm, index, base);
        self.b(0x0F);
        self.b(0x10);
        self.b(((xmm & 7) << 3) | 0x04); // ModRM rm=100 -> 有 SIB
        self.b((2 << 6) | ((index & 7) << 3) | (base & 7)); // scale=4
    }
    /// movss [base + disp32], xmm   F3 0F 11 /r  (mod=10)
    pub unsafe fn movss_store(&mut self, base: u8, disp: i32, xmm: u8) {
        self.b(0xF3);
        self.rex(0, xmm, 0, base);
        self.b(0x0F);
        self.b(0x11);
        self.modrm_d32(xmm, base);
        self.d32(disp);
    }
    /// mulss xmm, [base + disp32]   F3 0F 59 /r  (mod=10)
    pub unsafe fn mulss_mem(&mut self, xmm: u8, base: u8, disp: i32) {
        self.b(0xF3);
        self.rex(0, xmm, 0, base);
        self.b(0x0F);
        self.b(0x59);
        self.modrm_d32(xmm, base);
        self.d32(disp);
    }
    /// addss xmm, [base + disp32]   F3 0F 58 /r  (mod=10)
    pub unsafe fn addss_mem(&mut self, xmm: u8, base: u8, disp: i32) {
        self.b(0xF3);
        self.rex(0, xmm, 0, base);
        self.b(0x0F);
        self.b(0x58);
        self.modrm_d32(xmm, base);
        self.d32(disp);
    }
    /// comisd xmm, xmm2   66 0F 2F /r  (mod=11)
    ///
    /// 与 +0.0 比较，配合 jbe 实现「act == 0.0 -> 跳过」。
    /// ⚠️ 实测（不是推理）：本组合在 QNaN 上**不跳**，即 NaN 会被**乘**，
    /// 这与算子版的 `act == 0.0`（NaN 时为 false，故也乘）**同向**。
    /// 曾一度以为 comisd 的 unordered 会让 jbe 跳，实测推翻 —— 见模块头的
    /// 「一处曾经写错…」小节。用 ucomisd 也一样会踩同样的推理陷阱。
    pub unsafe fn comisd_reg(&mut self, xmm: u8, other: u8) {
        self.b(0x66);
        self.rex(0, xmm, 0, other);
        self.b(0x0F);
        self.b(0x2F);
        self.b(0xC0 | ((xmm & 7) << 3) | (other & 7));
    }
    /// movsd xmm, [base + disp32]   F2 0F 10 /r  (mod=10)
    /// ⚠️ 保留但**不要**用于 act[]（见 movss_load 的注释）。仅作原语备用。
    pub unsafe fn movsd_load(&mut self, xmm: u8, base: u8, disp: i32) {
        self.b(0xF2);
        self.rex(0, xmm, 0, base);
        self.b(0x0F);
        self.b(0x10);
        self.modrm_d32(xmm, base);
        self.d32(disp);
    }
    /// jbe rel32   0F 86 cd
    pub unsafe fn jbe_rel32(&mut self, rel: i32) {
        self.b(0x0F);
        self.b(0x86);
        self.d32(rel);
    }
    /// 回填一个 rel32 分支：rel = 目标偏移 - (rel32 字段之后的位置)
    pub unsafe fn patch_rel32(&mut self, field_at: usize, target: usize) {
        let rel = target as i64 - (field_at + 4) as i64;
        core::ptr::write_unaligned(self.p.add(field_at) as *mut i32, rel as i32);
    }
}

// ============================================================================
// 2. B1 单行转译器
// ============================================================================

/// 每神经元连接数上限（本验证只用 20）。
pub const MAX_K0: usize = 256;

/// Win64 序言：入口 rsp ≡ 8 (mod 16)；减 40 -> ≡ 0 (mod 16)。
/// 40 = 32 字节阴影空间 + 8 字节对齐修正。本函数不 call 任何东西，
/// 但按 ABI 留阴影空间，使这段码"可以被 C 正常调用"，也让将来若真要
/// 溢出到内核时 ABI 不必重排。
const STACK_ADJUST: u32 = 40;

/// 编译期已知的全部常量 —— 地址直接烘进代码，运行时零解析、零槽位查表。
#[derive(Clone, Copy)]
pub struct RowConsts {
    /// 本神经元的权重行基址 &w[i*k0]
    pub w_row: *const f32,
    /// 本神经元的 src id 行基址（u32 数组；引擎惯例是 u32-as-f32）
    pub src_row: *const u32,
    /// 本神经元的 act 掩码行基址（f32：1.0 激活 / 0.0 未激活）
    pub act_row: *const f32,
    /// x 基址。**只有它不能烘进内存操作数** —— src id 是数据。
    /// 但它由生成的代码自己算出源地址，不需要 call。
    pub x: *const f32,
    /// y 写回目标 &mut y[i]
    pub y_i: *mut f32,
    /// 本行私有的累加器槽位（4 字节，每次运行前清零）
    pub acc: *mut f32,
    /// 每神经元连接数
    pub k0: usize,
}

/// 一段已发射好的机器码，以及它需要的运行期资源。
pub struct RowCode {
    /// 可执行入口（VirtualAlloc 得到的 RWX 页首）
    pub entry: *const u8,
    /// 发射了多少字节
    pub len: usize,
    /// 累加器槽位（host 侧持有，保证比代码活得久）
    pub acc: *mut f32,
}

/// 这一行的函数签名：`extern "system" fn()`，无参数、无返回值。
pub type RowFn = unsafe extern "system" fn();

/// 转译**一个神经元的一行**：k0 条连接全部展开，无循环、无 call。
///
/// `code` 必须是 `cap` 字节的可写缓冲（调用方负责之后 make_exec）。
pub unsafe fn trans_row(rc: &RowConsts, code: *mut u8, cap: usize) -> RowCode {
    assert!(rc.k0 <= MAX_K0, "trans: k0 exceeds MAX_K0");
    let mut e = Emitter::new(code, cap);

    // ---- 序言 ----
    e.sub_rsp(STACK_ADJUST);

    // ---- 寄存器固定分配（整行不变）----
    //   r8  = w_row   （mulss 的 base）
    //   r9  = act_row （movss 的 base）
    //   r10 = src_row （mov 的 base）
    //   r11 = x       （间接取数的 base，不会被 movsxd 破坏）
    //   rcx = acc     （累加器槽位指针）
    //   rax = 临时的 src id
    //   xmm0 = 因子 / 累加器值
    //   xmm1 = act 值
    //   xmm2 = +0.0 比较常量
    e.mov_imm64(R8, rc.w_row as u64);
    e.mov_imm64(R9, rc.act_row as u64);
    e.mov_imm64(R10, rc.src_row as u64);
    e.mov_imm64(R11, rc.x as u64);
    e.mov_imm64(RCX, rc.acc as u64);
    e.pxor_xmm2_self(); // xmm2 = +0.0

    // ---- 累加器清零 ----
    // 与算子版 `if first { term } else { acc + term }` 的逐位等价性
    // 论证见模块头【边界 2】。
    e.mov_mem_imm32(RCX, 0, 0i32); // acc = +0.0

    // ---- k0 条连接，逐条展开 ----
    // 每条连接的起始偏移要记下来，供"跳过"分支回填。
    let mut conn_start = [0usize; MAX_K0];
    let mut jbe_field = [0usize; MAX_K0];

    for j in 0..rc.k0 {
        conn_start[j] = e.here();
        let d4 = (j * 4) as i32;

        // 这条连接激活吗？ act[j] != 0.0 ?
        // ⚠️ movss（32 位）+ 偏移 4*j —— act 的元素是 f32。见 movss_load 注释。
        e.movss_load(1, R9, d4); //  xmm1 = act[j]
        e.comisd_reg(1, 2); //      xmm1 vs +0.0
        e.jbe_rel32(0); //          占位（0 表示"跳到自己"，稍后回填）
        jbe_field[j] = e.here() - 4; // rel32 字段的位置

        // load x[src[j]] -> xmm0
        //
        // ⚠️ src 是「u32-as-f32」的**浮点数值**（如 63.0），必须用
        // cvttss2si 做 f32 -> 整数**转换**。曾经这里写的是
        // `mov eax, [src+4j]`（按位搬），于是 63.0f32 的位型 0x427C0000
        // 被当成索引 -> 访问违例 0xC0000005。这是本次实测抓到的真 bug，
        // 与引擎的做法（`*srcrow.add(j) as usize` / `_mm256_cvttps_epi32`）
        // 对齐后消失。
        e.movss_load(1, R10, d4); //         xmm1 = src[j]（浮点数值）
        e.cvttss2si_r64(RAX, 1); //          rax  = (i64)src[j]（向零截断）
        e.movss_load_idx4(0, R11, RAX); //   xmm0 = x[rax]

        // xmm0 *= w[j]
        e.mulss_mem(0, R8, d4);

        // acc += term  （中间值全程在寄存器）
        e.addss_mem(0, RCX, 0); // xmm0 += [rcx] (= acc)
        e.movss_store(RCX, 0, 0); // [rcx] = xmm0
    }

    // ---- 写回 y[i] ----
    // "跳过本连接"的分支统一跳到这里（即跳过该条连接的乘加）。
    // 从槽位读回是为了覆盖"整行零命中"的情形：那时 xmm0 从未被写过，
    // 而槽位已被显式清零 -> 得 +0.0，与算子版的 0.0 逐位一致。
    let writeback = e.here();
    e.movss_load(0, RCX, 0); // xmm0 = acc
    e.mov_imm64(RCX, rc.y_i as u64);
    e.movss_store(RCX, 0, 0); // *y_i = acc

    // ---- 收尾 ----
    e.add_rsp(STACK_ADJUST);
    e.ret();

    // ---- 回填所有"跳过本条连接"的分支 ----
    // 目标 = 下一条连接的起点；最后一条跳到写回段。
    // ⚠️ 这里**不能**用 e.here()（那在 ret 之后），否则"跳过"会落到函数外。
    for j in 0..rc.k0 {
        let target = if j + 1 < rc.k0 { conn_start[j + 1] } else { writeback };
        e.patch_rel32(jbe_field[j], target);
    }

    RowCode {
        entry: code,
        len: e.len,
        acc: rc.acc,
    }
}
