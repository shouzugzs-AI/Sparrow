// JIT 代码生成：指令流 → x86-64 机器码
//
// 设计（白皮书 §5.3）：JIT 生成「胶水」而不是「数学」。
//   每条指令 = 把槽位指针/形状常量装进寄存器 + call 预编译的 AVX2 内核
//   内核是 Rust 编译好的函数，JIT 不重复生成 SIMD 代码
//   → 生成器小、生成质量稳定、二进制小、且与解释器共用同一套内核
//
// 关键点：**槽位指针和形状在编译期就是已知常量**，直接编码成 imm64 /
// 内存寻址的 disp，运行时零解析、零槽位查表。这正是 JIT 相对解释器的收益。
//
// 调用约定（Win64）：
//   参数 1..4 → rcx, rdx, r8, r9；返回值 → rax
//   阴影空间 32 字节必须预留；rsp 必须 16 字节对齐
#![allow(dead_code)]

use crate::format::*;
use vmlm_tensor::act_op::{OP_EWADD, OP_EWDIV, OP_EWMUL, OP_EWSUB};

pub use vmlm_sys::mem::{alloc_rwx, make_exec};
use crate::ins::Ins;
use vmlm_ops::kernels;



pub use vmlm_sys::mem::kernel_addr;

pub struct Jit {
    pub code: *mut u8,
    pub len: usize,
    pub cap: usize,
}

extern "system" {
    fn VirtualAlloc(a: *mut u8, n: usize, t: u32, p: u32) -> *mut u8;
    fn VirtualProtect(a: *mut u8, n: usize, new: u32, old: *mut u32) -> i32;
}





impl Jit {
    pub unsafe fn new(cap: usize) -> Jit {
        Jit { code: alloc_rwx(cap), len: 0, cap }
    }
    #[inline]
    pub unsafe fn b(&mut self, v: u8) {
        if self.len >= self.cap {
            crate::print(b"JIT buffer overflow\n");
            crate::exit(11)
        }
        *self.code.add(self.len) = v;
        self.len += 1;
    }
    #[inline]
    pub unsafe fn d32(&mut self, v: i32) {
        if self.len + 4 > self.cap {
            crate::print(b"JIT buffer overflow\n");
            crate::exit(11)
        }
        core::ptr::write_unaligned(self.code.add(self.len) as *mut i32, v);
        self.len += 4;
    }
    #[inline]
    pub unsafe fn d64(&mut self, v: u64) {
        if self.len + 8 > self.cap {
            crate::print(b"JIT buffer overflow\n");
            crate::exit(11)
        }
        core::ptr::write_unaligned(self.code.add(self.len) as *mut u64, v);
        self.len += 8;
    }
    // mov r64, imm64
    pub unsafe fn mov_imm64(&mut self, reg: u8, imm: u64) {
        self.b(0x48 | if reg >= 8 { 0x01 } else { 0 });
        self.b(0xB8 + (reg & 7));
        self.d64(imm);
    }
    // sub rsp, imm32
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
    // call absolute: mov rax, imm64; call rax
    pub unsafe fn call_abs(&mut self, target: u64) {
        self.mov_imm64(0, target); // rax
        self.b(0xFF);
        self.b(0xD0); // call rax
    }
}

// 槽位视图（JIT 编译期读取，把指针/形状烘进代码）
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SlotView {
    pub p: *mut f32,
    pub rows: u32,
    pub cols: u32,
}

// Win64 参数寄存器
const RCX: u8 = 1;
const RDX: u8 = 2;
const R8: u8 = 8;
const R9: u8 = 9;
const RAX: u8 = 0;

pub struct JitCompiler<'a> {
    pub text: &'a [Ins],
    pub slots: &'a [SlotView],
}

impl<'a> JitCompiler<'a> {
    // 生成整个函数的机器码，返回入口地址
    pub unsafe fn compile(&self, jit: &mut Jit) -> usize {
        // 序言：push rbx; sub rsp, 32
        // 对齐分析（Win64 要求 call 时 rsp 是 16 的倍数）：
        //   进入函数时 rsp ≡ 8 (mod 16)（call 压入了返回地址）
        //   push rbx      → rsp ≡ 0 (mod 16)
        //   sub rsp, 32   → rsp ≡ 0 (mod 16)  ✅ 32 字节即阴影空间，且保持对齐
        jit.b(0x53); // push rbx
        jit.sub_rsp(32);

        for ins in self.text {
            self.emit(jit, ins);
        }

        // 收尾：xor eax,eax; add rsp,32; pop rbx; ret
        jit.b(0x31);
        jit.b(0xC0);
        jit.add_rsp(32);
        jit.b(0x5B); // pop rbx
        jit.b(0xC3); // ret
        jit.code as usize
    }

    unsafe fn ptr_of(&self, slot: u32) -> u64 {
        self.slots[slot as usize].p as u64
    }
    unsafe fn shape_of(&self, slot: u32) -> (u64, u64) {
        let s = self.slots[slot as usize];
        (s.rows as u64, s.cols as u64)
    }

    unsafe fn emit(&self, jit: &mut Jit, ins: &Ins) {
        let d = ins.dst;
        let a = ins.a;
        let b = ins.b;
        match ins.op {
            OP_HALT | OP_SHAPE => {}

            // ---- GEMM_NT: c[m,n] = a[m,k] * b[n,k]^T ----
            OP_GEMM => {
                let (m, k) = self.shape_of(a);
                let (n, _bk) = self.shape_of(b);
                jit.mov_imm64(RCX, self.ptr_of(a));
                jit.mov_imm64(RDX, self.ptr_of(b));
                jit.mov_imm64(R8, self.ptr_of(d));
                // 第 4 参数：dims 打包成 u64 = m | n<<16 | k<<32 | acc<<48
                let acc = ((ins.fl as u16) & FL_ACC != 0) as u64;
                let dims = m | (n << 16) | (k << 32) | (acc << 48);
                jit.mov_imm64(R9, dims);
                jit.call_abs(kernels::jit_gemm_nt as usize as u64);
            }

            // ---- ACT: p[0..n] = act(p[0..n], kind) ----
            OP_ACT => {
                let (m, c) = self.shape_of(d);
                let n = m * c;
                let src = self.ptr_of(a);
                let dst = self.ptr_of(d);
                if src != dst {
                    // 先复制（用 memcpy 内核省代码）
                    jit.mov_imm64(RCX, dst);
                    jit.mov_imm64(RDX, src);
                    jit.mov_imm64(R8, n * 4);
                    jit.call_abs(kernels::jit_memcpy as usize as u64);
                }
                jit.mov_imm64(RCX, dst);
                jit.mov_imm64(RDX, n);
                let kind = if ins.fl == 0 { ACT_RELU as u64 } else { ins.fl as u64 };
                jit.mov_imm64(R8, kind);
                jit.call_abs(kernels::jit_act as usize as u64);
            }

            // ---- 逐元素 ----
            OP_EWADD | OP_EWMUL | OP_EWSUB | OP_EWDIV => {
                let (m, c) = self.shape_of(a);
                jit.mov_imm64(RCX, self.ptr_of(d));
                jit.mov_imm64(RDX, self.ptr_of(a));
                jit.mov_imm64(R8, self.ptr_of(b));
                jit.mov_imm64(R9, m * c);
                // 第 5 参数 op 走栈；为省事把 op 编进单独内核选择
                jit.call_abs(op_kernel(ins.op) as usize as u64);
            }

            // ---- BIAS: dst[m,n] += bias[n] ----
            OP_BIAS => {
                let (m, c) = self.shape_of(d);
                jit.mov_imm64(RCX, self.ptr_of(d));
                jit.mov_imm64(RDX, self.ptr_of(a));
                jit.mov_imm64(R8, m);
                jit.mov_imm64(R9, c);
                jit.call_abs(kernels::jit_bias as usize as u64);
            }

            // ---- LayerNorm ----
            // kernel(x=rcx, g=rdx, b=r8, dims=r9, _pad=stack)
            OP_LN => {
                let (m, c) = self.shape_of(d);
                let dims = m | (c << 32);
                jit.mov_imm64(RCX, self.ptr_of(d));
                jit.mov_imm64(RDX, self.ptr_of(b));
                jit.mov_imm64(R8, self.ptr_of(b + 1));
                jit.mov_imm64(R9, dims);
                // 第 5 参数走栈，压 16 字节保持 16 对齐
                push_u64(jit, 0);
                push_u64(jit, 0);
                jit.call_abs(kernels::jit_ln as usize as u64);
                jit.add_rsp(16);
            }

            // ---- softmax ----
            OP_SOFTMAX => {
                let (m, c) = self.shape_of(d);
                jit.mov_imm64(RCX, self.ptr_of(d));
                let dims = m | (c << 32);
                jit.mov_imm64(RDX, dims);
                jit.mov_imm64(R8, ((ins.fl as u16) & FL_CAUSAL != 0) as u64);
                jit.call_abs(kernels::jit_softmax as usize as u64);
            }

            // ---- OUTER: S[m,n] += v[m] * k[n] ----
            OP_OUTER => {
                let (m, n) = self.shape_of(a);
                jit.mov_imm64(RCX, self.ptr_of(d));
                jit.mov_imm64(RDX, self.ptr_of(a));
                jit.mov_imm64(R8, self.ptr_of(b));
                let dims = m | (n << 32);
                jit.mov_imm64(R9, dims);
                jit.call_abs(kernels::jit_outer as usize as u64);
            }

            // ---- 调试打印 ----
            OP_PRINT => {
                let (m, c) = self.shape_of(d);
                jit.mov_imm64(RCX, self.ptr_of(d));
                jit.mov_imm64(RDX, m * c);
                jit.call_abs(kernels::jit_print as usize as u64);
            }

            _ => {
                // 未实现的 opcode —— 必须**在编译期硬失败**，不能静默生成一个空操作。
                //
                // P9 成员 J 发现：原来这里只生成一次 `jit_unimpl` 调用，而
                // `jit_unimpl` 只打印一行就正常返回。后果是走 `jit_run()` 时
                // 反向算子/PUSH/POP/SLICE 等全部变成 no-op —— 程序**正常退出**
                // 但梯度恒为 0、参数永不更新。这是最恶劣的一类静默降级。
                //
                // 当前实际无影响（全仓库 jit_run/jit_compile 零调用点，所有驱动
                // 走解释器 vm.run()），但 jit_run 是 pub 且会自动编译，
                // 将来任何人为了加速训练改用 JIT 都会踩中。所以在源头掐死。
                crate::print(b"JIT: unimplemented opcode, refusing to compile\n");
                crate::interp::exit_i(6);
            }
        }
    }
}

unsafe fn push_u64(jit: &mut Jit, v: u64) {
    // mov rax, imm64; push rax
    jit.mov_imm64(RAX, v);
    jit.b(0x50);
}

fn op_kernel(op: u8) -> usize {
    match op {
        OP_EWADD => kernels::jit_ewadd as usize,
        OP_EWMUL => kernels::jit_ewmul as usize,
        OP_EWSUB => kernels::jit_ewsub as usize,
        _ => kernels::jit_ewdiv as usize,
    }
}
