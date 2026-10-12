// Interpreter: load a .vmb and execute the instruction stream.
use crate::format::*;
use crate::kernels;
use crate::{u32_at, INS_SIZE};
// [抽库 L2b] 指令表示搬到 vmlm-trans
pub use vmlm_trans::ins::Ins;

// [缺陷修复] `HAS_AVX2` 曾经有**两份** static：这里一份（由下面 `init_cpu()` 赋值），
// `sys/src/cpu/mod.rs:33` 又一份（**全仓库没有任何地方给它赋值** ⇒ 恒 false）。
// 而 `ops/src/lib.rs` 的 shim `vmlm_ops::interp::HAS_AVX2` 指向 sys 那一份，
// 于是 `ops/src/kernels.rs` 里 `gemm_nt`(1005) / `outer`(1439) 的 AVX2 分支自 L2a 搬库起
// 就没进过（静默降级：回退分支仍被编译期自动向量化，体积与逐位判据都看不出来）。
// 现合并为**一份**：sys 持有，本处只重导出 ⇒ `init_cpu()` 写的就是所有读取者看到的那一份。
pub use vmlm_sys::cpu::HAS_AVX2;

// [抽库 L0-f] 探测原语搬到 vmlm-sys::cpu；路径 vmlm_core::interp::cpu_has_avx2 不变
pub use vmlm_sys::cpu::cpu_has_avx2;

/// Upper bound on the number of connections `OP_TOPK_BY_STRENGTH` will select
/// over, imposed by the engine's deliberately stubbed `__chkstk`.
///
/// The engine avoids the CRT, so `__chkstk` (the stack-probe helper the
/// compiler emits for large frames) is provided as an empty stub in lib.rs.
/// A frame big enough to skip over the guard page therefore SEGFAULTS with no
/// diagnostic instead of trapping cleanly -- measured as exit 3221225477
/// (0xC0000005) and an empty stdout, which is indistinguishable from many
/// other faults. Keeping the selection buffers under a few KB is what keeps
/// that from being reachable.
///
/// 2048 x (1 byte for the bitmap + 8 bytes for the index buffer) = 18 KB, and
/// P16's k0 is 20 -- three orders of magnitude of headroom. Exceeding it is an
/// explicit `exit(9)`, never a crash.
pub const TOPK_MAX_L: usize = 2048;



// /
#[derive(Clone, Copy)]
pub struct Slot {
    pub p: *mut f32,
    pub rows: u32,
    pub cols: u32,
    pub is_const: bool,
    /// P12：常量的存储 dtype（`DT_F32` / `DT_BF16` …）。激活槽恒为 `DT_F32`。
    ///
    /// 为什么必须带上它：常量表里**一直**有 dtype 字段，但引擎从不读，
    /// 于是任何非 f32 的常量都被当成 f32 解释 —— 读出来是垃圾数且不报错。
    /// MiniCPM 的权重是 bf16，所以这是「实际加载模型」的硬阻塞。
    ///
    /// 现在 `p` 仍按 `*mut f32` 声明（让 129 处既有的解引用保持不动），
    /// 但**消费常量张量的算子必须先检查 dtype**：
    /// GEMM/GEMV 分派到 bf16 内核；其它算子收到非 f32 常量一律硬失败。
    /// 这样「按错类型读内存」从静默垃圾变成可读的 `exit(9)`。
    pub dtype: u8,
}

pub struct Vm {
    pub text: *const Ins,
    pub n_ins: usize,
    pub cpool: *mut f32, // constant pool base (writable: params updated in place)
    pub slots: *mut Slot,
    pub n_slots: usize,
    pub act: *mut f32,   // activation area (temporaries, reused)
    // Persistent state area (RNN recurrence / fatigue counters). Survives
    // across steps -- that is what makes ESERNN-style recurrence work.
    pub state: *mut f32,
    pub const_table: *const u8,
    pub n_const: usize,
    pub n_act_bytes: usize,
    pub n_state_bytes: usize,
    pub jit_entry: usize,
    pub grad: *mut f32, // gradient area (same layout as the constant pool)
    pub slot_tab: *const u8, // slot table, original location (name lookup)
    pub name_tab: *const u8, // slot-name hash table
    pub loss: f32,
    // Adam state (m/v, same layout as the constant pool, allocated lazily)
    pub adam_m: *mut f32,
    pub adam_v: *mut f32,
    pub adam_t: u32,
    pub grad_dirty: bool,

pub ln_eps: f32, // LayerNorm eps1e-5LN_EPS 
    /// P10****C mmap     ///
    /// mmap  PAGE_READONLY
    /// ADAM`/`ADAMW`/`SGD`/    /// **
    /// `exit(9)` 0xC0000005  CI     /// "process exited with code 3221225477"
pub cpool_readonly: bool,
    /// P13 task E: runtime byte offset added to every CONSTANT slot pointer.
    ///
    /// Default 0, so a program that never issues OP_LAYERB behaves exactly as
    /// before -- the unrolled 24-layer artifact is unaffected.
    ///
    /// Why a single base rather than per-slot addressing: the artifact's
    /// per-layer constants were MEASURED to be laid out with a perfectly
    /// uniform stride (28,383,232 B for all 16 of them), so advancing one base
    /// moves every per-layer weight to the next layer at once. That keeps the
    /// loop body identical to a single layer's.
    ///
    /// Safety: every const access validates that base + offset stays inside the
    /// constant pool (see const_ptr), so a runaway loop hard-fails instead of
    /// reading the activation region as if it were weights.
    pub layer_base: usize,
    /// Size of the constant region, used for the OP_LAYERB bound check.
    pub const_bytes: usize,
}

impl Vm {
    pub unsafe fn load(buf: *mut u8) -> Vm {
        Vm::load_impl(buf, None)
    }

    /// P10  C*+  mmap**     ///
    /// `buf`  + META .exe
    /// `weights` mmap ***    ///
    ///  `load` mmap `load`     /// `cpool`  `OP_ADAM`/`OP_SGD` *
    /// *    ///      ///
    /// # Safety
    /// `weights`  `const_len` 
    /// `buf`  .vmbCONST 
pub unsafe fn load_weights_mapped(buf: *mut u8, weights: *const u8) -> Vm {
        Vm::load_impl(buf, Some(weights))
    }
    unsafe fn load_impl(buf: *mut u8, weights: Option<*const u8>) -> Vm {
        let magic = u32_at(buf, 0);
        if magic != MAGIC {
            crate::print(b"bad magic\n");
            exit_i(3);
        }
        let text_off = u32_at(buf, 8) as usize;
        let text_len = u32_at(buf, 12) as usize;
        let const_off = u32_at(buf, 16) as usize;
        let meta_off = u32_at(buf, 24) as usize;

        let n_const = u32_at(buf, meta_off) as usize;
        let n_slots = u32_at(buf, meta_off + 4) as usize;
        let slots_off = u32_at(buf, meta_off + 8) as usize;
        let act_bytes = u32_at(buf, meta_off + 12) as usize;
        let state_bytes = u32_at(buf, meta_off + 16) as usize;
        let grad_bytes = u32_at(buf, meta_off + 20) as usize;

        // ---- P10:  ----
        // buf const_off         //
        // ** CONST * .vmb
        // ---- P10: constant pool base ----
        // No mapping: take it from const_off inside the same buf (unchanged
        // behaviour, byte for byte).
        //
        // With a mapping: the weights file IS the CONST region (not a whole
        // .vmb), so the mapping base already IS the pool start -- do NOT add
        // const_off again. (First version added it, overshooting the end of
        // the mapping by 7760 bytes -> 0xC0000005 on the first element read.)
        // Per-constant offsets in META are already relative to the CONST
        // start, so once the base is right, `cpool + off/4` is right.
        let cpool_base: *mut u8 = match weights {
            Some(w) => {
                // Treating read-only memory as a writable pointer is
                // deliberate: the inference path never writes it, and the
                // training path is stopped by the `cpool_readonly` guard
                // (exit(9)) rather than by a raw access violation.
                w as *mut u8
            }
            None => buf.add(const_off),
        };

        // /  step  
        //
        // P13/P14: the ACTIVATION region is zeroed too, for the same reason
        // `state` and `grad` already were.
        //
        // It was the only region left uninitialised, which meant an activation
        // slot that a program reads BEFORE writing held whatever the allocator
        // returned. That is normally invisible -- the arithmetic overwrites it,
        // or a garbage value merely yields a garbage result. But it becomes a
        // HARD FAILURE against any guard that inspects a value's magnitude or
        // integrality: measured, `EMB` with an uninitialised row-index slot
        // exited 9 with "row index is not an integer" while PRINTING
        // "row=0.000000", because print_f32 truncates 6 decimals and the real
        // value was a tiny non-integer heap remnant.
        //
        // Zeroing is also the honest choice: a deterministic 0 makes an
        // unwritten slot behave the same on every run, which is what the
        // project's measured results have always assumed.
        let act = crate::alloc(act_bytes + 64) as *mut f32;
        core::ptr::write_bytes(act as *mut u8, 0, act_bytes + 64);
        let st = crate::alloc(state_bytes + 64) as *mut f32;
        core::ptr::write_bytes(st as *mut u8, 0, state_bytes + 64);
        let grad = crate::alloc(grad_bytes + 64) as *mut f32;
        core::ptr::write_bytes(grad as *mut u8, 0, grad_bytes + 64);

        let mut vm = Vm {
            text: buf.add(text_off) as *const Ins,
            n_ins: text_len / INS_SIZE,
            cpool: cpool_base as *mut f32,
            slots: core::ptr::null_mut(),
            n_slots,
            act,
            state: st,
            const_table: buf.add(meta_off + META_HDR),
            n_const,
            n_act_bytes: act_bytes,
            n_state_bytes: state_bytes,
            jit_entry: 0,
            grad,
            slot_tab: buf.add(meta_off + slots_off),
            name_tab: buf.add(meta_off + u32_at(buf, meta_off + 24) as usize),
            loss: 0.0,
            adam_m: crate::alloc(grad_bytes + 64) as *mut f32,
            adam_v: crate::alloc(grad_bytes + 64) as *mut f32,
            adam_t: 0,
            grad_dirty: false,
            ln_eps: 1e-5,
            cpool_readonly: weights.is_some(),
            // P13 task E: 0 by default, so unrolled programs are unaffected.
            layer_base: 0,
            // Bound for the OP_LAYERB check. The .vmb header's const length is
            // authoritative in BOTH paths: with mmap the weights file IS the
            // const region and is written with the same length in its header.
            const_bytes: u32_at(buf, 20) as usize,
        };
        core::ptr::write_bytes(vm.adam_m as *mut u8, 0, grad_bytes + 64);
        core::ptr::write_bytes(vm.adam_v as *mut u8, 0, grad_bytes + 64);
        vm.bind_slots(buf.add(meta_off + slots_off));
        vm.audit_const_dtypes();
        vm
    }

    /// P12 装载期审计：**非 f32 常量只允许作为 `GEMM` 的 B 操作数**。
    ///
    /// 为什么需要（这是实测崩溃换来的）：
    /// 引擎里 129 处代码直接把 `Slot.p` 当 `*const f32` 读。bf16 常量按
    /// 2 字节/元素存放，被当 f32 读就会**每元素多读一倍**，越界到映射外
    /// → `0xC0000005`（实测：`COPY h, prompt_emb` 崩在第一条指令）。
    ///
    /// 我原本只在 `OP_GEMM` 里加了 dtype 判断，但那只保护了 GEMM 一条路径；
    /// `COPY`/`RMSNORM`/`EWADD` … 全部照旧把它们当 f32。
    /// 与其逐个补 40 个分支（必定漏），不如**在装载时一次扫完指令流**：
    /// 任何「非 GEMM 指令的操作数是 bf16 常量」都直接拒绝装载。
    ///
    /// 代价 O(n_ins) 一次，且把「按错类型读内存」从**随机崩溃**
    /// 变成**装载期可读报错** —— 这正是本项目对静默降级的一贯处理。
    unsafe fn audit_const_dtypes(&self) {
        for i in 0..self.n_ins {
            let ins = *self.text.add(i);
            if ins.op == OP_GEMM {
                // GEMM: only the B operand may be non-f32 (A is an activation).
                //
                // P13 task D added DT_I8 here. The important part is not the
                // extra dtype but that the KERNEL existence is checked: a bf16
                // B is rejected below for NN/acc layouts, and an i8 B needs its
                // scale slot validated. Declaring a dtype the engine cannot
                // actually compute with must fail at LOAD, not produce wrong
                // numbers -- this is the same class of bug P12 fixed.
                let bdt = self.slot_dtype(ins.b as usize);
                if bdt == DT_BF16 || bdt == DT_I8 {
                    continue;
                }
                if bdt != DT_F32 {
                    crate::print(b"gemm B has no kernel for this dtype\n");
                    exit_i(9);
                }
                // A non-f32 activation is impossible (activations are
                // engine-owned), so nothing else to check here.
                let _ = ins.a;
                continue;
            }
            if ins.op == OP_EMB {
                // P13 task A: EMB is now dtype-aware like GEMM. Its `a` is the
                // table and may legitimately be bf16 (embed_tokens IS bf16:
                // [130560,1536] = 401 MB), while `b` is a scalar row index -- a
                // NUMBER that is never dereferenced as a table, so probing it
                // would produce false rejections (same reason SLICE/WRITECOL
                // skip their `b`).
                //
                // The fix must be a NAMED exception listing exactly which
                // operand may be non-f32, not a blanket weakening: `dst` is an
                // engine-owned f32 activation and is STILL probed below. If a
                // bf16 table is ever fed to an op that really does read it as
                // f32, this still fails loudly with opcode + slot number.
                if self.slot_dtype(ins.a as usize) != DT_BF16 {
                    continue; // f32 table: dst must still be checked
                }
                if self.slot_dtype(ins.dst as usize) == DT_F32 {
                    continue; // bf16 table + f32 dst: exactly what EMB supports
                }
                // Non-f32 dst: fall through so the generic probe reports it.
            }
            // Every other opcode reads its operands as f32.
            //
            // ... and so must instructions whose `b` (or `dt`) holds a NUMBER
            // rather than a slot: SLICE/WRITECOL store a column offset in `b`,
            // PUSH/POP store the step index, SHAPE stores rows/cols.
            // Checking those fields for "is this a bf16 slot" is meaningless
            // and produced false rejections.
            let (check_dst, check_a, check_b) = match ins.op {
                // SHAPE is a NO-OP meta instruction: dst names a slot but the
                // engine never dereferences it, and a/b are rows/cols. Checking
                // any of its fields produces false rejections (measured:
                // `SHAPE w, 64, 128` was flagged as "slot 64 used by a
                // non-GEMM op" and refused a valid program).
                OP_SHAPE => (false, false, false),
                // LOOP_CNT: a = counter slot, b unused.
                OP_LOOP_CNT => (false, true, false),
                // SLICE/WRITECOL: b = column offset. PUSH/POP: b = step index.
                OP_SLICE | OP_WRITECOL | OP_PUSH | OP_POP => (true, true, false),
                _ => (true, true, true),
            };
            let mut probe = |s: usize| {
                if s >= self.n_slots {
                    return;
                }
                // P13 task D: this used to test `== DT_BF16` only, so an int8
                // constant passed through unchecked. Measured: `COPY x, w` with
                // an i8 weight EXITED 0 -- the copy read i8 bytes as f32, which
                // is precisely the silent-garbage class P12 fixed for bf16.
                //
                // The invariant is "every operand of a non-GEMM op must be f32",
                // so test THAT rather than enumerating dtypes; a future dtype
                // then cannot silently reopen the hole.
                let dt = self.slot_dtype(s);
                if dt == DT_F32 {
                    return;
                }
                // Name the opcode and slot so the fix is obvious. A bare
                // "bf16 const used by non-GEMM op" cost me a bisect cycle, and
                // the dtype name matters because i8 is a 4x over-read while
                // bf16 is only 2x.
                crate::print(b"op ");
                print_u32(ins.op as u32);
                crate::print(match dt {
                    DT_BF16 => b" uses bf16 const at slot " as &[u8],
                    DT_I8 => b" uses i8 const at slot ",
                    _ => b" uses non-f32 const at slot ",
                });
                print_u32(s as u32);
                crate::print(b" (only a GEMM may read a non-f32 operand)\n");
                exit_i(9);
            };
            if check_dst {
                probe(ins.dst as usize);
            }
            if check_a {
                probe(ins.a as usize);
            }
            if check_b {
                probe(ins.b as usize);
            }
        }
    }

    // META ind /  / 
    unsafe fn bind_slots(&mut self, stab: *const u8) {
        self.slots = crate::alloc(self.n_slots * core::mem::size_of::<Slot>()) as *mut Slot;
        for i in 0..self.n_slots {
            let e = stab.add(i * SLOT_ENT);
            let kind = u32_at(e, 0);
            let idx = u32_at(e, 4) as usize;
            let rows = u32_at(e, 8);
            let cols = u32_at(e, 12);
            // P12: const slots carry their storage dtype. Activation / state /
            // grad are always f32 (they are engine-owned scratch).
            let mut dt = DT_F32;
            let p = match kind {
                SLOT_CONST | SLOT_PARAM => {
                    let ce = self.const_table.add(idx * CONST_ENT);
                    let off = u32_at(ce, 0) as usize;
                    // const entry layout: off(0) len(4) rows(8) cols(12) dtype(16)
                    dt = *self.const_table.add(idx * CONST_ENT + 16);
                    let esz = dtype_size(dt);
                    if esz == 0 {
                        // Unknown dtype must NOT be silently read as f32 --
                        // that is the exact bug this field was meant to stop.
                        crate::print(b"const has unknown dtype\n");
                        exit_i(9);
                    }
                    // `p` is typed *mut f32, so scale the byte offset by the
                    // element size: for f32 that is off/4 (unchanged); for bf16
                    // that is off/2 and the kernel widens on the fly.
                    //
                    // P13 task E: `layer_base` is added at ACCESS time, not
                    // here, because this binding happens ONCE at load while the
                    // base changes per loop iteration. See slot_mut.
                    (self.cpool as *mut u8).add(off) as *mut f32
                }
                SLOT_STATE => self.state.add(idx / 4),
                SLOT_GRAD => self.grad.add(idx / 4),
                _ => self.act.add(idx / 4),
            };
            *self.slots.add(i) = Slot {
                p,
                rows,
                cols,
                is_const: kind == SLOT_CONST,
                dtype: dt,
            };
        }
    }

    //  FNV 
    pub unsafe fn find(&self, name: &str) -> usize {
        let h = hash32(name);
        for i in 0..self.n_const {
            let e = self.const_table.add(i * 24);
            if u32_at(e, 20) == h {
                return i;
            }
        }
        crate::print(b"const not found\n");
        exit_i(4)
    }

    // /
pub unsafe fn find_slot(&self, name: &str) -> usize {
        let h = hash32(name);
        for i in 0..self.n_slots {
            if u32_at(self.name_tab.add(i * 4), 0) == h {
                return i;
            }
        }
        crate::print(b"slot not found\n");
        exit_i(7)
    }

    /// Byte pointer for a slot, with the P13 task E runtime layer base applied
    /// to CONSTANT slots.
    ///
    /// AVAILABILITY NOTE (this is a code-quality guard, not byte-shaving):
    /// `slot_mut` is the hottest accessor in the engine -- roughly 40 call sites,
    /// and `const_ptr` inlines into every one of them. The first version put the
    /// bounds check inline too, and the engine grew 57344 -> 70144 B (+12.8 KB)
    /// because that branch got duplicated 40 times.
    ///
    /// The check is therefore split: the overwhelmingly common case
    /// (`layer_base == 0`, i.e. every program that does not use OP_LAYERB, which
    /// is all of them except the looped-layer mode) is a single compare and
    /// returns the precomputed pointer. The arithmetic and the bound check live
    /// in `const_ptr_based`, which is `#[inline(never)]` so it exists ONCE.
    #[inline(always)]
    unsafe fn const_ptr(&self, i: usize) -> *mut f32 {
        let s = *self.slots.add(i);
        if self.layer_base == 0 || !s.is_const {
            return s.p;
        }
        self.const_ptr_based(s)
    }

    /// Cold half of `const_ptr`: apply `layer_base` and validate the result.
    ///
    /// A runaway LOOP_CNT would otherwise walk the base past the weights into the
    /// activation region and read whatever is there -- numbers that look
    /// plausible and are meaningless. That is the silent-degradation class this
    /// project converts to hard failures, so it exits(9) instead.
    #[inline(never)]
    unsafe fn const_ptr_based(&self, s: Slot) -> *mut f32 {
        let p = s.p as *mut u8;
        let cpool = self.cpool as *mut u8;
        // The stored pointer is cpool + off; recover off to bound-check it.
        let off = p as usize - cpool as usize;
        let bytes = (s.rows as usize) * (s.cols as usize) * dtype_size(s.dtype);
        if off + self.layer_base + bytes > self.const_bytes {
            // Print the numbers, not just a message. A bare "walks past" left me
            // guessing WHICH read and by how much, on a 0.6 GB artifact that is
            // expensive to instrument any other way.
            crate::print(b"layer base walks past the constant pool: off=");
            print_u32(off as u32);
            crate::print(b" base=");
            print_u32(self.layer_base as u32);
            crate::print(b" bytes=");
            print_u32(bytes as u32);
            crate::print(b" limit=");
            print_u32(self.const_bytes as u32);
            crate::print(b"\n");
            exit_i(9);
        }
        cpool.add(off + self.layer_base) as *mut f32
    }

    pub unsafe fn slot_mut(&mut self, i: usize) -> *mut f32 {
        self.const_ptr(i)
    }
    /// P12: storage dtype of a slot (always `DT_F32` for scratch slots).
    pub unsafe fn slot_dtype(&self, i: usize) -> u8 {
        (*self.slots.add(i)).dtype
    }
    /// P12: hard-fail when an op that assumes f32 receives a non-f32 constant.
    ///
    /// This is the guard that makes the whole dtype path safe: without it,
    /// feeding a bf16 constant to (say) `EWADD` would read half the elements
    /// and reinterpret their bits as f32 -- silent garbage, which is exactly
    /// the class of bug the dtype field was supposed to prevent but never did.
    #[inline(always)]
    unsafe fn require_f32(&self, i: usize, what: &[u8]) {
        if self.slot_dtype(i) != DT_F32 {
            crate::print(what);
            exit_i(9);
        }
    }
    pub unsafe fn slot_shape(&self, i: usize) -> (usize, usize) {
        let s = *self.slots.add(i);
        (s.rows as usize, s.cols as usize)
    }

    // ---------- rain feature---------
    #[cfg(feature = "train")]
    unsafe fn exec_bwd(&mut self, ins: Ins, d: usize, a: usize, b: usize) {
        match ins.op {
            // dw[N,K] += dy[M,N]^T  x[M,K]   dst=dw(, a=dy, b=x
            OP_GEMM_BWD_W => {
                let (m, n) = self.slot_shape(a);
                let (_mk, k) = self.slot_shape(b);
                crate::train::gemm_bwd_w(
                    self.slot_mut(a),
                    self.slot_mut(b),
                    self.slot_mut(d),
                    m,
                    n,
                    k,
                );
            }
            // dx[M,K] = dy[M,N]  w[N,K]   dst=dx, a=dy, b=w(
            OP_GEMM_BWD_X => {
                let (m, n) = self.slot_shape(a);
                let (_nn, k) = self.slot_shape(b);
                crate::train::gemm_bwd_x(
                    self.slot_mut(a),
                    self.slot_mut(b),
                    self.slot_mut(d),
                    m,
                    n,
                    k,
                );
            }
            OP_BIAS_BWD => {
                let (m, n) = self.slot_shape(a);
                crate::train::bias_bwd(self.slot_mut(a), self.slot_mut(d), m, n);
            }
            OP_ACT_BWD => {
                let (m, c) = self.slot_shape(a);
                let kind = if ins.fl == 0 { ACT_RELU } else { ins.fl as u16 };
                crate::train::act_bwd(
                    self.slot_mut(a),
                    self.slot_mut(b),
                    self.slot_mut(d),
                    m * c,
                    kind,
                );
            }
            // dst=logits, a=dlogits, b=target
            OP_CE_BWD => {
                let (r, c) = self.slot_shape(d);
                let tgt = *self.slot_mut(b) as usize;
                self.loss =
                    crate::train::ce_softmax_bwd(self.slot_mut(d), tgt, self.slot_mut(a), r * c);
            }
            OP_MSE_BWD => {
                let (r, c) = self.slot_shape(d);
                self.loss = crate::train::mse_bwd(
                    self.slot_mut(d),
                    self.slot_mut(a),
                    self.slot_mut(b),
                    r * c,
                );
            }
            OP_ZERO => {
                let (r, c) = self.slot_shape(d);
                core::ptr::write_bytes(self.slot_mut(d) as *mut u8, 0, r * c * 4);
            }
            OP_LOSS_ADD => {
                let v = *self.slot_mut(a);
                *self.slot_mut(d) += v;
            }
            OP_CLIP_GRAD => {
                let (r, c) = self.slot_shape(d);
                let mx = *self.slot_mut(a);
                crate::train::clip_grad(self.slot_mut(d), r * c, mx);
            }
            // ADAM: dst=, a= b=lr,b1,b2,eps]
            OP_ADAM => {
                let (r, c) = self.slot_shape(d);
                let hp = self.slot_mut(b);
                let mut ad = crate::train::Adam {
                    m: self.adam_m,
                    v: self.adam_v,
                    lr: *hp,
                    b1: *hp.add(1),
                    b2: *hp.add(2),
                    eps: *hp.add(3),
                    t: self.adam_t,
                };
                ad.step(self.slot_mut(d), self.slot_mut(a), r * c);
                self.adam_t = ad.t;
                self.grad_dirty = true;
            }
            // SGD: dst= a= b=lr
            OP_SGD => {
                let (r, c) = self.slot_shape(d);
                let lr = *self.slot_mut(b);
                crate::train::sgd(self.slot_mut(d), self.slot_mut(a), r * c, lr);
                self.grad_dirty = true;
            }

            // ============ P7  ============
            // SOFTMAX_BWD: dst=dx, a=dy, b=y( softmax )
            OP_SOFTMAX_BWD => {
                let (r, c) = self.slot_shape(a);
                let causal = (ins.fl as u16) & FL_CAUSAL != 0;
                crate::autograd::softmax_bwd(
                    self.slot_mut(a),
                    self.slot_mut(b),
                    self.slot_mut(d),
                    r,
                    c,
                    causal,
                );
            }
            // LN_BWD: dst=dx, a=dy, b=x, dt=gamma             //   dg/db  gamma gi+1 = dg, gi+2 = db            //   ****asm LN_BWD             //
            //   **x**dy            //  OP_LN bug  src             // OP_LN st             //  dy dy x  `mean` cols            // ***9  I 1.0000
            // 1.007~1.022
            OP_LN_BWD => {
                let (r, c) = self.slot_shape(b);
                let gi = ins.dt as usize;
                crate::autograd::layernorm_bwd(
                    self.slot_mut(a),
                    self.slot_mut(b),
                    self.slot_mut(gi),
                    self.slot_mut(d),
                    self.slot_mut(gi + 1),
                    self.slot_mut(gi + 2),
                    r,
                    c,
                    self.ln_eps,
                );
            }
            // OUTER_BWD: outer product backwards
            //   a   = ds
            //   b   = v
            //   dt  = k
            //   dst = dv
            OP_OUTER_BWD => {
                let (m, n) = self.slot_shape(a);
                crate::autograd::outer_bwd(
                    self.slot_mut(a),                   // ds
                    self.slot_mut(b),                   // v
                    self.slot_mut(ins.dt as usize),     // k
                    self.slot_mut(d),                   // dv
                    // dk = k + 1 (dk immediately follows k; see asm OUTER_BWD)
                    self.slot_mut(ins.dt as usize + 1),
                    m,
                    n,
                );
            }
            // EMB_BWD: dst=dtable, a=dy, b=
            OP_EMB_BWD => {
                let (tr, _tc) = self.slot_shape(d);
                let (_dr, dc) = self.slot_shape(a);
                let row = *self.slot_mut(b) as usize;
                crate::autograd::emb_bwd(self.slot_mut(a), (tr, dc), row, self.slot_mut(d));
            }
            // ADAMW: dst=, a=, b=lr,b1,b2,eps,wd]
            OP_ADAMW => {
                let (r, c) = self.slot_shape(d);
                let hp = self.slot_mut(b);
                let mut ad = crate::autograd::AdamW {
                    m: self.adam_m,
                    v: self.adam_v,
                    lr: *hp,
                    b1: *hp.add(1),
                    b2: *hp.add(2),
                    eps: *hp.add(3),
                    wd: *hp.add(4),
                    t: self.adam_t,
                };
                ad.step(self.slot_mut(d), self.slot_mut(a), r * c);
                self.adam_t = ad.t;
                self.grad_dirty = true;
            }
            _ => {}
        }
    }

    // un() ip=0 
    pub unsafe fn clear_state(&mut self) {
        core::ptr::write_bytes(self.state as *mut u8, 0, self.n_state_bytes);
    }

    // ---------- JIT----------
    pub unsafe fn jit_compile(&mut self) -> usize {
        let text = core::slice::from_raw_parts(self.text, self.n_ins);
        // JIT + 
            let view = crate::alloc(self.n_slots * core::mem::size_of::<crate::jit::SlotView>())
            as *mut crate::jit::SlotView;
        for i in 0..self.n_slots {
            let s = *self.slots.add(i);
            *view.add(i) = crate::jit::SlotView {
                p: s.p,
                rows: s.rows,
                cols: s.cols,
            };
        }
        let mut jit = crate::jit::Jit::new(64 * 1024);
        let comp = crate::jit::JitCompiler {
            text,
            slots: core::slice::from_raw_parts(view, self.n_slots),
        };
        let entry = comp.compile(&mut jit);
        //  ^X RW RX
        crate::jit::make_exec(jit.code, jit.cap);
        self.jit_entry = entry;
        entry
    }

    // JIT entry: compile on first use, then call the generated code.
    pub unsafe fn jit_run(&mut self) -> u32 {
        if self.jit_entry == 0 {
            self.jit_compile();
        }
        let f: extern "system" fn() -> u32 = core::mem::transmute(self.jit_entry);
        f()
    }

    // ---------- core interpreter loop ----------
    pub unsafe fn run(&mut self) -> u32 {
        let n = self.n_ins;
        let text = self.text;
        let mut ip = 0usize;
        while ip < n {
            let ins = *text.add(ip);
            ip += 1;
            let d = ins.dst as usize;
            let a = ins.a as usize;
            let b = ins.b as usize;
            match ins.op {
                OP_HALT => break,

                OP_GEMM => {
                    // NT (default):  C[M,N] = A[M,K] · B[N,K]^T   (nn.Linear weights [N,K])
                    // NN (fl=nn):    C[M,N] = A[M,K] · B[K,N]     (attention att @ v)
                    //
                    // The output width N comes from a DIFFERENT operand per layout:
                    //   NT: rows(B) is N     NN: cols(B) is N
                    let (m, k) = self.slot_shape(a);
                    let (b_rows, b_cols) = self.slot_shape(b);
                    let nt = (ins.fl as u16) & FL_TRANS_B == 0;
                    let acc = (ins.fl as u16) & FL_ACC != 0;
                    let cc = self.slot_mut(d);
                    let ca = self.slot_mut(a);

                    // ---- P12: dtype dispatch -------------------------------
                    // A bf16 constant must go through the widening kernel rather
                    // than be reinterpreted as f32 (which yields garbage without
                    // any error). Activations are always f32.
                    let bt = self.slot_dtype(b);
                    if bt == DT_BF16 {
                        if !nt {
                            // NN with a bf16 operand is not implemented. Fail
                            // loudly instead of computing with wrong bytes.
                            crate::print(b"bf16 NN gemm unimplemented\n");
                            exit_i(9);
                        }
                        if acc {
                            crate::print(b"bf16 acc gemm unimplemented\n");
                            exit_i(9);
                        }
                        let cb16 = self.slot_mut(b) as *const u16;
                        kernels::gemm_nt_bf16(ca, cb16, cc, m, b_rows, k);
                    } else if bt == DT_I8 {
                        // ---- P13 task D: int8 weights, dequantise inside GEMM --
                        //
                        // Layout: C[M,N] = (A[M,K] . W_int8[N,K]) * scale[N]
                        // The per-output-row scale lives in the slot named by
                        // `dt` -- the SAME 4th-operand convention OP_ROPE and
                        // OP_LN already use (Ins has op/fl/dt/dst/a/b with no
                        // spare byte, verified against the repr(C) definition,
                        // so a 5th operand is not available).
                        //
                        // The `off + sc > dc` style guards below are the point:
                        // a mismatched scale length is a SILENT wrong answer if
                        // unchecked, because the kernel would read past the
                        // scale array and multiply by unrelated memory.
                        if !nt {
                            crate::print(b"int8 NN gemm unimplemented\n");
                            exit_i(9);
                        }
                        if acc {
                            crate::print(b"int8 acc gemm unimplemented\n");
                            exit_i(9);
                        }
                        let sslot = ins.dt as usize;
                        if sslot >= self.n_slots || self.slot_dtype(sslot) != DT_F32 {
                            crate::print(b"int8 gemm: scale slot must be f32\n");
                            exit_i(9);
                        }
                        let (sr, sc_) = self.slot_shape(sslot);
                        // One scale per output row, so the scale slot must hold
                        // at least b_rows values. Length mismatch => refuse.
                        if sr * sc_ < b_rows {
                            crate::print(b"int8 gemm: scale shorter than N rows\n");
                            exit_i(9);
                        }
                        let cq = self.slot_mut(b) as *const i8;
                        let cs = self.slot_mut(sslot);
                        kernels::gemm_nt_int8(ca, cq, cs, cc, m, b_rows, k);
                    } else if bt != DT_F32 {
                        crate::print(b"gemm operand dtype unsupported\n");
                        exit_i(9);
                    } else {
                        let cb = self.slot_mut(b);
                        if nt {
                            kernels::gemm_nt(ca, cb, cc, m, b_rows, k, acc);
                        } else {
                            kernels::gemm(ca, cb, cc, m, b_cols, k, acc);
                        }
                    }
                }

                OP_GEMV => {
                    let (m, k) = self.slot_shape(a);
                    let ca = self.slot_mut(a);
                    let cb = self.slot_mut(b);
                    let cc = self.slot_mut(d);
                    for i in 0..m {
                        let mut s = 0.0f32;
                        for p in 0..k {
                            s += *ca.add(i * k + p) * *cb.add(p);
                        }
                        *cc.add(i) = s;
                    }
                }

                OP_EWADD | OP_EWMUL | OP_EWSUB | OP_EWDIV => {
                    let (r, c) = self.slot_shape(a);
                    let (br, bc) = self.slot_shape(b);
                    let ca = self.slot_mut(a);
                    let cb = self.slot_mut(b);
                    let cc = self.slot_mut(d);
                    if br == r && bc == c {
                        // 
                    kernels::ew_op(cc, ca, cb, r * c, ins.op);
                    } else if br == r && bc == 1 {
                        // [M,N] op [M,1]
                        for i in 0..r {
                            let s = *cb.add(i);
                            for j in 0..c {
                                let x = *ca.add(i * c + j);
                                let y = s;
                                *cc.add(i * c + j) = match ins.op {
                                    OP_EWADD => x + y,
                                    OP_EWMUL => x * y,
                                    OP_EWSUB => x - y,
                                    _ => x / y,
                                };
                            }
                        }
                    } else if br == 1 && bc == c {
                        // [M,N] op [1,N]
                            for i in 0..r {
                            for j in 0..c {
                                let x = *ca.add(i * c + j);
                                let y = *cb.add(j);
                                *cc.add(i * c + j) = match ins.op {
                                    OP_EWADD => x + y,
                                    OP_EWMUL => x * y,
                                    OP_EWSUB => x - y,
                                    _ => x / y,
                                };
                            }
                        }
                    } else if br * bc == 1 {
                        // 
                        let y = *cb;
                        for i in 0..r * c {
                            let x = *ca.add(i);
                            *cc.add(i) = match ins.op {
                                OP_EWADD => x + y,
                                OP_EWMUL => x * y,
                                OP_EWSUB => x - y,
                                _ => x / y,
                            };
                        }
                    } else {
                        crate::print(b"EW shape mismatch\n");
                        exit_i(8);
                    }
                }

                OP_ACT => {
                    let (r, c) = self.slot_shape(d);
                    let cd = self.slot_mut(d);
                    let ca = self.slot_mut(a);
                    core::ptr::copy_nonoverlapping(ca, cd, r * c);
                    let kind = if ins.fl == 0 { ACT_RELU } else { ins.fl as u16 };
                    kernels::act_avx2(cd, r * c, kind);
                }

                OP_LN => {
                    // LN dst, src, gamma, bias
                    //   16B dst/a/b bias  ins.dt                    //   rc src                    //   dst LN logits[1,4], h[1,32]                    //   src 32  4                     //   eps LN_EPS 1e-5OP_LN_EPS
                    let (r, c) = self.slot_shape(a);
                    let cd = self.slot_mut(d);
                    let ca = self.slot_mut(a);
                    core::ptr::copy_nonoverlapping(ca, cd, r * c);
                    let g = self.slot_mut(b);
                    let bias = self.slot_mut(ins.dt as usize);
                    kernels::layernorm(cd, g, bias, r, c, self.ln_eps);
                }

                // LN_EPS <  0 LN eps                //  1e-5ESERNN 1e-6 rms_eps                //  LN                 //  dst PRINT  a
            OP_LN_EPS => {
                    self.ln_eps = *self.slot_mut(d);
                }

                // P10: RMSNORM ** opcode**                //
                // OP_RMSNORM = 31` format.rs                 // 3.4 opcode  interp.rs *
                // *  catch-all `exit(5) "unimpl op"`                //
                // bug P10 MiniCPM layer-0                 // ***iniCPM                 // input_layernorm / post_attention_layernorm  RMSNorm                // 11 tools/p10_opcode_gaps.py                 //
                // RMSNormlama **bias**
                //   y = x / sqrt(mean(x^2) + eps) * gamma
                // LN 
                //   * layernorm                 //   *  beta  gamma                 //   * eps  ****                 //
                // LN 3                 //   dst=, a= b=gamma, dt=eps                 //  **** LN st 
                //
            OP_RMSNORM => {
                    let (r, c) = self.slot_shape(a);
                    let cd = self.slot_mut(d);
                    let ca = self.slot_mut(a);
                    let g = self.slot_mut(b);
                    // eps  4  0 LN bias                     // dt 
                    // DT_F32(=0)0  LN 
                    //  LN 
                    let eps = *self.slot_mut(ins.dt as usize);
                    kernels::rmsnorm(cd, ca, g, r, c, eps);
                }

                // ============ P8 BPTT step  ============
                //  PUSH 2D 
                //= [max_steps,  step  POP                 //                  //
                // PUSH dst_stack, src, step        (dst=, a=, b=step 
                //
                //  rc ****t_c                // * step *                // P9  I SHAPE stk_Sprev, 2, 16`  128                 //28 elems/step
                //  2/16                 //
            OP_PUSH => {
                    let (src_r, src_c) = self.slot_shape(a);
                    let (_st_r, st_c) = self.slot_shape(d);
                    let n = src_r * src_c;
                    if n != st_c {
                        crate::print(b"PUSH: src elems != stack row width\n");
                        exit_i(9);
                    }
                    let step = ins.b as usize;
                    let cs = self.slot_mut(a);
                    let cd = self.slot_mut(d);
                    core::ptr::copy_nonoverlapping(cs, cd.add(step * st_c), n);
                }
                // POP dst, stack, step            (dst= a=, b=step 
                //  PUSH
            OP_POP => {
                    let (d_r, d_c) = self.slot_shape(d);
                    let (_st_r, st_c) = self.slot_shape(a);
                    let n = d_r * d_c;
                    if n != st_c {
                        crate::print(b"POP: dst elems != stack row width\n");
                        exit_i(9);
                    }
                    let step = ins.b as usize;
                    let cs = self.slot_mut(a);
                    let cd = self.slot_mut(d);
                    core::ptr::copy_nonoverlapping(cs.add(step * st_c), cd, n);
                }
                // STACK_INIT stack, max_steps      (a=, b=max_steps) 
                OP_STACK_INIT => {
                    let (r, c) = self.slot_shape(a);
                    core::ptr::write_bytes(self.slot_mut(a) as *mut u8, 0, r * c * 4);
                }

                OP_SOFTMAX => {
                    let (r, c) = self.slot_shape(d);
                    let cd = self.slot_mut(d);
                    let ca = self.slot_mut(a);
                    core::ptr::copy_nonoverlapping(ca, cd, r * c);
                    kernels::softmax(cd, r, c, (ins.fl as u16) & FL_CAUSAL != 0);
                }

                OP_OUTER => {
                    // S[i,j] += v[i]k[j]
                    // v [1,D] -> m is the non-1 dim
                    let (vr, vc) = self.slot_shape(a);
                    let (kr, kc) = self.slot_shape(b);
                    let m = if vr == 1 { vc } else { vr }; // v shape
                    let nn = if kr == 1 { kc } else { kr }; // k shape
                    let cs = self.slot_mut(d);
                    let cv = self.slot_mut(a);
                    let ck = self.slot_mut(b);
                    kernels::outer(cs, cv, ck, m, nn, 1.0);
                }

                OP_TOPK_THRESH => {
                    // dst[i] = k  a[i, :]  16                     //  sparse_dyn 
                    let (r, c) = self.slot_shape(a);
                    let ca = self.slot_mut(a);
                    let cd = self.slot_mut(d);
                    let k = (ins.fl as usize).min(c);
                    for i in 0..r {
                        *cd.add(i) = kernels::topk_thresh(ca.add(i * c), c, k);
                    }
                }
                // P13 task E: layer_base += round(a[0])   (bytes)
                //
                // This is what turns the unrolled 24 layers into a LOOP_CNT
                // loop: one layer template, and the base advances by the
                // per-layer constant stride each iteration.
                //
                // The delta is read from a scalar slot so the stride stays an
                // artifact-level fact rather than a hardcoded engine constant
                // (the measured stride is 28,383,232 B for this model, but the
                // engine has no business knowing that).
                //
                // Bounds are enforced at every const access (see const_ptr), so
                // a runaway loop exits(9) rather than reading activations as if
                // they were weights.
                OP_LAYERB => {
                    // The delta slot is `a`. The assembler pins that encoding
                    // explicitly for this opcode (see the LAYERB rule in
                    // asm/src/main.rs): a single-register instruction would
                    // otherwise put it in `dst` and leave `a` = 0, making the
                    // engine read slot 0 forever. That cost two cycles, and the
                    // 1-layer loop is bit-exact either way, so the bug only
                    // appeared at 2+ layers.
                    //
                    // The delta is SIGNED: the builder restores the base after
                    // the loop by adding a NEGATIVE stride, so constants that
                    // are not per-layer (final_norm) are read at their real
                    // offsets. Clamping negatives to 0 made that restore a
                    // silent no-op.
                    //
                    // Bound safety lives in const_ptr: overshoot exits(9) rather
                    // than reading unrelated bytes as weights.
                    let d0 = *self.slot_mut(a);
                    if d0 >= 0.0 {
                        self.layer_base = self.layer_base.saturating_add(d0 as usize);
                    } else {
                        let back = (-d0) as usize;
                        self.layer_base = self.layer_base.saturating_sub(back);
                    }
                }
                // dst = max(a, b)  |h| = max(h, -h)
            OP_EWMAX => {
                    let (r, c) = self.slot_shape(a);
                    let ca = self.slot_mut(a);
                    let cb = self.slot_mut(b);
                    let cd = self.slot_mut(d);
                    for i in 0..r * c {
                        let x = *ca.add(i);
                        let y = *cb.add(i);
                        *cd.add(i) = if x > y { x } else { y };
                    }
                }
                // GEMASK dst, a dst = (a >= 0)  1 : 0
                //   a (s - )top-k
            OP_GEMASK => {
                    let (r, c) = self.slot_shape(a);
                    let ca = self.slot_mut(a);
                    let cd = self.slot_mut(d);
                    for i in 0..r * c {
                        *cd.add(i) = if *ca.add(i) >= 0.0 { 1.0 } else { 0.0 };
                    }
                }

                OP_MASK_MUL => {
                    let (r, c) = self.slot_shape(d);
                    let cd = self.slot_mut(d);
                    let ca = self.slot_mut(a);
                    let cb = self.slot_mut(b);
                    let (_, cb_cols) = self.slot_shape(b);
                    for i in 0..r {
                        for j in 0..c {
                            let m = if cb_cols == 1 {
                                *cb.add(i)
                            } else {
                                *cb.add(i * cb_cols + j)
                            };
                            *cd.add(i * c + j) = *ca.add(i * c + j) * m;
                        }
                    }
                }

                OP_JMP => {
                    // P10  `dst``a`                    //
                    //  `a``a`  0                     // ** 0**`dst`                    //  ins 3  ins 0                    //                     //  ins 0                     ip = d;
                }
                OP_CMPLE => {
                    let (_, _c) = self.slot_shape(a);
                    let ca = self.slot_mut(a);
                    let cb = self.slot_mut(b);
                    // [0] <= b[0] dst
                    if *ca <= *cb {
                        ip = d;
                    }
                }

                // LOOP_CNT ctr, @label P10                //   tr = ctr - 1; if ctr > 0 { ip = dst }
                //
                // +  0 ***                //   * for layer in 0..24` `0..6` 
                //                     //   *                 //   * HAPE ctr,1,1
                //     **.vmb **                //
                // 
                //   *  [1,N]  
                //   *  if***
                //     .vmb
            OP_LOOP_CNT => {
                    let (r, c) = self.slot_shape(a);
                    if r != 1 {
                        // 48 KB                         // format.rs /  opcode 
                        crate::print(b"LOOP_CNT: ctr must be [1,N]\n");
                        exit_i(9);
                    }
                    if d >= ip {
                        crate::print(b"LOOP_CNT: target must be back\n");
                        exit_i(9);
                    }
                    let ca = self.slot_mut(a);
                    //  [1,N] 0  CMPLE 
                    let _ = c;
                    let nv = *ca - 1.0;
                    *ca = nv;
                    if nv > 0.0 {
                        ip = d;
                    }
                }
                // ---- P10-B:  + RoPE ----
                //
                // SIN/COSRoPE                 // RoPE                 //   ***
                //   xp_f 6.5e-4
                //    Python
            OP_SIN | OP_COS => {
                    let (r, c) = self.slot_shape(a);
                    let ca = self.slot_mut(a);
                    let cd = self.slot_mut(d);
                    let n = r * c;
                    for i in 0..n {
                        let v = *ca.add(i);
                        *cd.add(i) = if ins.op == OP_SIN {
                            kernels::sin_f(v)
                        } else {
                            kernels::cos_f(v)
                        };
                    }
                }

                // ROPE dst, src, theta                 //   dst   =  [S,hd]
                //   a     =   [S,hd] k head                 //   b     = theta  0 5e6                //   dt    =  pos0 r  pos0 + r                //
                // HF rotate_half
                //   inv_freq_i = theta^(-2i/hd) = exp(-2i/hd * ln(theta))
                //   ang(r,i)   = (pos0 + r) * inv_freq_i
                //   dst[r,i]        = src[r,i]*cos(ang) - src[r,i+hd/2]*sin(ang)
                //   dst[r,i+hd/2]   = src[r,i]*sin(ang) + src[r,i+hd/2]*cos(ang)
                //
                // cos/sin                 // = S  hd  4  2 B=131072d=128  134 MB
                //  MiniCPM max_position  131072                //  theta 
                // ****LOOP_CNT
            OP_ROPE => {
                    let (r, c) = self.slot_shape(a);
                    if c < 2 || (c & 1) != 0 {
                        // /
                        crate::print(b"ROPE: hd must be even >= 2\n");
                        exit_i(9);
                    }
                    let hd = c;
                    let half = hd / 2;
                    let theta = *self.slot_mut(b);
                    if theta <= 0.0 {
                        crate::print(b"ROPE: theta must be > 0\n");
                        exit_i(9);
                    }
                    // inv_freq_i = theta^(-2i/hd) = exp(i * step)
                    let step = -2.0 / (hd as f32) * kernels::ln_f(theta);
                    let pos0 = ins.dt as f32;
                    let cs = self.slot_mut(a);
                    let cd = self.slot_mut(d);
                    const TWO_PI: f32 = 6.283_185_5;
                    for row in 0..r {
                        let pos = pos0 + row as f32;
                        let sr = cs.add(row * hd);
                        let dr = cd.add(row * hd);
                        for i in 0..half {
                            let inv = kernels::exp_f(i as f32 * step);
                            // KEY (P10, measured): reduce the POSITION modulo
                            // this component's period BEFORE forming the angle,
                            // so the argument to sin/cos stays < 2*pi.
                            //
                            // Why: computing `ang = pos * inv` directly and
                            // feeding it to sin_f gives ang = 1.3e5 at
                            // pos=131072, i=0. f32 ulp at 1e5 is 7.8e-3, so
                            // after reducing a big angle the remainder can only
                            // be accurate to ~ulp(ang) -- INDEPENDENT of how
                            // good the reduction is. Measured sin error
                            // 9.75e-03; ten different (HI bits, n split)
                            // combinations all converged to 7.5e-3.
                            //
                            // Reducing by the period puts the angle in
                            // [0, 2*pi), where the same sin_f is good to
                            // ~1e-7 -- five orders of magnitude better.
                            //
                            // period_i = 2*pi / inv_freq_i, so
                            //     ang = (pos mod period) * inv_freq
                            // For large i the period exceeds any position we
                            // will see, so no reduction is needed there.
                            let period = TWO_PI / inv;
                            let pr = if period >= 1.0e6 {
                                // Period far beyond the position range: the
                                // angle is already small, skip the modulo.
                                pos
                            } else {
                                pos - period * kernels::floor_f(pos / period)
                            };
                            let ang = pr * inv;
                            let co = kernels::cos_f(ang);
                            let si = kernels::sin_f(ang);
                            let x1 = *sr.add(i);
                            let x2 = *sr.add(i + half);
                            *dr.add(i) = x1 * co - x2 * si;
                            *dr.add(i + half) = x1 * si + x2 * co;
                        }
                    }
                }

                OP_SCALE => {
                    let (r, c) = self.slot_shape(d);
                    let cd = self.slot_mut(d);
                    let ca = self.slot_mut(a);
                    core::ptr::copy_nonoverlapping(ca, cd, r * c);
                    // b  0 
                    let s = *self.slot_mut(b);
                    for i in 0..r * c {
                        *cd.add(i) *= s;
                    }
                }
                OP_COPY => {
                    let (r, c) = self.slot_shape(a);
                    let cd = self.slot_mut(d);
                    let ca = self.slot_mut(a);
                    core::ptr::copy_nonoverlapping(ca, cd, r * c);
                }
                OP_SAVE => {
                    let (r, c) = self.slot_shape(a);
                    let cd = self.slot_mut(d);
                    let ca = self.slot_mut(a);
                    core::ptr::copy_nonoverlapping(ca, cd, r * c);
                }
                OP_LOAD => {
                    let (r, c) = self.slot_shape(a);
                    let cd = self.slot_mut(d);
                    let ca = self.slot_mut(a);
                    core::ptr::copy_nonoverlapping(ca, cd, r * c);
                }
                OP_PRINT => {
                    // PRINT <slot>dst dst
                    let (r, c) = self.slot_shape(d);
                    print_f32(self.slot_mut(d), r * c);
                }
                OP_BIAS => {
                    // dst[M,N] += bias[N]
                    let (r, c) = self.slot_shape(d);
                    let cd = self.slot_mut(d);
                    let cb = self.slot_mut(a);
                    for i in 0..r {
                        for j in 0..c {
                            *cd.add(i * c + j) += *cb.add(j);
                        }
                    }
                }
                OP_SHAPE => {} // 
                // ---- P4/P5  ----
                // EXP: dst[i] = exp(src[i])
                OP_EXP => {
                    let (r, c) = self.slot_shape(a);
                    let ca = self.slot_mut(a);
                    let cd = self.slot_mut(d);
                    for i in 0..r * c {
                        *cd.add(i) = kernels::exp_f(*ca.add(i));
                    }
                }
                // NEG: dst[i] = -src[i]
                OP_NEG => {
                    let (r, c) = self.slot_shape(a);
                    let ca = self.slot_mut(a);
                    let cd = self.slot_mut(d);
                    for i in 0..r * c {
                        *cd.add(i) = -*ca.add(i);
                    }
                }
                // LOG
                OP_LOG => {
                    let (r, c) = self.slot_shape(a);
                    let ca = self.slot_mut(a);
                    let cd = self.slot_mut(d);
                    for i in 0..r * c {
                        *cd.add(i) = kernels::ln_f(*ca.add(i));
                    }
                }
                // CLAMP: dst = clamp(src, lo, hi)o/hi b 
                OP_CLAMP => {
                    let (r, c) = self.slot_shape(a);
                    let ca = self.slot_mut(a);
                    let cd = self.slot_mut(d);
                    let cb = self.slot_mut(b);
                    let (lo, hi) = (*cb, *cb.add(1));
                    for i in 0..r * c {
                        let v = *ca.add(i);
                        *cd.add(i) = if v < lo {
                            lo
                        } else if v > hi {
                            hi
                        } else {
                            v
                        };
                    }
                }
                // EMB:  dst[1,C] = table[row, C]
                //   a=(rows x cols), b=)b
                //
                // P13 task A: this is what makes a prompt changeable WITHOUT
                // reassembling the model.
                //
                // Why it matters: before this, the prompt rows were extracted
                // HOST-SIDE and shipped as a small f32 `prompt_emb` constant
                // plus `COPY h, prompt_emb`. The token ids were therefore baked
                // into the 1.76 GB .vmb, so every new prompt meant a full
                // ~106 s reassembly. Embedding is a lookup, not a compilation.
                //
                // The gap was NOT "EMB is missing" (it was already here) -- it
                // was that the table was read as f32 (`slot_mut(a)`), while
                // `embed_tokens` is **bf16 [130560,1536] = 401 MB**. Feeding a
                // bf16 table used to trip the load-time audit (exit 9); widening
                // it to f32 on disk would cost 802 MB instead of 401 MB, i.e.
                // exactly the "don't widen bf16 to f32" rule P12 established.
                //
                // So: keep the table bf16 in place and widen on read, the same
                // way OP_GEMM does. The conversion is exact and free -- bf16 is
                // the top 16 bits of f32, so `(u32)h << 16` needs no FPU work.
                //
                // WHY AVX2 IS BACK (P13, after the budget re-baseline):
                //   An earlier revision used a scalar loop ONLY to fit the old
                //   53312 B budget -- the vector path measured +512 B and the
                //   headroom was 64 B, so the bytes bought the feature. That was
                //   a size-driven decision, not a performance one, and the user
                //   has since granted +10 KB and asked that performance not be
                //   traded away for bytes. So the vector path is restored.
                //   It reuses `kernels::bf16_widen_16`, the SAME primitive the
                //   bf16 GEMM uses, so there is no second hand-derived guess at
                //   AVX2 lane semantics -- P12 got that wrong twice, and
                //   tests/src/shufprobe.rs is what pinned the layout down.
                OP_EMB => {
                    let (r, c) = self.slot_shape(a);
                    let cd = self.slot_mut(d);
                    let rowf = *self.slot_mut(b);
                    // O6-A: the row index is a FLOAT that the program is free to
                    // compute -- and after O6 it IS computed, because the
                    // generation loop feeds OP_TOPK_IDX's result straight back
                    // into `st_gen`, which SLICE/EMB then read as a table row.
                    //
                    // So validate the float BEFORE casting it. `as usize` on a
                    // negative or NaN f32 SATURATES TO 0 in Rust, so
                    // `*slot as usize` alone turns "index -1" and "index NaN"
                    // into a perfectly plausible read of row 0 -- a wrong
                    // embedding with no error, which is the silent-degradation
                    // class this project converts to hard failures. Checking
                    // after the cast cannot see any of this: the information is
                    // already gone.
                    //
                    // The old code also DISCARDED the row count (`let (_r, c)`)
                    // and never bounded `row`. An out-of-range row read past the
                    // table into whatever the const pool holds next and returned
                    // plausible numbers (measured by the verifier: token id 99
                    // against a 4-row table exited 0 and printed zeros).
                    // Both the unbounded read and the silent saturation are
                    // rejected here, naming the reason.
                    if !(rowf >= 0.0) || rowf >= r as f32 {
                        // `!(rowf >= 0.0)` is deliberately not `rowf < 0.0`:
                        // NaN fails every ordered comparison, so the negated
                        // form catches NaN AND negatives in one test.
                        crate::print(b"EMB: row index out of range\n");
                        crate::print(b" row=");
                        print_f32(self.slot_mut(b), 1);
                        crate::print(b" table_rows=");
                        print_u32(r as u32);
                        crate::print(b"\n");
                        exit_i(9);
                    }
                    let row = rowf as usize;
                    // Fractional index: `as usize` truncates, so 2.5 would read
                    // row 2 -- an off-by-one that produces a plausible wrong
                    // embedding from an input that simply is not a row number.
                    // OP_TOPK_IDX always yields an exact integer, so requiring
                    // the integral form costs the O6 path nothing and closes the
                    // last way a bad index can slip through silently.
                    if row as f32 != rowf {
                        crate::print(b"EMB: row index is not an integer\n");
                        crate::print(b" row=");
                        print_f32(self.slot_mut(b), 1);
                        crate::print(b"\n");
                        exit_i(9);
                    }
                    let dt = self.slot_dtype(a);
                    if dt == DT_BF16 {
                        // bf16 table: widen in place. `(u32)h << 16` is exact
                        // (bf16 IS the top half of f32) and needs no FPU.
                        let src = (self.slot_mut(a) as *const u16).add(row * c);
                        let mut i = 0usize;
                        // 16 elements per iteration (one ymm load) via the same
                        // primitive the bf16 GEMM uses -- see the note above on
                        // why the lane layout is not re-derived here.
                        #[cfg(target_arch = "x86_64")]
                        {
                            if HAS_AVX2 {
                                while i + 16 <= c {
                                    let (w0, w1) = kernels::bf16_widen_16(src.add(i));
                                    core::arch::x86_64::_mm256_storeu_ps(cd.add(i), w0);
                                    core::arch::x86_64::_mm256_storeu_ps(cd.add(i + 8), w1);
                                    i += 16;
                                }
                            }
                        }
                        while i < c {
                            *cd.add(i) = f32::from_bits((*src.add(i) as u32) << 16);
                            i += 1;
                        }
                    } else {
                        if dt != DT_F32 {
                            crate::print(b"EMB table dtype unsupported\n");
                            exit_i(9);
                        }
                        let ca = self.slot_mut(a);
                        for i in 0..c {
                            *cd.add(i) = *ca.add(row * c + i);
                        }
                    }
                }
                // BMM: 3D [B, M, K] x [B, K, N] [B, M, N]
                //   3D
                //   a=[B,M,K]  rows=B*M, cols=K=[B,K,N]  rows=K, cols=N
                OP_BMM => {
                    let (abm, k) = self.slot_shape(a);
                    let (bk, nn) = self.slot_shape(b);
                    let ca = self.slot_mut(a);
                    let cb = self.slot_mut(b);
                    let cc = self.slot_mut(d);
                    if bk == k {
                        //  batch B  KN 
                        let m = abm;
                        kernels::gemm(ca, cb, cc, m, nn, k, false);
                    } else {
                        // batch=[B*M,K], b=[B*K,N]
                        let m = abm / (bk / k);
                        let bs = bk / k;
                        for bi in 0..bs {
                            kernels::gemm(
                                ca.add(bi * m * k),
                                cb.add(bi * k * nn),
                                cc.add(bi * m * nn),
                                m,
                                nn,
                                k,
                                false,
                            );
                        }
                    }
                }
                // TRANSPOSE: dst[C,M] = src[M,C]
                OP_TRANSPOSE => {
                    let (m, c) = self.slot_shape(a);
                    //  >= m*c `j*m+i` ****                    // ***,c****                    //                     // 9I `TRANSPOSE gtmp1, g_tm_w1pT` gtmp1                     //   [1,16] [16,16]
                    //   
                    let (dr, dc) = self.slot_shape(d);
                    if dr * dc < m * c {
                        crate::print(b"TRANSPOSE: dst smaller than src\n");
                        exit_i(9);
                    }
                    let ca = self.slot_mut(a);
                    let cd = self.slot_mut(d);
                    for i in 0..m {
                        for j in 0..c {
                            *cd.add(j * m + i) = *ca.add(i * c + j);
                        }
                    }
                }
                // SLICE dst, src, startCol                 //   **dst *  4                 //   dst cols  width
            OP_SLICE => {
                    let (dr, dc) = self.slot_shape(d);
                    let (_sr, sc) = self.slot_shape(a);
                    let off = ins.b as usize;
                    let ca = self.slot_mut(a);
                    let cd = self.slot_mut(d);
                    for i in 0..dr {
                        for j in 0..dc {
                            *cd.add(i * dc + j) = *ca.add(i * sc + off + j);
                        }
                    }
                }
                // CONCAT dst, a, b
            OP_CONCAT => {
                    let (r, ac) = self.slot_shape(a);
                    let (_rb, bc) = self.slot_shape(b);
                    let pp = self.slot_mut(a);
                    let qq = self.slot_mut(b);
                    let pd = self.slot_mut(d);
                    for i in 0..r {
                        for j in 0..ac {
                            *pd.add(i * (ac + bc) + j) = *pp.add(i * ac + j);
                        }
                        for j in 0..bc {
                            *pd.add(i * (ac + bc) + ac + j) = *qq.add(i * bc + j);
                        }
                    }
                }
                // WRITECOL dst, src, startCol src  dst 
                OP_WRITECOL => {
                    let (dr, dc) = self.slot_shape(d);
                    let (_sr, sc) = self.slot_shape(a);
                    let off = ins.b as usize;
                    // off + sc > dc` `i*dc + off + j` **                    //  head/step                     // P9  I 8 bug  4 
                    // ***                    // 9WRITECOL dSprev[8,16], dShScaled[8,8], 0..15`                     //   off>=9 7 
                    //   
                    if off + sc > dc {
                        crate::print(b"WRITECOL: off + src.cols exceeds dst row width\n");
                        exit_i(9);
                    }
                    let ca = self.slot_mut(a);
                    let cd = self.slot_mut(d);
                    for i in 0..dr {
                        for j in 0..sc {
                            *cd.add(i * dc + off + j) = *ca.add(i * sc + j);
                        }
                    }
                }
                // SUM:  dst[rows,1]
                OP_SUM => {
                    let (r, c) = self.slot_shape(a);
                    let ca = self.slot_mut(a);
                    let cd = self.slot_mut(d);
                    for i in 0..r {
                        let mut s = 0.0f32;
                        for j in 0..c {
                            s += *ca.add(i * c + j);
                        }
                        *cd.add(i) = s;
                    }
                }

                // ================ O6-A: argmax primitives ================
                //
                // WHY THESE TWO EXIST AT ALL: O6 is autoregressive generation,
                // and generation is `feed decode's argmax back as the next
                // token`. Until now there was no argmax in the engine at all:
                // OP_MAX and OP_TOPK_IDX were defined in format.rs but had no
                // arm here, so the O6 loop could not be written even in
                // principle. `tools/p10_opcode_gaps.py` listed both as
                // "user-visible trap".
                //
                // The trap had three layers, not one, and all three are fixed:
                //   1. no engine arm here -> caught by the catch-all exit(5);
                //   2. the assembler PANICKED at assembly time (MAX /
                //      TOPK_IDX were in asm/src/main.rs UNIMPLEMENTED_OPS);
                //   3. no shape inference, so the output slot stayed 0x0 and
                //      the result was silently dropped.
                // Layer 3 is the dangerous one -- it does not fail, it makes
                // the generator emit the same token forever. See the 0x0 guard
                // below and the OP_MAX / OP_TOPK_IDX arms in `infer_shapes`.

                // MAX dst, src -- row-wise max reduction, dst[rows,1].
                //
                // The FIRST element initialises the running max; there is no
                // `-inf` seed and, more importantly, no `0.0` seed. Seeding
                // with 0.0 is the tempting one-liner and it is wrong for every
                // all-negative row: a row of attention scores or logits that
                // happens to be entirely negative would report 0.0 as its max,
                // i.e. a plausible-looking wrong number with no error. That is
                // exactly the silent-degradation class this project converts to
                // hard failures, so the accumulator is never seeded with a
                // constant at all.
                //
                // `c == 0` (an unshaped / 0x0 source) leaves dst untouched
                // unless it is itself 0x0; the explicit guard below rejects
                // that case loudly instead of looping zero times and returning
                // a stale activation as if it were a maximum.
                OP_MAX => {
                    let (r, c) = self.slot_shape(a);
                    let (dr, dc) = self.slot_shape(d);
                    if c == 0 || dr == 0 || dc == 0 {
                        crate::print(b"MAX: src or dst slot has width 0\n");
                        exit_i(9);
                    }
                    if dr != r || dc != 1 {
                        crate::print(b"MAX: dst must be [rows,1]\n");
                        crate::print(b" src=");
                        print_u32(r as u32);
                        crate::print(b"x");
                        print_u32(c as u32);
                        crate::print(b" dst=");
                        print_u32(dr as u32);
                        crate::print(b"x");
                        print_u32(dc as u32);
                        crate::print(b"\n");
                        exit_i(9);
                    }
                    let ca = self.slot_mut(a);
                    let cd = self.slot_mut(d);
                    for i in 0..r {
                        let row = ca.add(i * c);
                        let mut m = *row;
                        for j in 1..c {
                            let v = *row.add(j);
                            if v > m {
                                m = v;
                            }
                        }
                        *cd.add(i) = m;
                    }
                }

                // TOPK_IDX dst, src -- argmax over row 0, dst[1,1] = index as f32.
                //
                // Two deliberate choices, both required by O6:
                //
                // * The index is written as an f32 NUMBER, matching how OP_EMB
                //   reads its row index (`*self.slot_mut(b) as usize`) and how
                //   the runtime already writes u32-as-f32 for token ids. A
                //   separate "index dtype" would be a second convention for the
                //   same thing. Exact for any index < 2^24; the MiniCPM vocab
                //   is 73,448, so this is lossless in practice, and an index
                //   that large cannot arise from a slot that fits in memory.
                //
                // * Comparison is NUMERIC f32 (IEEE `>`), never raw i32/bit
                //   patterns. Bit-pattern comparison looks equivalent because
                //   IEEE-754 is monotonic over non-negative floats, but it
                //   inverts the entire negative range, so an all-negative row
                //   (log-probs, masked attention scores) would pick the WORST
                //   element and return it as an argmax with no error.
                //
                // Ties break to the LOWEST index because `>` (not `>=`)
                // refuses to displace the incumbent. That is what makes
                // generation reproducible: a row with a repeated maximum must
                // not depend on which element the loop happened to visit last.
                //
                // NaN is not special-cased: the `if v > m` form propagates the
                // same way it does everywhere else in the engine. Worth stating
                // because "index 0 wins over NaN" is a real upstream behaviour
                // (torch.argmax) that this does NOT reproduce -- but neither
                // does the rest of this engine, and inventing a NaN rule here
                // would be a guess, not a measurement.
                OP_TOPK_IDX => {
                    let (r, c) = self.slot_shape(a);
                    let (dr, dc) = self.slot_shape(d);
                    if r == 0 || c == 0 || dr == 0 || dc == 0 {
                        crate::print(b"TOPK_IDX: src or dst slot has width 0\n");
                        exit_i(9);
                    }
                    if dr != 1 || dc != 1 {
                        crate::print(b"TOPK_IDX: dst must be [1,1]\n");
                        crate::print(b" src=");
                        print_u32(r as u32);
                        crate::print(b"x");
                        print_u32(c as u32);
                        crate::print(b" dst=");
                        print_u32(dr as u32);
                        crate::print(b"x");
                        print_u32(dc as u32);
                        crate::print(b"\n");
                        exit_i(9);
                    }
                    let ca = self.slot_mut(a);
                    let cd = self.slot_mut(d);
                    let mut best = 0usize;
                    let mut mv = *ca;
                    for j in 1..c {
                        let v = *ca.add(j);
                        if v > mv {
                            mv = v;
                            best = j;
                        }
                    }
                    *cd = best as f32;
                }

                // ================ P16: connection sparsity ================
                //
                // Two principles, two opcodes, ONE shared quantity:
                //     s_i = w_i * x_i        (connection activation strength)
                //   (1) connection sparsity     -- each neuron keeps only its top-k
                //       connections (k0 = 20)          => OP_TOPK_BY_STRENGTH
                //   (2) connection-SPARSE activation -- only the surviving
                //       connections are computed; zero connections are SKIPPED,
                //       not multiplied by zero        => OP_SPARSE_MATVEC
                //
                // WHY "SKIPPED" IS NOT PEDANTRY: the previous incarnation of this
                // model (前身项目 sparse_dyn) did "dense GEMM + mask" and its own
                // README records the result -- dense 110ms vs sparse 148ms, i.e.
                // SLOWER. Multiplying by zero saves arithmetic, not time. So the
                // second opcode below does not merely omit zero terms; it COUNTS
                // the multiply-adds it actually performs and leaves the count in
                // the destination slot, so "we skipped them" is falsifiable by
                // measurement instead of being an assertion in a comment.

                // TOPK_BY_STRENGTH dst[L,2], strengths, k
                //   dst[:,0] = surviving slot index, ASCENDING (0 <= idx < L)
                //   dst[:,1] = that connection's strength, unchanged
                //   rows 0..nk  are valid; rows nk..L are zero-filled.
                //
                // DESIGN DECISION -- INDEX PAIRS, NOT IN-PLACE ZEROING.
                // The task left this open. Index pairs win for a reason that is
                // not aesthetic: the obvious alternative (zero out the non-top-k
                // entries in place, then let the matvec multiply by zero) CANNOT
                // be bit-exact. An accumulator that is exactly -0.0 flips to
                // +0.0 when you add (0.0 * w) to it, because IEEE-754 leaves the
                // sign of exact-zero sums unspecified (`-0.0 + 0.0 == +0.0`).
                // A "skip" leaves -0.0 alone; a "multiply by zero" does not.
                // With index pairs the matvec reads ONLY the surviving (src,w)
                // pairs, so the surviving terms are summed in ascending CSR
                // order -- the very same order the dense product uses. That is
                // what makes the equality exact rather than approximate, and it
                // is measured in tools/p16_sparse_ops.py (max|diff| reported).
                //
                // AND the output is a PERMUTATION-FREE FILTER, not a sort:
                // indices come out ascending, which is exactly the order the CSRs
                // already store them in. So `OUT` can be aliased onto `STR` for
                // the common "zero the read tail, use it as the activation"
                // pattern with no scratch slot and no reorder. (If it sorted by
                // strength the caller would need a second slot just to reorder.)
                //
                // k comes from a SLOT (`b`), not from `dt`/`imm`, even though
                // OP_ROPE and OP_TOPK_THRESH both encode a scalar in the
                // instruction. Reason: k0 = 20 is a MODEL HYPERPARAMETER that
                // changes per layer/phase, and OP_TOPK_THRESH's `k=` immediate
                // already forces a reassembly for every change. Encoding it in
                // `dt` would inherit exactly that. A scalar slot costs one
                // 64 B-aligned activation slot and keeps k data-driven; the
                // count `nk` is then also available for OP_SPARSE_MATVEC's nnz
                // measurement without a second instruction.
                //
                // HARD FAILURE ON k > L, deliberately NOT a clamp: in this
                // model k0 = 20 but a layer with fewer than 20 connections would
                // silently degrade to "keep everything", i.e. silently dense --
                // which is the exact class of silent degradation this project
                // converts to hard failures.
                OP_TOPK_BY_STRENGTH => {
                    // Read BOTH shapes first and check both for zero width. A
                    // 0x0 source with a 0x0 destination would otherwise loop zero
                    // times and leave a stale activation reading as a result --
                    // the same trap OP_MAX documents above.
                    let (sr, sc) = self.slot_shape(a);
                    let (dr, dc) = self.slot_shape(d);
                    if sr == 0 || sc == 0 || dr == 0 || dc == 0 {
                        crate::print(b"TOPK_BY_STRENGTH: src or dst slot has width 0\n");
                        exit_i(9);
                    }
                    // P16-5 SEGMENTED mode (fl bit7): run top-k PER ROW.
                    //
                    // WHY THIS EXISTS -- the composition gap, from the other
                    // side. In flat mode this opcode numbers its output 0..L-1
                    // over the whole array it was handed, while OP_SPARSE_MATVEC
                    // numbers connections 0..nnz-1 over the whole CSR. Those two
                    // numberings only agree for the FIRST row of the matrix, so
                    // a flat `sel` cannot drive a CSR walk -- the engine's own
                    // guard caught exactly this ("sel index outside its CSR row,
                    // idx=8 row=[0,6)") on the first gated attempt.
                    //
                    // In segmented mode `a` is `[n, m]` (one row per neuron),
                    // `dst` is `[n, 2k]`, and each row's survivors are emitted
                    // with indices RELATIVE TO THAT ROW, ascending, zero-padded.
                    // OP_SPARSE_MATVEC then reads `o0 + rel`, so the two
                    // numberings agree by construction.
                    //
                    // This keeps the opcode NUMBER and the 16-byte instruction
                    // unchanged; only the meaning of the existing `fl` bit7
                    // differs between 93 and 94, which is safe because a bit is
                    // only ever read by the opcode that owns it.
                    let seg_mode = (ins.fl as u16) & (FL_SELGATE as u16) != 0;
                    if seg_mode {
                        if sc < 1 || dc % 2 != 0 {
                            crate::print(b"TOPK_BY_STRENGTH: segmented dst must be [n,2k]\n");
                            print_u32(dr as u32);
                            crate::print(b"x");
                            print_u32(dc as u32);
                            crate::print(b"\n");
                            exit_i(9);
                        }
                        if dr != sr {
                            crate::print(b"TOPK_BY_STRENGTH: segmented dst rows != src rows\n");
                            crate::print(b" src=");
                            print_u32(sr as u32);
                            crate::print(b" dst=");
                            print_u32(dr as u32);
                            crate::print(b"\n");
                            exit_i(9);
                        }
                        let kseg = dc / 2;
                        let cs = self.slot_mut(a);
                        let ck = self.slot_mut(b);
                        let cd = self.slot_mut(d);
                        let kf = *ck;
                        if !(kf >= 0.0) {
                            crate::print(b"TOPK_BY_STRENGTH: k is negative or NaN\n");
                            exit_i(9);
                        }
                        let k = kf as usize;
                        if k > sc {
                            crate::print(b"TOPK_BY_STRENGTH: k > row width\n");
                            crate::print(b" k=");
                            print_u32(k as u32);
                            crate::print(b" row=");
                            print_u32(sc as u32);
                            crate::print(b"\n");
                            exit_i(9);
                        }
                        if k > kseg {
                            crate::print(b"TOPK_BY_STRENGTH: k > dst segment width\n");
                            crate::print(b" k=");
                            print_u32(k as u32);
                            crate::print(b" seg=");
                            print_u32(kseg as u32);
                            crate::print(b"\n");
                            exit_i(9);
                        }
                        let mut taken = [false; TOPK_MAX_L];
                        if sc > taken.len() {
                            crate::print(b"TOPK_BY_STRENGTH: row width exceeds the bitmap (2048)\n");
                            exit_i(9);
                        }
                        let mut hit = [0usize; TOPK_MAX_L];
                        for r in 0..sr {
                            let row = cs.add(r * sc);
                            let out = cd.add(r * dc);
                            let mut nk = 0usize;
                            for i in 0..sc {
                                taken[i] = false;
                            }
                            for _ in 0..k {
                                let mut best = 0usize;
                                let mut found = false;
                                let mut bestv = 0.0f32;
                                for i in 0..sc {
                                    if taken[i] {
                                        continue;
                                    }
                                    let v = *row.add(i);
                                    if !found || v > bestv {
                                        bestv = v;
                                        best = i;
                                        found = true;
                                    }
                                }
                                if !found {
                                    break;
                                }
                                taken[best] = true;
                                hit[nk] = best;
                                nk += 1;
                            }
                            // ascending by ROW-RELATIVE index (insertion sort;
                            // k is the model's k0, so this is a handful of
                            // comparisons and no sort routine is worth its bytes)
                            for i in 1..nk {
                                let v = hit[i];
                                let mut q = i;
                                while q > 0 && hit[q - 1] > v {
                                    hit[q] = hit[q - 1];
                                    q -= 1;
                                }
                                hit[q] = v;
                            }
                            for i in 0..kseg {
                                if i < nk {
                                    *out.add(i * 2) = hit[i] as f32;
                                    *out.add(i * 2 + 1) = *row.add(hit[i]);
                                } else {
                                    // Zero-fill the tail: a stale row from a
                                    // previous frame would be read as real
                                    // survivors by the matvec.
                                    *out.add(i * 2) = 0.0;
                                    *out.add(i * 2 + 1) = 0.0;
                                }
                            }
                        }
                        // No `continue` needed: the flat path below is guarded by
                        // the matching `else`-style early exit chosen here.
                    } else {
                    let l = sr * sc;
                    if dc != 2 {
                        crate::print(b"TOPK_BY_STRENGTH: dst must be [L,2]\n");
                        crate::print(b" src=");
                        print_u32(sr as u32);
                        crate::print(b"x");
                        print_u32(sc as u32);
                        crate::print(b" dst=");
                        print_u32(dr as u32);
                        crate::print(b"x");
                        print_u32(dc as u32);
                        crate::print(b"\n");
                        exit_i(9);
                    }
                    // The OUTPUT must be able to hold L index pairs: it is a
                    // 1:1 filter, so it can never need more rows than the
                    // source has entries. A smaller dst is a program error.
                    if dr < l {
                        crate::print(b"TOPK_BY_STRENGTH: dst needs L rows; L=");
                        print_u32(l as u32);
                        crate::print(b" dst=");
                        print_u32(dr as u32);
                        crate::print(b"x");
                        print_u32(dc as u32);
                        crate::print(b"\n");
                        exit_i(9);
                    }
                    let cs = self.slot_mut(a);
                    let ck = self.slot_mut(b);
                    let cd = self.slot_mut(d);
                    let kf = *ck;
                    // NaN / negative k: `as usize` on NaN saturates to 0 in
                    // Rust, which would silently mean "keep nothing". Reject it
                    // explicitly -- it is a malformed k, not a small k.
                    if !(kf >= 0.0) {
                        crate::print(b"TOPK_BY_STRENGTH: k is negative or NaN\n");
                        exit_i(9);
                    }
                    let k = kf as usize;
                    if k > l {
                        crate::print(b"TOPK_BY_STRENGTH: k > L\n");
                        crate::print(b" k=");
                        print_u32(k as u32);
                        crate::print(b" L=");
                        print_u32(l as u32);
                        crate::print(b"\n");
                        exit_i(9);
                    }
                    // Selection = k iterations of "max of what is left", with a
                    // `taken` bitmap. Choosing by `>` (never `>=`) makes ties
                    // break to the LOWEST index -- the same rule OP_TOPK_IDX
                    // uses, and the reason generation stays reproducible.
                    //
                    // The selection order is by DESCENDING STRENGTH (that is
                    // what "max of what is left" produces). The OUTPUT order is
                    // different: ASCENDING BY INDEX. The emission loop below
                    // sorts the survivors before writing them, and that sort is
                    // load-bearing rather than cosmetic:
                    //
                    //   * it is what makes the pairs directly usable as a CSR
                    //     adjacency list, so a host/tool building the P16-1
                    //     structure does not need a scratch slot to reorder;
                    //   * it is what OP_SPARSE_MATVEC's contract assumes when it
                    //     says surviving terms are summed in ascending CSR order,
                    //     which is in turn what makes the product bit-exact
                    //     against a dense reference.
                    //
                    // CAUGHT BY MEASUREMENT, not by review: the first version
                    // selected correctly and emitted in strength order, and
                    // tools/p16_sparse_ops.py's D4 control reported
                    // `indices=[4, 0] ascending=False`. The strengths were
                    // right; the promised order was not.
                    let mut taken = [false; TOPK_MAX_L];
                    if l > taken.len() {
                        crate::print(b"TOPK_BY_STRENGTH: L exceeds the selection bitmap (2048)\n");
                        exit_i(9);
                    }
                    let mut nk = 0usize;
                    // Survivors are collected into a small stack buffer first
                    // (indices only -- the strengths are re-read from `cs`,
                    // which stays valid because selection never mutates it),
                    // then emitted ASCENDING BY INDEX.
                    //
                    // WHY THE BUFFER IS 2048 AND NOT "whatever is convenient":
                    // the first version used [usize; 4096] for this and for
                    // `taken`, i.e. ~33 KB of stack in one frame. The engine
                    // ships a DELIBERATE `#[no_mangle] __chkstk` stub that
                    // returns immediately (crate::__chkstk, to avoid pulling in
                    // the CRT), so a frame that large never touches the guard
                    // page and the process dies with 0xC0000005 instead of a
                    // diagnostic. Measured: exit=3221225477, no output at all.
                    // Bounding the selection to 2048 connections costs 12 KB of
                    // headroom and keeps the frame small. A model needing more
                    // gets the explicit hard failure above rather than a crash.
                    let mut hit = [0usize; TOPK_MAX_L];
                    for _ in 0..k {
                        let mut best = 0usize;
                        let mut found = false;
                        let mut bestv = 0.0f32;
                        for i in 0..l {
                            if taken[i] {
                                continue;
                            }
                            let v = *cs.add(i);
                            if !found || v > bestv {
                                bestv = v;
                                best = i;
                                found = true;
                            }
                        }
                        if !found {
                            break;
                        }
                        taken[best] = true;
                        hit[nk] = best;
                        nk += 1;
                    }
                    // Insertion sort. k is the model's k0 (20 here), so this is
                    // a handful of comparisons; a full sort routine would be
                    // more code than data, and the engine's size budget is a
                    // first-class constraint (see format.rs / the README's 30KB
                    // discussion). Selection already visited ks in descending
                    // strength, so the input is not sorted by index and a plain
                    // "is it already sorted" shortcut would not pay.
                    for i in 1..nk {
                        let v = hit[i];
                        let mut j = i;
                        while j > 0 && hit[j - 1] > v {
                            hit[j] = hit[j - 1];
                            j -= 1;
                        }
                        hit[j] = v;
                    }
                    for i in 0..nk {
                        *cd.add(i * 2) = hit[i] as f32;
                        *cd.add(i * 2 + 1) = *cs.add(hit[i]);
                    }
                    // Zero the unused tail so a reused slot does not leak the
                    // previous frame's indices -- those would be read as real
                    // connections by OP_SPARSE_MATVEC and quietly corrupt y.
                    for i in nk..l {
                        *cd.add(i * 2) = 0.0;
                        *cd.add(i * 2 + 1) = 0.0;
                    }
                    } // end of the NON-segmented (flat) path
                }

                // SPARSE_MATVEC dst[n+1], offsets[n+1], src[nnz]   (w in fl, x in dt)
                //
                //   dst[j]         = sum over kept CSRs t of  w[t] * x[src[t]]   (j = owner(t))
                //   dst[n_neurons] = multiply-adds ACTUALLY PERFORMED  (if FL_NNZ)
                //
                // SEMANTICS OF "KEPT" -- this is the whole point of P16-2, so it
                // is stated exactly: a CSR entry t is KEPT iff
                //     offsets[owner(t)] <= t < offsets[owner(t)+1]
                // i.e. it survived OP_TOPK_BY_STRENGTH and therefore has a
                // nonzero weight. Entries not covered by the CSR structure are
                // zero connections. They are not visited, not multiplied, not
                // added -- and dst[n_neurons] proves it by counting.
                //
                // WHY THE NNZ IS WRITTEN TO THE SLOT RATHER THAN PRINTED: the
                // engine deliberately has no core::fmt (P13 reclaimed 3 KB by
                // hand-rolling print_f32), and a printed number cannot be
                // compared numerically by a script. Reading it out of the
                // activation slot and asserting `nnz == dense_count - 0` in
                // tools/p16_sparse_ops.py makes the skip claim FAILABLE: corrupt
                // the CSR so it covers all n^2 pairs and the number changes.
                //
                // WITHOUT OP_SCATTER, deliberately: OP_GATHER and OP_SCATTER are
                // both still in asm/src/main.rs UNIMPLEMENTED_OPS, i.e. the
                // assembler refuses them outright (verified by reading the list,
                // not by trying). More importantly a scatter needs an index
                // slot and a second pass; a CSR walk needs neither.
                OP_SPARSE_MATVEC => {
                    let fl8 = ins.fl;
                    let wslot = (fl8 & SPARSE_WSLOT_MASK) as usize;
                    let want_nnz = (fl8 & FL_NNZ) != 0;
                    let xslot = ins.dt as usize;
                    // P16-5: SEL-GATED mode (fl bit7). See format.rs FL_SELGATE
                    // for the three design reasons. When OFF, everything below
                    // is byte-for-byte what task-8 shipped -- the gate only adds
                    // a second branch, it does not perturb the first.
                    let selgate = (fl8 & FL_SELGATE) != 0;

                    let (orows, ocols) = self.slot_shape(a);
                    let (srows, scols) = self.slot_shape(b);
                    let (dr, dc) = self.slot_shape(d);
                    if orows == 0 || ocols == 0 || srows == 0 || scols == 0 {
                        crate::print(b"SPARSE_MATVEC: offsets or src slot has width 0\n");
                        exit_i(9);
                    }
                    if dr == 0 || dc == 0 {
                        crate::print(b"SPARSE_MATVEC: dst slot has width 0\n");
                        exit_i(9);
                    }
                    // In gated mode the candidate list lives in the slot
                    // IMMEDIATELY AFTER the CSR src slot, i.e. `b + 1`.
                    //
                    // WHY ADJACENCY RATHER THAN A 5th OPERAND -- this is the one
                    // real design constraint of P16-5, so it is stated plainly:
                    // the instruction has exactly four usable slot fields after
                    // `dst` (a, b, fl, dt), and gated mode needs five arrays
                    // (offsets, src, weights, sel, x). Something must give.
                    // Rather than widen the 16-byte instruction or burn a new
                    // opcode number, the sel slot is named by CONVENTION as
                    // `src + 1`, which is exactly the pattern this project
                    // already uses twice:
                    //     LN_BWD    -> gamma / dg / db   are consecutive slots
                    //     OUTER_BWD -> k / dk            are consecutive slots
                    // and, as there, the convention is VALIDATED rather than
                    // trusted: the assembler hard-fails unless the slot the
                    // script passes as the 5th operand really is `src + 1`
                    // (see asm/src/main.rs), and the engine re-checks that the
                    // slot exists before dereferencing it.
                    let selslot = if selgate { b + 1 } else { b };
                    if selgate && selslot >= self.n_slots {
                        crate::print(b"SPARSE_MATVEC: sel slot (src+1) does not exist\n");
                        print_u32(selslot as u32);
                        crate::print(b"\n");
                        exit_i(9);
                    }
                    let (serows, secols) = self.slot_shape(selslot);
                    // P16-5: the selection table is [n_neurons, 2k] -- per
                    // NEURON, not flat -- so the only structural requirements
                    // are an even width and enough rows. Demanding width == 2
                    // (the flat layout) made every segmented program fail here;
                    // that was the first thing this guard caught. The row-count
                    // check happens below, once `n` is known.
                    if selgate && (secols == 0 || secols % 2 != 0) {
                        crate::print(b"SPARSE_MATVEC: selgate needs an even sel width\n");
                        crate::print(b" sel=");
                        print_u32(serows as u32);
                        crate::print(b"x");
                        print_u32(secols as u32);
                        crate::print(b"\n");
                        exit_i(9);
                    }
                    if ocols != 1 || scols != 1 {
                        crate::print(b"SPARSE_MATVEC: offsets and src must be column vectors\n");
                        exit_i(9);
                    }
                    let n1 = orows;
                    if n1 < 2 {
                        crate::print(b"SPARSE_MATVEC: offsets must have n_neurons+1 entries\n");
                        exit_i(9);
                    }
                    let n = n1 - 1;
                    let nnz = srows;
                    if dc != 1 || dr < n {
                        crate::print(b"SPARSE_MATVEC: dst must be [n_neurons(,1)]\n");
                        crate::print(b" n=");
                        print_u32(n as u32);
                        crate::print(b" dst=");
                        print_u32(dr as u32);
                        crate::print(b"x");
                        print_u32(dc as u32);
                        crate::print(b"\n");
                        exit_i(9);
                    }
                    // FL_NNZ needs one more element than the plain product.
                    if want_nnz && dr < n + 1 {
                        crate::print(b"SPARSE_MATVEC: nnz needs dst to be [n_neurons+1]\n");
                        print_u32(dr as u32);
                        crate::print(b"x");
                        print_u32(dc as u32);
                        crate::print(b"\n");
                        exit_i(9);
                    }
                    let (xr, xc) = self.slot_shape(xslot);
                    let nx = xr * xc;
                    if nx == 0 {
                        crate::print(b"SPARSE_MATVEC: x slot (dt) has width 0\n");
                        exit_i(9);
                    }
                    let (wr, wc) = self.slot_shape(wslot);
                    if wr * wc == 0 {
                        crate::print(b"SPARSE_MATVEC: weight slot (fl) has width 0\n");
                        exit_i(9);
                    }
                    let co = self.slot_mut(a);
                    let cs = self.slot_mut(b);
                    let cw = self.slot_mut(wslot);
                    let cx = self.slot_mut(xslot);
                    let cd = self.slot_mut(d);
                    let mut nnz_done = 0usize;
                    for j in 0..n {
                        // Offsets are carried as f32 (the whole engine stores
                        // indices as f32 -- OP_EMB reads its row id the same
                        // way). Exact for any index < 2^24, and the artifacts
                        // here are orders of magnitude smaller than that.
                        let o0 = *co.add(j) as usize;
                        let o1 = *co.add(j + 1) as usize;
                        // A CSR that walks off the end would read past the src
                        // array and produce a number that looks like a real
                        // activation. Hard-fail instead.
                        if o1 > nnz || o0 > o1 {
                            crate::print(b"SPARSE_MATVEC: offsets out of range\n");
                            crate::print(b" j=");
                            print_u32(j as u32);
                            crate::print(b" o0=");
                            print_u32(o0 as u32);
                            crate::print(b" o1=");
                            print_u32(o1 as u32);
                            crate::print(b" nnz=");
                            print_u32(nnz as u32);
                            crate::print(b"\n");
                            exit_i(9);
                        }
                        // NOT seeded with 0.0 and then added into: the first
                        // KEPT term initialises the accumulator, exactly as
                        // OP_MAX does. Two reasons -- (a) `0.0 + (-0.0)` is
                        // +0.0, so seeding would destroy the sign of a
                        // legitimately negative-zero row; (b) a neuron whose
                        // connections are all dropped must yield something
                        // unambiguous, and `0.0` is the value that makes
                        // "dropped" and "computed zero" indistinguishable by
                        // design here -- the nnz counter is what distinguishes
                        // them, which is the honest place to draw that line.
                        if o0 == o1 {
                            *cd.add(j) = 0.0;
                            continue;
                        }
                        // ---- P16-5 SEL-GATED accumulation -------------------
                        //
                        // GATED OPERAND MAP (the contract; mirrored in
                        // asm/src/main.rs and format.rs):
                        //   a        offsets[n+1]    row spans
                        //   b        src[nnz]        source-neuron id per connection
                        //   b + 1    sel[n,2k]       per-neuron TOP-K selection
                        //   fl       weights[nnz]    connection weight per connection
                        //   dt       x               input activation vector
                        // Ungated mode is identical except that `b+1` is unused.
                        //
                        // WHY sel IS PER-NEURON AND ROW-RELATIVE -- this is the
                        // one thing that had to be got right for the two opcodes
                        // to compose at all, and the first attempt got it wrong
                        // (the engine hard-failed with "sel index outside its CSR
                        // row, idx=8 row=[0,6)", which is the guard doing its job):
                        //
                        //   OP_TOPK_BY_STRENGTH numbers its output 0..L-1 over
                        //   the array it was handed. OP_SPARSE_MATVEC numbers
                        //   connections 0..nnz-1 over the whole CSR. With a dense
                        //   CSR these two numberings AGREE only for row 0 of the
                        //   matrix -- for every later row the top-k indices are
                        //   offset by that row's start.
                        //
                        // So `sel` carries n segments of 2k entries, segment j
                        // holding neuron j's survivors with indices RELATIVE to
                        // its own row; the absolute connection index is
                        // `o0 + rel`. That keeps TOPK_BY_STRENGTH's opcode
                        // unchanged (it is still an array-local top-k) while
                        // making the composition well defined, and it is why
                        // segment length is `2 * K_MAX` rather than `2 * L`.
                        //
                        // This branch is what makes k matter: row j's total is
                        // now the sum over its SELECTED connections only, so
                        // shrinking k shrinks y. Measured before this change:
                        // y was bit-identical at k=2/5/24 with nnz pinned at 24.
                        //
                        // Invariants ENFORCED, not assumed:
                        //   * `o0 + rel` must lie inside row j's span [o0,o1);
                        //     a `sel` built against a different CSR would pull a
                        //     weight belonging to another neuron -- a number that
                        //     looks like a real activation and is not. exit(9).
                        //   * a row-relative index of 0 after the first entry is
                        //     the zero-pad sentinel, so the walk STOPS there.
                        //
                        // Summation order: ascending within the row, identical to
                        // the ungated CSR order. That shared order is the stated
                        // precondition for the Python reference in
                        // tools/p16_compose.py to be bit-exact.
                        if selgate {
                            let csel = self.slot_mut(selslot);
                            // Segment stride comes from the DECLARED sel width,
                            // not from k: the table is a fixed-layout artifact
                            // and k is a runtime scalar, so the layout must not
                            // move when k does.
                            let seg = secols / 2;
                            if seg == 0 || (j + 1) * seg * 2 > serows * secols {
                                crate::print(b"SPARSE_MATVEC: sel table too small for n_neurons\n");
                                print_u32(serows as u32);
                                crate::print(b"x");
                                print_u32(secols as u32);
                                crate::print(b" n=");
                                print_u32(n as u32);
                                crate::print(b"\n");
                                exit_i(9);
                            }
                            let base_q = j * seg;
                            let mut acc = 0.0f32;
                            let mut first = true;
                            for q in 0..seg {
                                let rel = *csel.add((base_q + q) * 2) as usize;
                                if q > 0 && rel == 0 {
                                    break; // pad => no further survivors
                                }
                                let idx = o0 + rel;
                                if idx < o0 || idx >= o1 {
                                    crate::print(b"SPARSE_MATVEC: sel index outside its CSR row\n");
                                    crate::print(b" j=");
                                    print_u32(j as u32);
                                    crate::print(b" rel=");
                                    print_u32(rel as u32);
                                    crate::print(b" row=[");
                                    print_u32(o0 as u32);
                                    crate::print(b",");
                                    print_u32(o1 as u32);
                                    crate::print(b")\n");
                                    exit_i(9);
                                }
                                let sid = *cs.add(idx) as usize;
                                if sid >= nx {
                                    crate::print(b"SPARSE_MATVEC: src_id out of range\n");
                                    crate::print(b" t=");
                                    print_u32(idx as u32);
                                    crate::print(b" src=");
                                    print_u32(sid as u32);
                                    crate::print(b" nx=");
                                    print_u32(nx as u32);
                                    crate::print(b"\n");
                                    exit_i(9);
                                }
                                let term = *cw.add(idx) * *cx.add(sid);
                                acc = if first { term } else { acc + term };
                                first = false;
                                nnz_done += 1;
                            }
                            *cd.add(j) = acc;
                            continue;
                        }
                        let mut acc = 0.0f32;
                        for t in o0..o1 {
                            // src id out of range = the CSR and x disagree about
                            // the input width. Reading it anyway would index
                            // past x into the next slot: plausible numbers, no
                            // error -- the silent-degradation class.
                            let sid = *cs.add(t) as usize;
                            if sid >= nx {
                                crate::print(b"SPARSE_MATVEC: src_id out of range\n");
                                crate::print(b" t=");
                                print_u32(t as u32);
                                crate::print(b" src=");
                                print_u32(sid as u32);
                                crate::print(b" nx=");
                                print_u32(nx as u32);
                                crate::print(b"\n");
                                exit_i(9);
                            }
                            let term = *cw.add(t) * *cx.add(sid);
                            if t == o0 {
                                acc = term;
                            } else {
                                acc += term;
                            }
                            nnz_done += 1;
                        }
                        *cd.add(j) = acc;
                    }
                    if want_nnz {
                        *cd.add(n) = nnz_done as f32;
                    }
                }

                // ============ backward ops (train feature) ============
                // The whole backward path is behind `train`; when off these
                // opcodes all hard-fail instead of linking backward kernels.
                OP_GEMM_BWD_W | OP_GEMM_BWD_X | OP_BIAS_BWD | OP_ACT_BWD | OP_CE_BWD
                | OP_MSE_BWD | OP_ZERO | OP_ADAM | OP_SGD | OP_LOSS_ADD | OP_CLIP_GRAD
                | OP_SOFTMAX_BWD | OP_LN_BWD | OP_OUTER_BWD | OP_EMB_BWD | OP_ADAMW => {
                    // P10  C ***                    //
                    // `ADAM`/`ADAMW`/`SGD` ***                    // mmap(PAGE_READONLY)                    // 0xC0000005 CI exit=3221225477                    //                     //
                    // exit(9)
                        if self.cpool_readonly
                        && matches!(ins.op, OP_ADAM | OP_ADAMW | OP_SGD)
                    {
                        crate::print(b"weights are mmap RO; no training\n");
                        exit_i(9);
                    }
                    #[cfg(feature = "train")]
                    self.exec_bwd(ins, d, a, b);
                    #[cfg(not(feature = "train"))]
                    {
                        let _ = (&ins, d, a, b);
                        crate::print(b"train feature disabled\n");
                        exit_i(6);
                    }
                }

                _ => {
                    crate::print(b"unimpl op\n");
                    exit_i(5);
                }
            }
        }
        0
    }
}

pub unsafe fn exit_i(code: i32) -> ! {
    crate::exit(code)
}

// [抽库 L4-2] `print_u32` 是 IO 原语，下移到 vmlm-sys::io；这里重导出 ⇒ 路径不变。
pub use vmlm_sys::io::print_u32;


// Minimal f32 printing -- deliberately avoids core::fmt (saves tens of KB).
//
// P13 (size budget): this was 3072 B of the engine's 53248 B -- the single
// largest reclaimable region -- and it is on the ACCEPTANCE path
// (p12_logits_check.py reads the engine's stdout to compare 522240 logits
// against HuggingFace), so it cannot simply be deleted or #[cfg]-ed away.
//
// What was measured before touching it (tools/p13_sizeprobe.py):
//   print_f32 body alone .......... 3072 B
//   print_u32 body alone .......... 1024 B
//   sin_f + cos_f bodies .......... 1536 B
//   RoPE per-element body ......... 2048 B   (replacing only its trig CALL
//                                             SITES saves 0 B -- the calls
//                                             are inlined; the bytes live in
//                                             the kernels, not the callers)
//
// The precision question is NOT a guess -- it is bounded by the artifact:
//   logits absmax = 21.045, tightest top1-top2 margin = 0.103
//   => relative resolution needed for a correct argmax ~ 4.9e-03.
// Measured: 6 fraction digits gives rel 9.06e-08 / argmax 4/4;
//           3 fraction digits gives rel 4.75e-05 / argmax 4/4.
// Both pass, but 3 digits would WEAKEN the headline `rel` metric from
// 1.88e-06 to ~5e-05 for no reason. So this rewrite keeps ALL SIX digits and
// instead removes the redundant machinery: the 64-byte scratch buffer and the
// separate 20-byte integer-digit array (the fraction is emitted in place).
pub unsafe fn print_f32(p: *const f32, n: usize) {
    let mut buf = [0u8; 40];
    for i in 0..n {
        let v = *p.add(i);
        let mut len = 0usize;
        let neg = v < 0.0;
        let av = if neg { -v } else { v };
        if neg {
            buf[len] = b'-';
            len += 1;
        }
        // Integer part: digits are generated least-significant first into the
        // tail of the buffer, then the whole run is rotated into place. This
        // replaces the old `d[20]` + copy loop with arithmetic on the same
        // buffer that already holds the sign.
        let ip = av as u64;
        let mut t = ip;
        let int_start = len;
        if t == 0 {
            buf[len] = b'0';
            len += 1;
        } else {
            // Reserve exactly the digits this number needs, then fill backwards.
            let mut tmp = [0u8; 20];
            let mut dn = 0usize;
            while t > 0 {
                tmp[dn] = b'0' + (t % 10) as u8;
                t /= 10;
                dn += 1;
            }
            // Guard: 40-byte buffer, sign(1) + 20 int digits + '.' + 6 + sep
            // = 29, so this never overflows; keep the bound explicit anyway.
            while dn > 0 && len < 34 {
                dn -= 1;
                buf[len] = tmp[dn];
                len += 1;
            }
        }
        let _ = int_start;
        buf[len] = b'.';
        len += 1;
        // 6 fraction digits: the tightest argmax margin in the artifact is
        // 0.103 on values up to 21.04, so 6 digits is ~4 orders of margin.
        let mut frac = av - (ip as f32);
        let mut k = 0;
        while k < 6 && len < 38 {
            frac *= 10.0;
            let dig = frac as u32;
            buf[len] = b'0' + (dig % 10) as u8;
            len += 1;
            frac -= dig as f32;
            k += 1;
        }
        buf[len] = if i + 1 == n { b'\n' } else { b' ' };
        len += 1;
        crate::print(&buf[..len]);
    }
}



pub unsafe fn init_cpu() {
    HAS_AVX2 = cpu_has_avx2();
}
