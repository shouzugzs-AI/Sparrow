// P15 VERIFIER PROBE -- compiled by tools/p15_verify_guards.py.
//
// This file is written by the verifier's harness and lives under
// tools/p15_verify_guardprobe/. It is NOT part of the engine and NOT part of
// any implementation task: it exists so the verifier can observe the REAL
// guards firing (exit code + message) instead of trusting a Python mirror of
// them.
//
// It is a std binary on purpose: the engine is no_std and links /NODEFAULTLIB,
// and a probe that had to reproduce that link configuration would itself be a
// measurement hazard. Calling into vmlm-core from std works because the crate
// exports ordinary pub fns; exit(9) maps to process exit code 9 either way,
// which is the number under test.
use vmlm_core::neuron_kv::{self, NeuronId, NeuronKvLayout};

fn heap_f32(n: usize) -> *mut f32 {
    let mut v: Vec<f32> = Vec::with_capacity(n);
    v.resize(n, 0.0f32);
    let p = v.as_mut_ptr();
    core::mem::forget(v);
    p
}
fn heap_u8(n: usize) -> *mut u8 {
    let mut v: Vec<u8> = Vec::with_capacity(n);
    v.resize(n, 0u8);
    let p = v.as_mut_ptr();
    core::mem::forget(v);
    p
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let case = args.get(1).cloned().unwrap_or_default();
    unsafe {
        match case.as_str() {
            // ---- G0a: zero dimension ----
            "g0a_zero_layers" => {
                let l = NeuronKvLayout::with_stride(0, 2, 128, 1024, 1024);
                l.validate();
            }
            "g0a_zero_heads" => {
                let l = NeuronKvLayout::with_stride(24, 0, 128, 1024, 1024);
                l.validate();
            }
            "g0a_zero_dim" => {
                let l = NeuronKvLayout::with_stride(24, 2, 0, 1024, 1024);
                l.validate();
            }
            // ---- G0b: max_seq == 0 ----
            "g0b_zero_maxseq" => {
                let l = NeuronKvLayout::with_stride(24, 2, 128, 0, 0);
                l.validate();
            }
            // ---- G1: stride < max_seq ----
            "g1_stride_lt_maxseq" => {
                let l = NeuronKvLayout::with_stride(24, 2, 128, 1024, 1023);
                l.validate();
            }
            "g1_legal_boundary" => {
                // stride == max_seq is the LEGAL boundary (tight()). Must NOT fail.
                let l = NeuronKvLayout::tight(24, 2, 128, 1024);
                l.validate();
                println!("OK validate accepted tight stride=PMAX");
            }
            "g1_legal_padded" => {
                // stride > max_seq is legal padding. Must NOT fail.
                let l = NeuronKvLayout::with_stride(24, 2, 128, 1024, 2048);
                l.validate();
                println!("OK validate accepted padded stride=2*PMAX");
            }
            // ---- G2: slot >= max_seq ----
            "g2_slot_eq_maxseq" => {
                let l = NeuronKvLayout::tight(24, 2, 128, 1024);
                let id = NeuronId { layer: 0, kv_head: 0, dim: 0 };
                l.slot_addr(id, 1024);
            }
            "g2_slot_gt_maxseq" => {
                let l = NeuronKvLayout::tight(24, 2, 128, 1024);
                let id = NeuronId { layer: 0, kv_head: 0, dim: 0 };
                l.slot_addr(id, 99999);
            }
            "g2_legal_last_slot" => {
                let l = NeuronKvLayout::tight(24, 2, 128, 1024);
                let id = NeuronId { layer: 0, kv_head: 0, dim: 0 };
                let a = l.slot_addr(id, 1023);
                println!("OK slot_addr(last legal slot 1023) = {}", a);
            }
            // ---- G2b: slot check must key off max_seq, NOT stride ----
            //   With stride=2048 > max_seq=1024, slot=1500 is INSIDE the stride
            //   but OUTSIDE max_seq. It must STILL fail: the neuron's own slot
            //   space is max_seq, and a reader that walks to stride would be
            //   reading padding/another neuron's future. This is the
            //   discriminating case between "guard keyed on stride" (a silent
            //   degradation) and "guard keyed on max_seq" (correct).
            "g2b_slot_inside_stride_outside_maxseq" => {
                let l = NeuronKvLayout::with_stride(24, 2, 128, 1024, 2048);
                let id = NeuronId { layer: 0, kv_head: 0, dim: 0 };
                l.slot_addr(id, 1500);
            }
            // ---- G5: neuron id out of range ----
            "g5_layer_oob" => {
                let l = NeuronKvLayout::tight(24, 2, 128, 1024);
                l.neuron_base(NeuronId { layer: 24, kv_head: 0, dim: 0 });
            }
            "g5_head_oob" => {
                let l = NeuronKvLayout::tight(24, 2, 128, 1024);
                l.neuron_base(NeuronId { layer: 0, kv_head: 2, dim: 0 });
            }
            "g5_dim_oob" => {
                let l = NeuronKvLayout::tight(24, 2, 128, 1024);
                l.neuron_base(NeuronId { layer: 0, kv_head: 0, dim: 128 });
            }
            "g5_legal_last_id" => {
                let l = NeuronKvLayout::tight(24, 2, 128, 1024);
                let b = l.neuron_base(NeuronId { layer: 23, kv_head: 1, dim: 127 });
                println!("OK neuron_base(last legal id) = {}", b);
            }
            // ---- G3: offsets[] disagrees with stride ----
            "g3_offsets_disagree" => {
                // Build a section by hand with entry[1] wrong, then read it.
                let l = NeuronKvLayout::tight(2, 1, 2, 8); // 4 neurons
                let n = l.section_bytes();
                let buf = heap_u8(n);
                neuron_kv::write_section(&l, buf);
                // corrupt entry index 1: it should be 1*8 = 8
                let off = 20 + 1 * 4;
                core::ptr::write_unaligned(buf.add(off) as *mut u32, 9u32);
                // NOTE: read_section now takes an explicit `avail` byte count
                // (the S8 fix threads the length through). `n` is the true
                // section size here, so this is the exact-length case and the
                // offsets[] mismatch is the only thing wrong.
                let got = neuron_kv::read_section(buf as *const u8, n);
                println!("HOLE read_section ACCEPTED a corrupt offsets[]: stride={}",
                         got.stride);
            }
            "g3_offsets_ok" => {
                let l = NeuronKvLayout::tight(2, 1, 2, 8);
                let n = l.section_bytes();
                let buf = heap_u8(n);
                let written = neuron_kv::write_section(&l, buf);
                let got = neuron_kv::read_section(buf as *const u8, n);
                println!("OK write_section wrote {} B; read_section -> stride={} n_neurons={}",
                         written, got.stride, got.n_neurons());
            }
            // ---- G3b: the 128x form must be REJECTED by read_section ----
            //   A section whose entries are i*head_dim*stride must not be
            //   accepted. This is the shape the S5-A defect would have written.
            "g3b_section_head_dim_stride" => {
                let l = NeuronKvLayout::tight(2, 1, 2, 8);
                let n = l.section_bytes();
                let buf = heap_u8(n);
                neuron_kv::write_section(&l, buf);
                for i in 0..l.n_neurons() {
                    let v = (i as u64 * 2 * 8) as u32; // i*head_dim*stride
                    core::ptr::write_unaligned(buf.add(20 + i * 4) as *mut u32, v);
                }
                let got = neuron_kv::read_section(buf as *const u8, n);
                println!("HOLE read_section ACCEPTED i*head_dim*stride entries: stride={}",
                         got.stride);
            }
            // ---- G3c: a section whose header stride disagrees with its entries ----
            "g3c_header_stride_disagrees" => {
                let l = NeuronKvLayout::tight(2, 1, 2, 8);
                let n = l.section_bytes();
                let buf = heap_u8(n);
                neuron_kv::write_section(&l, buf);
                // entries were written for stride 8; claim stride 16.
                core::ptr::write_unaligned(buf.add(16) as *mut u32, 16u32);
                let got = neuron_kv::read_section(buf as *const u8, n);
                println!("HOLE read_section ACCEPTED a stride header that disagrees with entries: stride={}",
                         got.stride);
            }
            // ---- G4: n_pos > max_seq ----
            "g4_npos_gt_maxseq" => {
                let l = NeuronKvLayout::tight(2, 1, 2, 8); // 4 neurons, max_seq 8
                let n = l.n_neurons();
                let src = heap_f32(64);
                let dst = heap_f32(n * l.stride as usize + 16);
                neuron_kv::reorder_compact_to_neurons(&l, src, dst, 9);
                println!("HOLE reorder ACCEPTED n_pos=9 > max_seq=8");
            }
            "g4_legal_npos_eq_maxseq" => {
                let l = NeuronKvLayout::tight(2, 1, 2, 8);
                let n = l.n_neurons();
                let src = heap_f32(64);
                let dst = heap_f32(n * l.stride as usize);
                neuron_kv::reorder_compact_to_neurons(&l, src, dst, 8);
                println!("OK reorder accepted n_pos==max_seq==8");
            }
            "g4_legal_npos_zero" => {
                // n_pos == 0 is LEGAL and must not fail -- but it must also not
                // look like a successful measurement, which is why the caller
                // is responsible for saying so. Recorded here as behaviour.
                let l = NeuronKvLayout::tight(2, 1, 2, 8);
                let n = l.n_neurons();
                let src = heap_f32(64);
                let dst = heap_f32(n * l.stride as usize);
                neuron_kv::reorder_compact_to_neurons(&l, src, dst, 0);
                println!("OK reorder accepted n_pos=0 (it copies nothing; a caller must not read that as 'verified')");
            }
            // ---- G4b: the reverse reorder must have the SAME guard ----
            "g4b_npos_gt_maxseq_reverse" => {
                let l = NeuronKvLayout::tight(2, 1, 2, 8);
                let n = l.n_neurons();
                let src = heap_f32(n * l.stride as usize);
                let dst = heap_f32(64);
                neuron_kv::reorder_neurons_to_compact(&l, src, dst, 9);
                println!("HOLE reverse reorder ACCEPTED n_pos=9 > max_seq=8");
            }
            // ---- G6: layout exceeds 32-bit offsets ----
            "g6_exceeds_u32" => {
                // 4 neurons * stride 2^30 = 2^32 > u32::MAX
                let l = NeuronKvLayout::with_stride(2, 2, 1, 1, 1 << 30);
                l.validate();
            }
            // ---- write_section must reject an illegal layout too ----
            "g7_write_rejects_bad_layout" => {
                let l = NeuronKvLayout::with_stride(24, 2, 128, 1024, 1023);
                let buf = heap_u8(l.section_bytes());
                neuron_kv::write_section(&l, buf);
                println!("HOLE write_section ACCEPTED stride<max_seq");
            }
            // ---- addresses: measured, not asserted ----
            "addr_probe" => {
                let l = NeuronKvLayout::tight(24, 2, 128, 1024);
                let b0 = l.neuron_base(NeuronId { layer: 0, kv_head: 0, dim: 0 });
                let b1 = l.neuron_base(NeuronId { layer: 0, kv_head: 0, dim: 1 });
                let b_last = l.neuron_base(NeuronId { layer: 23, kv_head: 1, dim: 127 });
                let s0 = l.slot_addr(NeuronId { layer: 0, kv_head: 0, dim: 0 }, 0);
                let s1023 = l.slot_addr(NeuronId { layer: 0, kv_head: 0, dim: 0 }, 1023);
                println!("ADDR base(d0)={} base(d1)={} spacing={} base(last)={} slot0={} slot1023={} n_neurons={} section_bytes={}",
                         b0, b1, b1 - b0, b_last, s0, s1023, l.n_neurons(), l.section_bytes());
            }
            // ---- G8: the section descriptor's declared len ----
            //   load_find_section checks `len >= NKV_HDR` and
            //   `off + len <= file_len`, but does NOT check
            //   `len == NKV_HDR + n_neurons*NKV_ENTRY`. read_section then reads
            //   n_neurons entries derived from the HEADER, ignoring `len`.
            //   This case builds an artifact whose descriptor says len=20 while
            //   the header behind it declares 2x1x2 (4 neurons) -> 36 bytes of
            //   entries. If it is accepted, the declared length does not bound
            //   the contents.
            "g8_declared_len_too_short" => {
                // layout: [0..20) header fields  [20..32) misc  [32..44) descriptor
                //         [44..)   section
                let layout = NeuronKvLayout::tight(2, 1, 2, 8); // 4 neurons
                let sec_bytes = layout.section_bytes();          // 20 + 16 = 36
                let total = 44 + sec_bytes + 64;
                let buf = heap_u8(total);
                core::ptr::write_bytes(buf, 0, total);
                // header fields that validate_artifact_header compares
                core::ptr::write_unaligned(buf.add(0) as *mut u32, layout.n_layers);
                core::ptr::write_unaligned(buf.add(4) as *mut u32, layout.n_kv_heads);
                core::ptr::write_unaligned(buf.add(8) as *mut u32, layout.head_dim);
                core::ptr::write_unaligned(buf.add(12) as *mut u32, layout.max_seq_per_neuron);
                core::ptr::write_unaligned(buf.add(16) as *mut u32, layout.stride);
                // descriptor: tag OK, off=44, len=20 -- SHORT BY 16 BYTES
                core::ptr::write_unaligned(buf.add(32) as *mut u32, neuron_kv::TAG);
                core::ptr::write_unaligned(buf.add(36) as *mut u32, 44u32);
                core::ptr::write_unaligned(buf.add(40) as *mut u32, 20u32);
                neuron_kv::write_section(&layout, buf.add(44));
                // the file is long enough that off+len <= total holds
                let secoff = neuron_kv::load_find_section(buf, total);
                // `avail` = bytes actually readable from secoff. This is the
                // SEMANTIC under test: the descriptor said 20, but 36 bytes are
                // really there. read_section must NOT trust the descriptor to
                // shrink its own read below what the header needs.
                let got = neuron_kv::read_section(secoff, total - 44);
                println!("HOLE declared_len=20 accepted while the header needs {} bytes; read_section(avail={}) returned n_neurons={} stride={}",
                         sec_bytes, total - 44, got.n_neurons(), got.stride);
            }
            "g8_declared_len_exact_ok" => {
                let layout = NeuronKvLayout::tight(2, 1, 2, 8);
                let sec_bytes = layout.section_bytes();
                let total = 44 + sec_bytes + 64;
                let buf = heap_u8(total);
                core::ptr::write_bytes(buf, 0, total);
                core::ptr::write_unaligned(buf.add(0) as *mut u32, layout.n_layers);
                core::ptr::write_unaligned(buf.add(4) as *mut u32, layout.n_kv_heads);
                core::ptr::write_unaligned(buf.add(8) as *mut u32, layout.head_dim);
                core::ptr::write_unaligned(buf.add(12) as *mut u32, layout.max_seq_per_neuron);
                core::ptr::write_unaligned(buf.add(16) as *mut u32, layout.stride);
                core::ptr::write_unaligned(buf.add(32) as *mut u32, neuron_kv::TAG);
                core::ptr::write_unaligned(buf.add(36) as *mut u32, 44u32);
                core::ptr::write_unaligned(buf.add(40) as *mut u32, sec_bytes as u32);
                neuron_kv::write_section(&layout, buf.add(44));
                let secoff = neuron_kv::load_find_section(buf, total);
                // avail must be the SECTION's own size here, not the whole tail:
                // the buffer has 64 bytes of slack after the section, and the
                // fix requires avail == section_bytes() exactly. (My first
                // version passed total-44=100, which the fix correctly rejects
                // -- that was a harness bug, not a code bug.)
                let got = neuron_kv::read_section(secoff, sec_bytes);
                println!("OK exact declared_len={} accepted, n_neurons={}",
                         sec_bytes, got.n_neurons());
            }
            // ---- G9: descriptor len larger than the file ----
            "g9_len_exceeds_file" => {
                let layout = NeuronKvLayout::tight(2, 1, 2, 8);
                let sec_bytes = layout.section_bytes();
                let total = 44 + sec_bytes;
                let buf = heap_u8(total);
                core::ptr::write_bytes(buf, 0, total);
                core::ptr::write_unaligned(buf.add(0) as *mut u32, layout.n_layers);
                core::ptr::write_unaligned(buf.add(4) as *mut u32, layout.n_kv_heads);
                core::ptr::write_unaligned(buf.add(8) as *mut u32, layout.head_dim);
                core::ptr::write_unaligned(buf.add(12) as *mut u32, layout.max_seq_per_neuron);
                core::ptr::write_unaligned(buf.add(16) as *mut u32, layout.stride);
                core::ptr::write_unaligned(buf.add(32) as *mut u32, neuron_kv::TAG);
                core::ptr::write_unaligned(buf.add(36) as *mut u32, 44u32);
                // claim 4096 bytes when only sec_bytes exist
                core::ptr::write_unaligned(buf.add(40) as *mut u32, 4096u32);
                neuron_kv::write_section(&layout, buf.add(44));
                let _ = neuron_kv::load_find_section(buf, total);
                println!("HOLE descriptor len=4096 ACCEPTED on a {}-byte file",
                         total);
            }
            // ---- G10: tag == 0 means "no section" ----
            "g10_tag_zero" => {
                let buf = heap_u8(128);
                core::ptr::write_bytes(buf, 0, 128);
                let _ = neuron_kv::load_find_section(buf, 128);
                println!("HOLE tag=0 (no section) accepted, code proceeded");
            }
            // ---- G11: the S8 FIX path (read_section_declared / avail) ----
            //   The verifier's S8 finding was that the descriptor's declared len
            //   does not bound the contents. The implementation now threads an
            //   `avail` length into read_section_impl and requires
            //   avail == layout.section_bytes() exactly. These cases verify the
            //   FIX and that it is not over-strict.
            //
            //   NOTE on the API name: the implementation renamed this while the
            //   verifier was probing it (read_section_bounded -> the descriptor
            //   wrapper read_section_declared + a private read_section_impl).
            //   The probe below drives the PUBLIC wrapper, which is the path the
            //   load entry actually uses, so the test tracks the shipped API
            //   rather than an intermediate name.
            "g11_bounded_under_declared" => {
                let layout = NeuronKvLayout::tight(2, 1, 2, 8); // needs 36 B
                let total = 44 + 64;
                let buf = heap_u8(total);
                core::ptr::write_bytes(buf, 0, total);
                core::ptr::write_unaligned(buf.add(0) as *mut u32, layout.n_layers);
                core::ptr::write_unaligned(buf.add(4) as *mut u32, layout.n_kv_heads);
                core::ptr::write_unaligned(buf.add(8) as *mut u32, layout.head_dim);
                core::ptr::write_unaligned(buf.add(12) as *mut u32, layout.max_seq_per_neuron);
                core::ptr::write_unaligned(buf.add(16) as *mut u32, layout.stride);
                core::ptr::write_unaligned(buf.add(32) as *mut u32, neuron_kv::TAG);
                core::ptr::write_unaligned(buf.add(36) as *mut u32, 44u32);
                // declared 20, actual need 36 -> must now FAIL
                core::ptr::write_unaligned(buf.add(40) as *mut u32, 20u32);
                neuron_kv::write_section(&layout, buf.add(44));
                let (_sec, got) = neuron_kv::read_section_declared(buf, total);
                println!("HOLE read_section_declared ACCEPTED declared_len=20 while the header needs {} (n_neurons={})",
                         layout.section_bytes(), got.n_neurons());
            }
            "g11_bounded_exact_ok" => {
                let layout = NeuronKvLayout::tight(2, 1, 2, 8);
                let total = 44 + 64;
                let buf = heap_u8(total);
                core::ptr::write_bytes(buf, 0, total);
                core::ptr::write_unaligned(buf.add(0) as *mut u32, layout.n_layers);
                core::ptr::write_unaligned(buf.add(4) as *mut u32, layout.n_kv_heads);
                core::ptr::write_unaligned(buf.add(8) as *mut u32, layout.head_dim);
                core::ptr::write_unaligned(buf.add(12) as *mut u32, layout.max_seq_per_neuron);
                core::ptr::write_unaligned(buf.add(16) as *mut u32, layout.stride);
                core::ptr::write_unaligned(buf.add(32) as *mut u32, neuron_kv::TAG);
                core::ptr::write_unaligned(buf.add(36) as *mut u32, 44u32);
                core::ptr::write_unaligned(buf.add(40) as *mut u32,
                                           layout.section_bytes() as u32);
                neuron_kv::write_section(&layout, buf.add(44));
                let (_sec, got) = neuron_kv::read_section_declared(buf, total);
                println!("OK read_section_declared accepted the exact declared_len={} n_neurons={}",
                         layout.section_bytes(), got.n_neurons());
            }
            "g11_bounded_over_declared" => {
                // declared LONGER than needed: the extra bytes are not entries,
                // so this is also a mismatch and must fail.
                let layout = NeuronKvLayout::tight(2, 1, 2, 8);
                let total = 44 + 64;
                let buf = heap_u8(total);
                core::ptr::write_bytes(buf, 0, total);
                core::ptr::write_unaligned(buf.add(0) as *mut u32, layout.n_layers);
                core::ptr::write_unaligned(buf.add(4) as *mut u32, layout.n_kv_heads);
                core::ptr::write_unaligned(buf.add(8) as *mut u32, layout.head_dim);
                core::ptr::write_unaligned(buf.add(12) as *mut u32, layout.max_seq_per_neuron);
                core::ptr::write_unaligned(buf.add(16) as *mut u32, layout.stride);
                core::ptr::write_unaligned(buf.add(32) as *mut u32, neuron_kv::TAG);
                core::ptr::write_unaligned(buf.add(36) as *mut u32, 44u32);
                core::ptr::write_unaligned(buf.add(40) as *mut u32, 40u32);
                neuron_kv::write_section(&layout, buf.add(44));
                let (_sec, got) = neuron_kv::read_section_declared(buf, total);
                println!("HOLE read_section_declared ACCEPTED declared_len=40 while the header needs {} (n_neurons={})",
                         layout.section_bytes(), got.n_neurons());
            }
            // ---- G12: REVERSE direction and OVERLAP, measured on the real
            //      functions rather than inferred from comments.
            //      The reorder is row->col: dst[i*stride+s] = src[s*n+i].
            //      Running it backwards (i.e. reading src as [n][npos]) produces
            //      a TRANSPOSED result that is shape-correct and numerically
            //      wrong -- the silent failure this project hunts. The engine's
            //      polarity_demo asserts the correct direction produces constant
            //      rows equal to the row index; this case reproduces that
            //      predicate independently in the probe.
            "g12_reverse_direction_is_distinguishable" => {
                let l = NeuronKvLayout::tight(2, 1, 4, 5); // 8 neurons, PMAX=5
                let n = l.n_neurons();
                let npos = 3usize;
                // compact[p*n + i] = i  (value == neuron id)
                let compact = heap_f32(npos * n);
                for p in 0..npos {
                    for i in 0..n {
                        *compact.add(p * n + i) = i as f32;
                    }
                }
                // CORRECT direction via the shipped function
                let right = heap_f32(n * l.stride as usize);
                core::ptr::write_bytes(right as *mut u8, 0, n * l.stride as usize * 4);
                neuron_kv::reorder_compact_to_neurons(&l, compact, right, npos as u32);
                // WRONG direction, hand-written (reading compact as [n][npos])
                let wrong = heap_f32(n * l.stride as usize);
                core::ptr::write_bytes(wrong as *mut u8, 0, n * l.stride as usize * 4);
                for i in 0..n {
                    for p in 0..npos {
                        *wrong.add(i * l.stride as usize + p) =
                            *compact.add(i * npos + p);
                    }
                }
                // predicate: a CORRECT result has constant rows equal to row idx
                let mut right_const = 0usize;
                let mut right_val_ok = 0usize;
                let mut wrong_const = 0usize;
                for i in 0..n {
                    let first = *right.add(i * l.stride as usize);
                    let mut allsame = true;
                    for p in 1..npos {
                        if *right.add(i * l.stride as usize + p) != first {
                            allsame = false;
                        }
                    }
                    if allsame { right_const += 1; }
                    if first == i as f32 { right_val_ok += 1; }
                    let wf = *wrong.add(i * l.stride as usize);
                    let mut wall = true;
                    for p in 1..npos {
                        if *wrong.add(i * l.stride as usize + p) != wf {
                            wall = false;
                        }
                    }
                    if wall { wrong_const += 1; }
                }
                println!("G12 n_neurons={} npos={} correct_const_rows={} correct_val_ok={} wrong_const_rows={}",
                         n, npos, right_const, right_val_ok, wrong_const);
                if right_const != n || right_val_ok != n {
                    println!("HOLE correct direction did NOT satisfy the predicate");
                }
                if wrong_const == n {
                    println!("HOLE wrong direction ALSO satisfied the predicate -- the predicate cannot distinguish direction");
                }
            }
            // ---- G13: overlapping src/dst. The doc says the buffers must not
            //      overlap. There is no runtime check. This case runs the
            //      reorder with dst aliasing src and reports what happens, so
            //      the precondition's absence is OBSERVED rather than assumed.
            "g13_overlapping_buffers" => {
                let l = NeuronKvLayout::tight(2, 1, 2, 8); // 4 neurons, PMAX=8
                let n = l.n_neurons();
                let stride = l.stride as usize;
                let npos = 4usize;
                // one buffer used as BOTH src and dst
                let buf = heap_f32(n * stride);
                // seed the compact region: buf[p*n + i] = i as f32
                for p in 0..npos {
                    for i in 0..n {
                        *buf.add(p * n + i) = i as f32;
                    }
                }
                let before: Vec<f32> = (0..npos * n).map(|k| *buf.add(k)).collect();
                neuron_kv::reorder_compact_to_neurons(&l, buf, buf, npos as u32);
                let after: Vec<f32> = (0..npos * n).map(|k| *buf.add(k)).collect();
                let changed = before.iter().zip(after.iter()).filter(|(a, b)| a != b).count();
                println!("G13 aliased src==dst: no guard fired; {} of {} source elements changed (doc precondition is unchecked)",
                         changed, npos * n);
            }
            _ => {
                eprintln!("PROBE: unknown case {:?}", case);
                std::process::exit(64);
            }
        }
    }
    // If we get here, no guard fired. That is a PASS only for the legal cases;
    // the harness decides which. Print a sentinel so the harness can tell
    // "returned normally" from "process died".
    println!("NOGUARD");
}
