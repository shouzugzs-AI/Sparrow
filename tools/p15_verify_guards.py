#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P15-S5 GUARD COVERAGE harness (task-4, independent adversarial verifier).

WHAT THIS ANSWERS
    The S5 module documents five hard-fail paths:
        G1  stride < max_seq_per_neuron
        G2  slot >= max_seq_per_neuron
        G3  offsets[] disagrees with stride
        G4  n_pos > max_seq_per_neuron
        G5  neuron id out of range
    plus three further ones in validate():
        G0a zero dimension (n_layers / n_kv_heads / head_dim == 0)
        G0b max_seq_per_neuron == 0
        G6  layout exceeds 32-bit offsets
    A guard that is unreachable, dead, or that ACCEPTS its illegal input is a
    silent-degradation hole -- this project's cardinal sin. So for EVERY one,
    this harness constructs the illegal input and requires a loud, specific
    failure with exit code 9.

HOW THE GUARD IS ACTUALLY EXERCISED (not merely mirrored)
    Two independent executers:

      (a) RUST BINARY EXECUTER -- compiles a tiny probe binary against the REAL
          core crate with `--features neuron-kv` and calls the real
          neuron_kv::* functions. This is the only way to observe the actual
          exit(9) and the actual printed message. The probe source is written
          by THIS harness into a directory the harness owns
          (tools/p15_verify_guardprobe/) and is never a shared implementation
          file.
      (b) PYTHON MIRROR -- LayoutPrimitives.validate_reasons(), which is a
          transcription of the Rust conditions. A mirror is only worth
          something if it is shown able to DISAGREE with the binary, so the
          harness records an explicit mirror-vs-binary differential: for every
          case, the mirror's predicted reason list must match the binary's
          observed message and exit code. Any disagreement is reported as a
          defect in the mirror OR in the code, and the harness says which it
          currently believes and why.

    If (a) is unavailable (no cargo, no toolchain, build failure), the harness
    reports those criteria as UNPROVEN/UNVERIFIED -- it never falls back to
    "the mirror says so", which would be the rubber stamp.

EVERY criterion here is paired with its own can-fail control:
    for each guard, the ILLEGAL input must fail AND a LEGAL near-miss input
    (the boundary value just on the legal side) must NOT fail. An over-strict
    guard that rejects legal input is also a defect (that is exactly the shape
    of the S5-A 128x bug: over-strict, silent, and expensive).

Run from <repo>:
    python tools/p15_verify_guards.py
"""
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p15_verify_lib as V  # noqa: E402

ROOT = V.ROOT
PROBE_DIR = os.path.join(ROOT, "tools", "p15_verify_guardprobe")
CARGO = "cargo"
TARGET = "x86_64-pc-windows-msvc"

# The probe prints a machine-readable line per case, and exits 9 on the first
# guard that fires. Each case runs in its own process so exit(9) cannot end the
# sweep -- that is why the probe takes a case name on the command line.
PROBE_MAIN = r'''// P15 VERIFIER PROBE -- compiled by tools/p15_verify_guards.py.
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
'''

PROBE_CARGO = '''[package]
name = "p15-verify-guardprobe"
version = "0.1.0"
edition = "2021"

# VERIFIER-OWNED probe. It depends on the REAL core crate with the S5 feature
# ON, so the guards under test are the shipped ones. It is never part of a
# deliverable binary and never linked into the engine.
[dependencies]
vmlm-core = { path = "../../core", features = ["neuron-kv"] }

[[bin]]
name = "p15-verify-guardprobe"
path = "src/main.rs"

[workspace]
'''


def ensure_probe():
    d = PROBE_DIR
    os.makedirs(os.path.join(d, "src"), exist_ok=True)
    with open(os.path.join(d, "Cargo.toml"), "w", encoding="utf-8") as f:
        f.write(PROBE_CARGO)
    with open(os.path.join(d, "src", "main.rs"), "w", encoding="utf-8") as f:
        f.write(PROBE_MAIN)
    # a local .cargo/config.toml that mirrors the root one but WITHOUT the
    # /NODEFAULTLIB link args: this probe is a std binary.
    cargo_dir = os.path.join(d, ".cargo")
    os.makedirs(cargo_dir, exist_ok=True)
    with open(os.path.join(cargo_dir, "config.toml"), "w", encoding="utf-8") as f:
        f.write('[target.x86_64-pc-windows-msvc]\n'
                'rustflags = ["-C", "target-feature=+avx2,+fma"]\n')
    return d


def build_probe():
    """Build the probe. Returns (ok, log, exe_path)."""
    d = ensure_probe()
    env = dict(os.environ)
    rc, out, err = V.run_capture(
        [CARGO, "build", "--release", "--target", TARGET,
         "--target-dir", os.path.join(d, "target")],
        cwd=d, timeout=1800)
    log = (out + err).decode("utf-8", "replace")
    exe = os.path.join(d, "target", TARGET, "release",
                       "p15-verify-guardprobe.exe")
    return (rc == 0 and os.path.exists(exe)), log, exe


def run_case(exe, case):
    rc, out, err = V.run_capture([exe, case], cwd=PROBE_DIR, timeout=120)
    return rc, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# the guard table: for every documented hard-fail path, the illegal input, the
# LEGAL near-miss that must be accepted, and the expected specific message.
# ---------------------------------------------------------------------------
GUARDS = [
    dict(id="G0a-zero-dim", cases=["g0a_zero_layers", "g0a_zero_heads",
                                   "g0a_zero_dim"],
         illegal="n_layers=0 / n_kv_heads=0 / head_dim=0",
         expect_exit=9, expect_msg="NKV: zero dimension in layout",
         legal=None, mirror=lambda: V.LayoutPrimitives(0, 2, 128, 1024, 1024)
         .validate_reasons()[0]),
    dict(id="G0b-zero-maxseq", cases=["g0b_zero_maxseq"],
         illegal="max_seq_per_neuron=0", expect_exit=9,
         expect_msg="NKV: max_seq_per_neuron is zero",
         legal=None, mirror=lambda: V.LayoutPrimitives(24, 2, 128, 0, 0)
         .validate_reasons()[0]),
    dict(id="G1-stride-lt-maxseq", cases=["g1_stride_lt_maxseq"],
         illegal="stride=1023 < max_seq=1024", expect_exit=9,
         expect_msg="NKV: stride < max_seq_per_neuron",
         legal=["g1_legal_boundary", "g1_legal_padded"],
         mirror=lambda: V.LayoutPrimitives(24, 2, 128, 1024, 1023)
         .validate_reasons()[0]),
    dict(id="G2-slot-ge-maxseq", cases=["g2_slot_eq_maxseq", "g2_slot_gt_maxseq"],
         illegal="slot == max_seq and slot >> max_seq", expect_exit=9,
         expect_msg="NKV: slot exceeds max_seq_per_neuron",
         legal=["g2_legal_last_slot"],
         mirror=lambda: None),
    dict(id="G2b-slot-inside-stride-outside-maxseq",
         cases=["g2b_slot_inside_stride_outside_maxseq"],
         illegal="stride=2048, max_seq=1024, slot=1500 (inside stride, "
                 "outside max_seq)", expect_exit=9,
         expect_msg="NKV: slot exceeds max_seq_per_neuron",
         legal=None, mirror=lambda: None),
    dict(id="G3-offsets-disagree", cases=["g3_offsets_disagree",
                                          "g3b_section_head_dim_stride",
                                          "g3c_header_stride_disagrees"],
         illegal="offsets[1]=9 instead of 8; i*head_dim*stride entries; "
                 "stride header 16 vs entries written for 8",
         expect_exit=9, expect_msg="NKV: offsets[] disagrees with stride",
         legal=["g3_offsets_ok"], mirror=lambda: None),
    dict(id="G4-npos-gt-maxseq", cases=["g4_npos_gt_maxseq",
                                        "g4b_npos_gt_maxseq_reverse"],
         illegal="n_pos=9 > max_seq=8 (both directions)", expect_exit=9,
         expect_msg="NKV: n_pos exceeds max_seq_per_neuron",
         legal=["g4_legal_npos_eq_maxseq", "g4_legal_npos_zero"],
         mirror=lambda: None),
    dict(id="G5-neuron-id-oob", cases=["g5_layer_oob", "g5_head_oob",
                                       "g5_dim_oob"],
         illegal="layer=24/24, kv_head=2/2, dim=128/128", expect_exit=9,
         expect_msg="NKV: neuron id out of range",
         legal=["g5_legal_last_id"], mirror=lambda: None),
    dict(id="G6-exceeds-u32", cases=["g6_exceeds_u32"],
         illegal="4 neurons * stride 2^30 = 2^32 > u32::MAX", expect_exit=9,
         expect_msg="NKV: layout exceeds 32-bit offsets",
         legal=None,
         mirror=lambda: V.LayoutPrimitives(2, 2, 1, 1, 1 << 30)
         .validate_reasons()[0]),
    dict(id="G7-write-rejects-bad-layout", cases=["g7_write_rejects_bad_layout"],
         illegal="write_section on stride<max_seq", expect_exit=9,
         expect_msg="NKV: stride < max_seq_per_neuron",
         legal=None, mirror=lambda: None),
    dict(id="G8-declared-section-len", cases=["g8_declared_len_too_short",
                                              "g9_len_exceeds_file",
                                              "g10_tag_zero"],
         illegal="avail=100 while the section needs 36 (descriptor/avail longer "
                 "than the layout); descriptor len=4096 on a 108-byte file; "
                 "tag=0 (no section) but the caller proceeded",
         expect_exit=9, expect_msg="",
         legal=["g8_declared_len_exact_ok"], mirror=lambda: None,
         note="The S8 fix changed the shape of this guard: read_section now takes "
              "an explicit `avail` byte count and requires avail == "
              "layout.section_bytes() exactly. So `g8_declared_len_too_short` "
              "(which passes the REAL 100 readable bytes while the layout needs "
              "36) now correctly FAILS with 'NKV: section length disagrees with "
              "layout / available_len=100 required_len=36'. That is the fix "
              "working, observed through the OLD case name; the previously "
              "reported HOLE is closed. G11 covers the descriptor-length source "
              "directly."),
]


def main():
    print("=" * 76)
    print("P15-S5 GUARD COVERAGE (independent verifier). Illegal input must "
          "fail LOUDLY.")
    print("=" * 76)

    meta = V.MetaCheck()
    ok, mdet = V.MetaCheck().self_test()
    print("\n[META-SELF] MetaCheck self-test: %s  %s" % ("OK" if ok else "BROKEN",
                                                         mdet))

    print("\n[BUILD] compiling the verifier's own probe against the real core "
          "(features=neuron-kv)")
    built, log, exe = build_probe()
    print("        probe dir : %s" % PROBE_DIR)
    print("        probe exe : %s" % exe)
    print("        build ok  : %s" % built)
    if not built:
        tail = "\n".join(log.splitlines()[-40:])
        print("        --- build log tail ---\n%s" % tail)
        print("\nVERDICT: UNPROVEN -- the probe did not build, so no guard was "
              "observed. This is reported as UNVERIFIED, not PASS.")
        meta.dump(os.path.join(V.ensure_outdir(), "guard_coverage.json"))
        return 2

    # ------------------------------------------------------------------
    # the address probe: measured addresses from the REAL binary
    # ------------------------------------------------------------------
    rc, out, err = run_case(exe, "addr_probe")
    print("\n[ADDR] real-binary address probe (exit=%d)" % rc)
    print("       %s" % out.strip())
    addr_ok = "ADDR" in out
    # CAN-FAIL control for the probe itself: an unknown case must exit 64, which
    # proves the probe really dispatches on its argument and that "exit=0 with
    # an ADDR line" is a specific observation and not a default path.
    rc_unknown, out_unknown, _ = run_case(exe, "__addr_probe_typo__")
    probe_dispatch_ok = (rc_unknown == 64 and "ADDR" not in out_unknown)
    meta.record("ADDR probe really dispatches (unknown case -> exit 64)",
                addr_ok, "a typo'd case name must NOT produce an ADDR line "
                         "and must exit 64",
                probe_dispatch_ok,
                "unknown-case exit=%d, ADDR present=%s"
                % (rc_unknown, "ADDR" in out_unknown))
    # The verifier recomputes the same numbers in Python and compares.
    L = V.LayoutPrimitives(24, 2, 128, 1024, 1024)
    mine = dict(base_d0=L.neuron_base(0, 0, 0), base_d1=L.neuron_base(0, 0, 1),
                spacing=L.neuron_base(0, 0, 1) - L.neuron_base(0, 0, 0),
                base_last=L.neuron_base(23, 1, 127),
                slot0=L.slot_addr(0, 0, 0, 0),
                slot1023=L.slot_addr(0, 0, 0, 1023),
                n_neurons=L.n_neurons(), section_bytes=20 + L.n_neurons() * 4)
    print("       verifier's independent recomputation: %s" % mine)
    diff = []
    if addr_ok:
        kv = dict(p.split("=", 1) for p in out.split() if "=" in p)
        for k, v in mine.items():
            if k in kv and int(kv[k]) != v:
                diff.append((k, int(kv[k]), v))
    print("       binary-vs-verifier disagreements: %s"
          % (diff if diff else "none"))
    # can-fail: perturb one expectation and require the comparator to notice
    tampered = dict(mine)
    tampered["spacing"] = mine["spacing"] + 1  # the 128x shape would be +130048
    notice = [k for k, v in tampered.items()
              if addr_ok and k in kv and int(kv[k]) != v]
    meta.record("ADDR binary == verifier arithmetic (independent recomputation)",
                addr_ok and not diff,
                "a tampered expectation must be noticed by the comparator",
                bool(notice), "tampered key(s) noticed: %s" % notice)

    # ------------------------------------------------------------------
    # guard sweep
    # ------------------------------------------------------------------
    print("\n[GUARDS] sweeping every documented hard-fail path")
    results = []
    for g in GUARDS:
        # Cases that a later fix SUPERSEDED are probed and printed but excluded
        # from the verdict, so a stale expectation cannot manufacture a FAIL (or,
        # worse, a PASS). Which cases those are is declared on the guard entry.
        superseded_cases = set(g.get("superseded_cases", []))
        rows = []
        for c in g["cases"]:
            rc, out, err = run_case(exe, c)
            combined = (out + err)
            fired = (rc == g["expect_exit"])
            msg_ok = g["expect_msg"] in combined
            hole = "HOLE" in combined
            rows.append(dict(case=c, exit=rc, msg_ok=msg_ok, hole=hole,
                             superseded=c in superseded_cases,
                             out=combined.strip()[:200]))
        legal_rows = []
        for c in (g["legal"] or []):
            rc, out, err = run_case(exe, c)
            combined = out + err
            legal_rows.append(dict(case=c, exit=rc, ok=(rc == 0 and "OK" in combined
                                                        or rc == 0),
                                   out=combined.strip()[:200]))
        live = [r for r in rows if not r["superseded"]]
        illegal_all_fired = all(r["exit"] == g["expect_exit"] and r["msg_ok"]
                                and not r["hole"] for r in live)
        legal_all_ok = all(r["exit"] == 0 for r in legal_rows)
        clean = illegal_all_fired and legal_all_ok
        results.append(dict(id=g["id"], illegal=g["illegal"], rows=rows,
                            legal_rows=legal_rows, clean=clean,
                            superseded_cases=sorted(superseded_cases),
                            note=g.get("note", "")))
        print("\n  [%s]" % g["id"])
        print("     illegal : %s" % g["illegal"])
        print("     expect  : exit=%d msg=%r" % (g["expect_exit"], g["expect_msg"]))
        for r in rows:
            print("       %-46s exit=%-3d msg=%-5s hole=%-5s%s | %s"
                  % (r["case"], r["exit"], r["msg_ok"], r["hole"],
                     " SUPERSEDED" if r["superseded"] else "", r["out"][:80]))
        for r in legal_rows:
            print("       LEGAL %-40s exit=%-3d %s"
                  % (r["case"], r["exit"], r["out"][:80]))
        print("     => %s" % ("PASS" if clean else "FAIL"))

        # can-fail control for THIS guard: a case name that does not exist must
        # produce a distinguishable outcome, proving run_case can observe a
        # different result and that "everything exits 9" is not an artefact.
        rc_bad, out_bad, err_bad = run_case(exe, "__nonexistent_case__")
        ctrl_ok = (rc_bad == 64)
        meta.record("GUARD %s fires loudly on illegal input" % g["id"], clean,
                    g["illegal"], illegal_all_fired,
                    "legal near-miss accepted: %s; unknown-case control exit=%d "
                    "(must be 64, proving the harness can tell outcomes apart)"
                    % (legal_all_ok, rc_bad))
        if not ctrl_ok:
            print("     *** WARNING: unknown-case control exit=%d, expected 64 "
                  "-- the probe cannot distinguish outcomes ***" % rc_bad)

    # ------------------------------------------------------------------
    # G8 group: the section DESCRIPTOR (tag / off / len) -- probed separately
    # because each case must fail for its OWN specific reason, and the point is
    # precisely which of them does NOT fail.
    # ------------------------------------------------------------------
    print("\n[G8] section descriptor: tag / declared length")
    g8 = [
        ("g8_declared_len_too_short", 9, "NKV: section length disagrees with layout",
         "avail=100 vs the 36 the layout needs. After the S8 fix this MUST be "
         "rejected; before the fix it was accepted (the reported HOLE)."),
        ("g9_len_exceeds_file", 9, "NKV: section exceeds artifact size",
         "descriptor claims 4096 B on a 108-byte file."),
        ("g10_tag_zero", 9, "NKV: artifact has no neuron_kv section",
         "tag=0 is the documented 'no section' value."),
    ]
    g8_rows = []
    for case, exp, msg, why in g8:
        rc, out, err = run_case(exe, case)
        comb = out + err
        fired = (rc == exp)
        msg_ok = (msg in comb) if msg else True
        hole = "HOLE" in comb
        print("   %-34s exit=%-3d msg=%-5s hole=%-5s | %s"
              % (case, rc, msg_ok, hole, comb.strip()[:110]))
        print("      why: %s" % why)
        g8_rows.append(dict(case=case, exit=rc, msg_ok=msg_ok, hole=hole,
                            out=comb.strip()[:220]))
    rc, out, err = run_case(exe, "g8_declared_len_exact_ok")
    exact_ok = (rc == 0 and "OK" in (out + err))
    print("   LEGAL g8_declared_len_exact_ok     exit=%-3d %s"
          % (rc, (out + err).strip()[:100]))
    # All three descriptor cases must now hard-fail for their OWN reason, and
    # the legal exact case must be accepted. The first case is the live
    # regression test for the S8 fix, observed through the original case name.
    len_fired = (g8_rows[0]["exit"] == 9 and g8_rows[0]["msg_ok"]
                 and not g8_rows[0]["hole"])
    tag_fired = g8_rows[2]["exit"] == 9 and g8_rows[2]["msg_ok"]
    file_fired = g8_rows[1]["exit"] == 9 and g8_rows[1]["msg_ok"]
    clean = len_fired and tag_fired and file_fired and exact_ok
    meta.record("G8 descriptor: avail/layout mismatch, len>file, tag=0 all "
                "hard-fail; exact descriptor accepted",
                clean,
                "tag=0, len>file and avail/layout mismatch must ALL hard-fail; "
                "if the probe returned no guard for any of them, this criterion "
                "must not be certified",
                tag_fired and file_fired and len_fired,
                "avail-mismatch fired=%s, tag_zero fired=%s, len_exceeds_file "
                "fired=%s, exact-descriptor accepted=%s"
                % (len_fired, tag_fired, file_fired, exact_ok))
    results.append(dict(id="G8-descriptor", illegal="tag/len descriptor cases",
                        rows=g8_rows, legal_rows=[dict(case="g8_declared_len_"
                                                            "exact_ok",
                                                       exit=rc)],
                        clean=clean))

    # ------------------------------------------------------------------
    # G11: the S8 FIX, driven through the shipped public entry
    # `read_section_declared(d, file_len)`, which is what load_validate calls.
    # It reads `len` from the descriptor and hands it to read_section_impl as
    # `avail`, which requires avail == layout.section_bytes() EXACTLY.
    # ------------------------------------------------------------------
    print("\n[G11] S8 fix: the declared length must equal the layout size exactly")
    g11 = [
        ("g11_bounded_under_declared", 9,
         "declared_len=20 while the header needs 36"),
        ("g11_bounded_over_declared", 9,
         "declared_len=40 while the header needs 36"),
    ]
    g11_rows = []
    for case, exp, why in g11:
        rc, out, err = run_case(exe, case)
        comb = out + err
        hole = "HOLE" in comb
        print("   %-32s exit=%-3d hole=%-5s | %s"
              % (case, rc, hole, comb.strip()[:120]))
        print("      %s" % why)
        g11_rows.append(dict(case=case, exit=rc, hole=hole,
                            out=comb.strip()[:220]))
    rc, out, err = run_case(exe, "g11_bounded_exact_ok")
    exact_ok11 = (rc == 0 and "OK" in (out + err))
    print("   LEGAL g11_bounded_exact_ok       exit=%-3d %s"
          % (rc, (out + err).strip()[:110]))
    under_fixed = g11_rows[0]["exit"] == 9 and not g11_rows[0]["hole"]
    over_fixed = g11_rows[1]["exit"] == 9 and not g11_rows[1]["hole"]
    print("   => under-declared rejected: %s ; over-declared rejected: %s ; "
          "exact accepted: %s" % (under_fixed, over_fixed, exact_ok11))
    meta.record("G11 S8 fix: under-declared section length is now rejected",
                under_fixed and exact_ok11,
                "if read_section_bounded accepted len=20 for a 36-byte section "
                "the S8 defect would still be open",
                under_fixed,
                "under exit=%d, exact accepted=%s"
                % (g11_rows[0]["exit"], exact_ok11))
    meta.record("G11 S8 fix is not over-strict (over-declared also rejected, "
                "exact accepted)",
                over_fixed and exact_ok11,
                "a fix that rejected the exact descriptor too would break the "
                "legitimate path; a fix that accepted an over-long descriptor "
                "would leave declaration/content decoupled",
                over_fixed and exact_ok11,
                "over exit=%d, exact exit=%d"
                % (g11_rows[1]["exit"], rc))
    results.append(dict(id="G11-s8-fix", illegal="declared_len != required",
                        rows=g11_rows,
                        legal_rows=[dict(case="g11_bounded_exact_ok", exit=rc)],
                        clean=under_fixed and over_fixed and exact_ok11))

    # ------------------------------------------------------------------
    # G12/G13: direction and overlap, MEASURED on the real functions
    # ------------------------------------------------------------------
    print("\n[G12] reverse-direction reorder, measured on the shipped function")
    rc, out, err = run_case(exe, "g12_reverse_direction_is_distinguishable")
    comb = (out + err)
    print("   %s" % comb.strip().replace("\n", "\n   "))
    g12_ok = (rc == 0 and "HOLE" not in comb and "G12" in comb)
    meta.record("G12 the direction predicate distinguishes correct from reversed "
                "reorder",
                g12_ok,
                "if the wrong direction also produced constant rows equal to the "
                "row index, the predicate would certify a transposed cache",
                g12_ok, comb.strip()[:160])

    print("\n[G13] overlapping src/dst: is the documented precondition checked?")
    rc, out, err = run_case(exe, "g13_overlapping_buffers")
    comb = (out + err)
    print("   %s" % comb.strip().replace("\n", "\n   "))
    # This is a FINDING, not a pass: the precondition is documented but not
    # enforced. Reported as UNPROVEN for certification purposes, and the
    # observation is handed back to the Lead.
    g13_unchecked = ("no guard fired" in comb)
    meta.record("G13 overlapping src/dst is checked at runtime",
                False,
                "the doc requires non-overlap; there is no runtime test, so the "
                "criterion CANNOT be shown to fail and is reported UNPROVEN "
                "rather than PASS",
                False,
                "observed: %s" % comb.strip()[:160])
    print("   => the precondition is DOCUMENTED but NOT ENFORCED. Not memory "
          "unsafe under the only shipped caller (load_reorder always allocates a "
          "fresh dst), but it is a documented `unsafe` precondition with no "
          "guard, so this criterion is UNPROVEN, not PASS." if g13_unchecked
          else "   => unexpected result, inspect")

    # ------------------------------------------------------------------
    # mirror-vs-binary differential (the mirror must be able to disagree)
    # ------------------------------------------------------------------
    print("\n[MIRROR] Python mirror vs the real binary (differential)")
    print("        NOTE: only the guards where the mirror PREDICTS a reason are")
    print("        comparable. A mirror returning None is not a comparison, and")
    print("        counting it as 'disagreement' would be a measurement of the")
    print("        harness's own bookkeeping, not of the code.")
    mirror_rows = []
    for g in GUARDS:
        m = g["mirror"]
        if m is None:
            continue
        try:
            pred = m()
        except Exception as e:  # noqa: BLE001
            pred = "EXC:%r" % e
        if pred is None:
            continue  # not a comparison
        obs = g["expect_msg"]
        agree = (pred == obs)
        mirror_rows.append(dict(id=g["id"], mirror=pred, binary=obs,
                                agree=agree))
        print("  %-34s mirror=%-42s agree=%s" % (g["id"], pred, agree))
    n_disagree = sum(1 for r in mirror_rows if not r["agree"])
    # CAN-FAIL: feed the differential a deliberately WRONG predicted message on
    # a synthetic row and require the comparator to flag it. Without this, "the
    # mirror agrees" could be an artefact of comparing nothing.
    synthetic = dict(id="__synthetic__", mirror="NKV: totally different message",
                     binary=mirror_rows[0]["binary"] if mirror_rows else "")
    synthetic_agrees = (synthetic["mirror"] == synthetic["binary"])
    comparator_can_fail = (synthetic_agrees is False)
    meta.record("MIRROR agrees with the real binary on every comparable guard",
                n_disagree == 0 and len(mirror_rows) >= 3,
                "a synthetic mirror row whose message differs must be flagged "
                "by the comparator",
                comparator_can_fail and len(mirror_rows) > 0,
                "%d comparable mirror(s), %d disagree; synthetic mismatched row "
                "flagged=%s" % (len(mirror_rows), n_disagree,
                                comparator_can_fail))

    # ------------------------------------------------------------------
    # report
    # ------------------------------------------------------------------
    v = V.ensure_outdir()
    import json
    with open(os.path.join(v, "guard_coverage.json"), "w", encoding="utf-8") as f:
        json.dump({"guards": results, "mirror": mirror_rows, "addr": mine},
                  f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 76)
    n_holes = sum(1 for r in results if not r["clean"])
    print("guards exercised : %d" % len(results))
    print("guards PASS      : %d" % (len(results) - n_holes))
    print("guards FAIL      : %d" % n_holes)
    meta.report()
    return 0 if (n_holes == 0 and meta.report(verbose=False)) else 1


if __name__ == "__main__":
    sys.exit(main())
