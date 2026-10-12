#!/usr/bin/env python3
"""
A2 / P8: BPTT training-step `.vml` generator.

Why a generator instead of hand-writing:
  The backward pass has identical structure per step, differing only in the step
  index. Hand-writing T copies invites silent misalignment -- and the engine does
  NO type checking, so a mismatch corrupts memory silently. The generator keeps
  "forward PUSH" and "backward POP" strictly paired, and makes T a parameter.

Outputs:
  bptt_t{T}.vml    -- the train step (forward + BPTT backward + AdamW)
  weights.args     -- assembler weight-injection args (relative paths)

Assemble with (paths must be relative; cd into this dir first):
  $a = @('bptt_t2.vml','bptt_t2.vmb') + (Get-Content weights.args)
  & <repo>/target/release/vmlm-asm.exe @a

--------------------------------------------------------------------------
Operator semantics (verified against core/src/interp.rs + core/src/autograd.rs)
--------------------------------------------------------------------------
  GEMM        default NT: C[M,N] = A[M,K] * B[N,K]^T   (B is nn.Linear [N,K])
  LN          dst, src, gamma, bias   (normalizes over src's row width)
  LN_BWD      dst=dx, dy, x, dg ; db is the slot right AFTER dg. dg/db ACCUMULATE,
              dx is OVERWRITTEN.
  ACT_BWD     dst=dx, dy, ypre, kind
  OUTER       S[i,j] += v[i]*k[j]     (v,k stored as row vectors [1,D])
  OUTER_BWD   dst=dv, ds, v, dk ; dv OVERWRITES, dk ACCUMULATES
  GEMM_BWD_W  dW += dy^T * x          (ACCUMULATES)
  GEMM_BWD_X  dx  = dy * W            (OVERWRITES)
  BIAS_BWD    db += sum_i dy[i]       (ACCUMULATES)
  SUM         dst[i] = sum_j a[i,j]   (reduces last dim)
  EXP/NEG/EWMUL/EWADD/COPY/TRANSPOSE   OVERWRITE
  PUSH/POP/STACK_INIT  3rd operand is a numeric step index

Because the *_BWD ops accumulate, every dW / dGrad / dBias is handled by the
engine; only dS and dh need manual cross-step accumulation.

--------------------------------------------------------------------------
Forward (per step), matching examples/rnn/ese_step.vml
--------------------------------------------------------------------------
  u     = LN_1(x_t)
  d     = h_{t-1} - u
  r     = sigmoid(W_r (u + d*xr))
  k     = W_k (u + d*xk)
  v     = W_v (u + d*xv)
  decay = exp(-exp(W2d tanh(W1d (u + d*xw)) + w0))
  S_t   = S_{t-1} * decay + v outer k        (decay scales S's ROW index i)
  o     = LN_t(S_t r)
  h_t   = h_{t-1} + W_o o
  logits_t = LN_out(h_t) E^T

--------------------------------------------------------------------------
Backward (per step t, from T-1 down to 0)
--------------------------------------------------------------------------
  CE:            dlogits_t = softmax(logits_t) - onehot(y_t)
  logits=LN(h)E^T: dE += dlogits^T hNorm ; dhNorm = dlogits E
                   dh += LN_BWD(dhNorm, h_t)
  h = h_prev + W_o o:  dW_o += dh^T o ; do = dh W_o ; (dh_prev gets +dh)
  o = LN_t(sr):    d(sr) = LN_BWD(do, sr)
  sr = S_t r:      dS += d(sr) outer r ;  dr = S_t^T d(sr)
  S_t = S_prev*decay + v outer k:
                   (dv,dk) = OUTER_BWD(dS, v, k)
                   dS_prev = dS * decay
                   ddecay[i] = sum_j dS[i,j] S_prev[i,j]
  decay=exp(-exp(w)):  dw = ddecay * decay * exp(w)
  w = W2d gdact + w0:  dW2d += dw^T gdact ; dw0 += dw ; dgdact = dw W2d
  gdact = tanh(gdpre): dgdpre = ACT_BWD(dgdact, gdpre, tanh)
  gdpre = W1d g32w:    dW1d += dgdpre^T g32w ; dg32w = dgdpre W1d
  g32w = u + d*xw:     du += dg32w ; dd += dg32w * xw ; dxw += dg32w * d
  (gates r,k,v handled identically; r adds an ACT_BWD through sigmoid)
  u = LN_1(x), d = h_prev - u:
                   du_total = du - dd ; dx = LN_BWD(du_total, x_t)
                   dh (to step t-1) += dd
"""
import os
import sys

T = int(sys.argv[1]) if len(sys.argv) > 1 else 2
H = int(sys.argv[2]) if len(sys.argv) > 2 else 16
D = int(sys.argv[3]) if len(sys.argv) > 3 else 8
V = int(sys.argv[4]) if len(sys.argv) > 4 else 4

DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DIR, f"bptt_t{T}.vml")

PARAMS = ["W_r", "W_k", "W_v", "W_o", "W1d", "W2d", "w0",
          "xr", "xk", "xv", "xw", "ln1_g", "ln1_b",
          "ln_t_g", "ln_t_b", "ln_out_g", "ln_out_b", "E"]

L = []


def e(s=""):
    L.append(s)


# BPTT stacks: (name, elements per step, comment)
STACKS = [
    ("stk_xt", H, "x_t (input to LN_1 backward)"),
    ("stk_d", H, "d_t (gate branch d*x*)"),
    ("stk_hprev", H, "h_{t-1} (residual + d = h_prev - u)"),
    ("stk_Sprev", D * D, "S_{t-1} (dS passthrough + ddecay)"),
    ("stk_Scur", D * D, "S_t (dr = S^T d(sr))"),
    ("stk_v", D, "v_t"),
    ("stk_k", D, "k_t"),
    ("stk_r", D, "r_t"),
    ("stk_rpre", D, "pre-sigmoid r"),
    ("stk_gpr", H, "gpr = u + d*xr"),
    ("stk_gpk", H, "gpk = u + d*xk"),
    ("stk_gpv", H, "gpv = u + d*xv"),
    ("stk_gdpre", 32, "pre-tanh"),
    ("stk_gdact", 32, "post-tanh"),
    ("stk_gpw", H, "gpw = u + d*xw"),
    ("stk_expom", D, "exp(w) for the decay chain"),
    ("stk_decay", D, "decay_t"),
    ("stk_oRow", D, "o_t (input to GEMM_BWD_W for W_o)"),
    ("stk_sr_t", D, "S_t r_t (input x to LN_t backward)"),
    ("stk_h", H, "h_t (input x to LN_out backward)"),
    ("stk_hNorm", H, "LN_out(h_t) (input x to dE's GEMM_BWD_W)"),
]

_shaped = set()


def sh(name, r, c):
    if name not in _shaped:
        _shaped.add(name)
        e(f"    SHAPE       {name}, {r}, {c}")


def sh_grad_pair(gamma, dg, db, width):
    """Declare a LayerNorm gamma together with its dgamma/dbias slots.

    CRITICAL (see interp.rs OP_LN_BWD): the engine reads the gamma slot index
    from the instruction's `dt` byte and then writes dgamma and dbias to
    slot_mut(gi + 1) and slot_mut(gi + 2). Any 5th operand written in the .vml
    source is IGNORED by the interpreter.

    So the three slots must occupy ADJACENT slot-table entries, in the order
    gamma, dgamma, dbias. Slot allocation follows first-mention order in the
    source, and the assembler auto-creates `g_<const>` gradient slots as soon
    as that name appears, so we lay them down here, before anything else
    mentions them, in exactly that order.
    """
    sh(gamma, 1, width)
    sh(dg, 1, width)
    sh(db, 1, width)


# ============================== header ==============================
e("; ============================================================================")
e("; A2 / P8: ESERNN BPTT training step")
e("; *** GENERATED by gen_bptt.py -- do not hand-edit ***")
e(";")
e(f"; config: T={T}  H={H}  D={D}  V={V}")
e(f"; regenerate: python gen_bptt.py {T} {H} {D} {V}")
e(";")
e("; Forward math follows examples/rnn/ese_step.vml; the backward style follows")
e("; examples/mnist/train_step.vml. BPTT uses the P8 STACK_INIT/PUSH/POP ops:")
e("; activation slots are a single shared buffer overwritten each step, so every")
e("; quantity the backward pass needs is PUSHed during the forward and POPped")
e("; back in reverse step order.")
e(";")
e("; Operator semantics (verified against core/src/interp.rs, core/src/autograd.rs):")
e(";   GEMM        default NT: C[M,N] = A[M,K] * B[N,K]^T")
e(";   LN          4 operands: LN dst, src, gamma, bias")
e(";   LN_BWD      dst=dx, dy, x, dg ; db is the slot right after dg")
e(";               (dg/db ACCUMULATE, dx is OVERWRITTEN)")
e(";   ACT_BWD     dst=dx, dy, ypre, kind")
e(";   OUTER       S[i,j] += v[i]*k[j]")
e(";   OUTER_BWD   dst=dv, ds, v, dk (dv OVERWRITES, dk ACCUMULATES)")
e(";   GEMM_BWD_W  dW += dy^T x   (accumulates)")
e(";   GEMM_BWD_X  dx  = dy W     (overwrites)")
e(";   BIAS_BWD    db += sum_i dy[i] (accumulates)")
e(";   PUSH/POP/STACK_INIT: 3rd operand is a numeric step index")
e("; ============================================================================")
e()
e("; Weights and inputs are injected by the assembler on the command line as")
e(";   name:file.bin:rows:cols      (f32 little-endian, relative path)")
e("; See weights.args in this directory, generated alongside this file.")
e()
e(".section state")
e(f"    SHAPE       st_S,     {D}, {D}     ; recurrent state S[D,D], persists across steps")
e(f"    SHAPE       st_hprev, 1, {H}    ; h from the previous step")
e()
e(".section forward")

# ============================== slot declarations ==============================
e("    ; ---------- per-step working slots ----------")
# Declare the three LayerNorm gamma/dgamma/dbias triples FIRST, so the allocator
# places gamma, dgamma, dbias in adjacent slot-table entries (OP_LN_BWD relies
# on that adjacency -- see sh_grad_pair above). The dgamma/dbias names are the
# assembler's own `g_<const>` gradient slots, so the driver can read them with
# the same naming convention it uses for every other parameter.
sh_grad_pair("ln1_g", "g_ln1_g", "g_ln1_b", H)
sh_grad_pair("ln_t_g", "g_ln_t_g", "g_ln_t_b", D)
sh_grad_pair("ln_out_g", "g_ln_out_g", "g_ln_out_b", H)
e("")
for nm, r, c in [("x_cur", 1, H), ("u", 1, H), ("d", 1, H), ("g32", 1, H),
                 ("dtimesx", 1, H), ("rpre", 1, D), ("r", 1, D), ("k", 1, D),
                 ("v", 1, D), ("gdpre", 1, 32), ("gdact", 1, 32), ("om", 1, D),
                 ("expom", 1, D), ("decay", 1, D), ("decayCol", D, 1),
                 ("oCol", D, 1), ("oRow", 1, D), ("woCol", H, 1), ("hpCol", H, 1),
                 ("hCol", H, 1), ("h", 1, H), ("hNorm", 1, H), ("sr", 1, D)]:
    sh(nm, r, c)
for t in range(T):
    sh(f"logits{t}", 1, V)
# per-gate temporaries (one set per gate, to avoid reuse bugs)
for gn in ("r", "k", "v"):
    sh(f"dtx_{gn}", 1, H)   # d * x{gn}
    sh(f"g{gn}", 1, H)      # u + d*x{gn}

e("")
e("    ; ---------- BPTT stacks: [T, elements-per-step] ----------")
for nm, w, _c in STACKS:
    e(f"    SHAPE       {nm}, {T}, {w}")

e("")
e("    ; ---------- backward working slots ----------")
for nm, r, c in [
    ("dh", 1, H), ("dhAdd", 1, H), ("dhNorm", 1, H),
    ("doRow", 1, D), ("dlnoRow", 1, D), ("dlnoCol", D, 1),
    ("dS", D, D), ("dSprev", D, D), ("dSwork", D, D), ("dSworkT", D, D),
    ("S_tT", D, D), ("decayCol", D, 1),
    ("ddecay", 1, D), ("dom", 1, D), ("ddom", 1, D),
    ("dv", 1, D),
    ("drRow", 1, D), ("drpre", 1, D),
    ("dgr", 1, H), ("dgk", 1, H), ("dgv", 1, H),
    ("dgdact", 1, 32), ("dgdpre", 1, 32),
    ("dg32w", 1, H),
    ("tmpXw", 1, H), ("tmpDw", 1, H),
    ("tmpXr", 1, H), ("tmpDr", 1, H),
    ("tmpXk", 1, H), ("tmpDk", 1, H),
    ("tmpXv", 1, H), ("tmpDv", 1, H),
    ("dU", 1, H), ("dUtot", 1, H), ("dDtot", 1, H), ("dDneg", 1, H),
    ("dxln", 1, H),
    ("dlogits", 1, V),
]:
    sh(nm, r, c)
for t in range(T):
    sh(f"loss_out{t}", 1, V)

# IMPORTANT: POP does not propagate shapes in the assembler, so every POP
# destination needs an explicit SHAPE. Declare them all here (shapes taken from
# the forward pass, which is where the value was pushed).
#
# `k_t` and `dkAcc` are declared FIRST and back-to-back: OUTER_BWD takes dk
# from `k_slot + 1` (see asm/src/main.rs OP_OUTER_BWD), so the two must be
# adjacent in that order. The assembler now validates this and panics otherwise.
sh("k_t", 1, D)
sh("dkAcc", 1, D)
sh("dkAccT", 1, D)
for _nm, _r, _c in [
    ("hNorm", 1, H), ("h_t", 1, H), ("oRow", 1, D), ("sr_t", 1, D),
    ("r_t", 1, D), ("S_t", D, D), ("v_t", 1, D),
    ("decay_t", 1, D), ("S_prev", D, D), ("expom_t", 1, D),
    ("gdact_t", 1, 32), ("gdpre_t", 1, 32), ("gpw_t", 1, H),
    ("d_t", 1, H), ("rpre_t", 1, D), ("gpr_t", 1, H),
    ("gpk_t", 1, H), ("gpv_t", 1, H), ("x_t", 1, H),
]:
    sh(_nm, _r, _c)

# ============================== prologue ==============================
e("")
e("    LN_EPS      eps1e6")
e("")
e("    ; ---------- zero gradients ----------")
for g in PARAMS:
    e(f"    ZERO        g_{g}")
e("")
e("    ; ---------- reset stacks and state (a train step is self-contained) ----------")
for nm, _w, _c in STACKS:
    e(f"    STACK_INIT  {nm}, {T}")
e("    ZERO        st_S")
e("    ZERO        st_hprev")

# ============================== forward ==============================
for t in range(T):
    e("")
    e("    ; ================================================================")
    e(f"    ;  FORWARD step {t}")
    e("    ; ================================================================")
    e(f"    COPY        x_cur, x{t}")
    e(f"    LN          u, x_cur, ln1_g, ln1_b")
    e(f"    EWSUB       d, st_hprev, u")
    e(f"    PUSH        stk_xt,       x_cur,    {t}")
    e(f"    PUSH        stk_d,        d,        {t}")
    e(f"    PUSH        stk_hprev,    st_hprev, {t}")
    e(f"    PUSH        stk_Sprev,    st_S,     {t}")
    e("")
    e("    ; r = sigmoid(W_r (u + d*xr))")
    e(f"    EWMUL       dtx_r, d, xr")
    e(f"    EWADD       gr, u, dtx_r")
    e(f"    PUSH        stk_gpr, gr, {t}")
    e(f"    GEMM        rpre, gr, W_r")
    e(f"    PUSH        stk_rpre, rpre, {t}")
    e(f"    ACT         r, rpre, sigmoid")
    e(f"    PUSH        stk_r, r, {t}")
    e("")
    e("    ; k = W_k (u + d*xk)")
    e(f"    EWMUL       dtx_k, d, xk")
    e(f"    EWADD       gk, u, dtx_k")
    e(f"    PUSH        stk_gpk, gk, {t}")
    e(f"    GEMM        k, gk, W_k")
    e(f"    PUSH        stk_k, k, {t}")
    e("")
    e("    ; v = W_v (u + d*xv)")
    e(f"    EWMUL       dtx_v, d, xv")
    e(f"    EWADD       gv, u, dtx_v")
    e(f"    PUSH        stk_gpv, gv, {t}")
    e(f"    GEMM        v, gv, W_v")
    e(f"    PUSH        stk_v, v, {t}")
    e("")
    e("    ; decay = exp(-exp(W2d tanh(W1d (u + d*xw)) + w0))")
    e(f"    EWMUL       dtimesx, d, xw")
    e(f"    EWADD       g32, u, dtimesx")
    e(f"    PUSH        stk_gpw, g32, {t}")
    e(f"    GEMM        gdpre, g32, W1d")
    e(f"    PUSH        stk_gdpre, gdpre, {t}")
    e(f"    COPY        gdact, gdpre")
    e(f"    ACT         gdact, gdact, tanh")
    e(f"    PUSH        stk_gdact, gdact, {t}")
    e(f"    GEMM        om, gdact, W2d")
    e(f"    EWADD       om, om, w0")
    e(f"    EXP         expom, om")
    e(f"    PUSH        stk_expom, expom, {t}")
    e(f"    NEG         expom, expom")
    e(f"    EXP         decay, expom")
    e(f"    PUSH        stk_decay, decay, {t}")
    e("")
    e("    ; S_t = S_{t-1} * decay + v outer k   (decay scales the row index i)")
    e(f"    TRANSPOSE   decayCol, decay")
    e(f"    EWMUL       st_S, st_S, decayCol")
    e(f"    OUTER       st_S, v, k")
    e(f"    PUSH        stk_Scur, st_S, {t}")
    e("")
    e("    ; o = LN_t(S r) ; keep S r for the LN_t backward")
    e(f"    GEMM        oCol, st_S, r")
    e(f"    TRANSPOSE   oRow, oCol")
    e(f"    COPY        sr, oRow")
    e(f"    PUSH        stk_sr_t, sr, {t}")
    e(f"    LN          oRow, oRow, ln_t_g, ln_t_b")
    e(f"    PUSH        stk_oRow, oRow, {t}")
    e("")
    e("    ; h_t = h_{t-1} + W_o o")
    e(f"    GEMM        woCol, W_o, oRow")
    e(f"    TRANSPOSE   hpCol, st_hprev")
    e(f"    EWADD       hCol, hpCol, woCol")
    e(f"    TRANSPOSE   h, hCol")
    e(f"    COPY        st_hprev, h")
    e(f"    PUSH        stk_h, h, {t}")
    e("")
    e("    ; logits = LN_out(h) E^T")
    e(f"    LN          hNorm, h, ln_out_g, ln_out_b")
    e(f"    PUSH        stk_hNorm, hNorm, {t}")
    e(f"    GEMM        logits{t}, hNorm, E")

# ============================== backward ==============================
e("")
e("    ; ================================================================")
e(f"    ;  BACKWARD: step {T-1} down to 0")
e("    ;  dh and dS are cross-step accumulators: on entry to step t they")
e("    ;  already hold the contribution arriving from step t+1.")
e("    ; ================================================================")
e("    ZERO        dS")
e("    ZERO        dh")
e("")
e("    ; only the final step's CE is surfaced (vm.loss holds the last one)")

for t in reversed(range(T)):
    e("")
    e(f"    ; ---------------- BACKWARD step {t} ----------------")
    e(f"    CE_BWD      logits{t}, dlogits, y{t}")
    e("")
    e("    ; logits = LN_out(h) E^T")
    e(f"    POP         hNorm, stk_hNorm, {t}")
    e(f"    GEMM_BWD_W  g_E, dlogits, hNorm")
    e(f"    GEMM_BWD_X  dhNorm, dlogits, E")
    e(f"    POP         h_t, stk_h, {t}")
    e(f"    LN_BWD      dhAdd, dhNorm, h_t, ln_out_g")
    e(f"    EWADD       dh, dh, dhAdd        ; dh = total dh_t")
    e("")
    e("    ; h_t = h_{t-1} + W_o o  (residual: dh flows straight to h_{t-1})")
    e("    ;")
    e("    ; GEMM_BWD_X gives dx[M,K] = dy[M,N] . W[N,K], with (M,N) taken from")
    e("    ; dy and K from W's columns. Feeding dy = dh as a ROW [1,H] and")
    e("    ; W = W_o[H,D] therefore yields do[1,D] -- exactly W_o^T dh.")
    e("    ; (Passing dh as a COLUMN [H,1] instead makes M=H,N=1 and asks the")
    e("    ;  kernel for a [H,D] result, which overruns do's buffer.)")
    e(f"    POP         oRow, stk_oRow, {t}")
    e(f"    GEMM_BWD_W  g_W_o, dh, oRow")
    e(f"    GEMM_BWD_X  doRow, dh, W_o")
    e("")
    e("    ; o = LN_t(sr)")
    e(f"    POP         sr_t, stk_sr_t, {t}")
    e(f"    LN_BWD      dlnoRow, doRow, sr_t, ln_t_g")
    e("")
    e("    ; sr = S_t r:  dS += d(sr) outer r ;  dr = d(sr) . S_t^T")
    e("    ;")
    e("    ; dS[i,j] = d(sr)[i] * r[j]      -> OUTER (which is += , so it stacks")
    e("    ;                                   onto any dS arriving from step t+1)")
    e("    ; dr[j]   = sum_i d(sr)[i]*S_t[i,j]")
    e("    ;         = d(sr)[1,D] . S_t[D,D]^T   -> GEMM NT: A=dlnoRow, B=S_t")
    e("    ;           (NT gives C[M,N] = A[M,K] B[N,K]^T with N = rows(B) = D)")
    e(f"    POP         r_t, stk_r, {t}")
    e(f"    POP         S_t, stk_Scur, {t}")
    e(f"    TRANSPOSE   dlnoCol, dlnoRow")
    e(f"    OUTER       dS, dlnoCol, r_t")
    e(f"    TRANSPOSE   S_tT, S_t")
    e(f"    GEMM        drRow, dlnoRow, S_tT      ; dr = d(sr) . S_t^T")
    e("")
    e("    ; S_t = S_prev * decay + v outer k")
    e("    ;")
    e("    ; (dv, dk) = OUTER_BWD(dS, v, k)")
    e("    ;")
    e("    ; Calling convention (P8, fixed): OUTER_BWD dv, ds, v, k [, dk]")
    e("    ;   the interpreter takes k from ins.dt and dk from ins.dt+1, so")
    e("    ;   k and dk MUST be adjacent slots in the order k, dk.")
    e("    ;   (Before the P8 fix the assembler never encoded dt for OUTER_BWD,")
    e("    ;   so dk/dv silently landed in slot 0/1 = st_S/st_hprev and the")
    e("    ;   k-gate gradient was identically zero.)")
    e("    ; dv is overwritten and dk accumulates; both are step-local here")
    e("    ; because v and k are recomputed every step, so we zero dk first.")
    e(f"    POP         v_t, stk_v, {t}")
    e(f"    POP         k_t, stk_k, {t}")
    e(f"    POP         decay_t, stk_decay, {t}")
    e(f"    POP         S_prev, stk_Sprev, {t}")
    e(f"    ZERO        dkAcc")
    e(f"    OUTER_BWD   dv, dS, v_t, k_t, dkAcc")
    e(f"    TRANSPOSE   decayCol, decay_t")
    e(f"    EWMUL       dSprev, dS, decayCol")
    e(f"    EWMUL       dSwork, dS, S_prev")
    e(f"    ; ddecay[i] = sum_j dS[i,j]*S_prev[i,j]  (sum over the COLUMN index)")
    e(f"    ; SUM reduces the LAST dimension, so feed dSwork directly. Inserting a")
    e(f"    ; TRANSPOSE here sums over i instead and silently yields a wrong but")
    e(f"    ; plausible per-channel decay gradient.")
    e(f"    SUM         ddecay, dSwork")
    e("")
    e("    ; decay = exp(-exp(w))")
    e("    ;   let e = exp(w);  decay = exp(-e)")
    e("    ;   d(decay)/dw = exp(-e) * (-1) * de/dw = -decay * exp(w)")
    e("    ; The MINUS is easy to drop: the forward has TWO nested exps, and the")
    e("    ; inner one contributes -1 to the chain. Forgetting it flips the sign")
    e("    ; of every gradient that flows through decay.")
    e("    ;   dom = -(ddecay * decay) * exp(w)")
    e(f"    POP         expom_t, stk_expom, {t}")
    e(f"    EWMUL       ddom, ddecay, decay_t")
    e(f"    EWMUL       ddom, ddom, expom_t")
    e(f"    NEG         dom, ddom")
    e("")
    e("    ; w = W2d gdact + w0")
    e(f"    POP         gdact_t, stk_gdact, {t}")
    e(f"    GEMM_BWD_W  g_W2d, dom, gdact_t")
    e(f"    BIAS_BWD    g_w0, dom")
    e(f"    GEMM_BWD_X  dgdact, dom, W2d      ; dgdact[1,32] = dom . W2d")
    e("")
    e("    ; gdact = tanh(gdpre)")
    e(f"    POP         gdpre_t, stk_gdpre, {t}")
    e(f"    ACT_BWD     dgdpre, dgdact, gdpre_t, tanh")
    e("")
    e("    ; gdpre = W1d gpw , gpw = u + d*xw")
    e(f"    POP         gpw_t, stk_gpw, {t}")
    e(f"    GEMM_BWD_W  g_W1d, dgdpre, gpw_t")
    e(f"    GEMM_BWD_X  dg32w, dgdpre, W1d    ; dg32w[1,16] = dgdpre . W1d")
    e(f"    POP         d_t, stk_d, {t}")
    e(f"    EWMUL       tmpXw, dg32w, d_t")
    e(f"    BIAS_BWD    g_xw, tmpXw")
    e(f"    ZERO        dU")
    e(f"    EWADD       dU, dU, dg32w")
    e(f"    ZERO        dDtot")
    e(f"    EWMUL       tmpDw, dg32w, xw")
    e(f"    EWADD       dDtot, dDtot, tmpDw")
    e("")
    e("    ; gate r: r = sigmoid(W_r (u + d*xr))")
    e(f"    POP         rpre_t, stk_rpre, {t}")
    e(f"    ACT_BWD     drpre, drRow, rpre_t, sigmoid")
    e(f"    POP         gpr_t, stk_gpr, {t}")
    e(f"    GEMM_BWD_W  g_W_r, drpre, gpr_t")
    e(f"    GEMM_BWD_X  dgr, drpre, W_r      ; dgr[1,16] = drpre . W_r")
    e(f"    EWMUL       tmpXr, dgr, d_t")
    e(f"    BIAS_BWD    g_xr, tmpXr")
    e(f"    EWMUL       tmpDr, dgr, xr")
    e(f"    EWADD       dDtot, dDtot, tmpDr")
    e(f"    EWADD       dU, dU, dgr")
    e("")
    e("    ; gate k: k = W_k (u + d*xk)")
    e(f"    POP         gpk_t, stk_gpk, {t}")
    e(f"    GEMM_BWD_W  g_W_k, dkAcc, gpk_t")
    e(f"    GEMM_BWD_X  dgk, dkAcc, W_k      ; dgk[1,16] = dkAcc . W_k")
    e(f"    EWMUL       tmpXk, dgk, d_t")
    e(f"    BIAS_BWD    g_xk, tmpXk")
    e(f"    EWMUL       tmpDk, dgk, xk")
    e(f"    EWADD       dDtot, dDtot, tmpDk")
    e(f"    EWADD       dU, dU, dgk")
    e("")
    e("    ; gate v: v = W_v (u + d*xv)")
    e(f"    POP         gpv_t, stk_gpv, {t}")
    e(f"    GEMM_BWD_W  g_W_v, dv, gpv_t")
    e(f"    GEMM_BWD_X  dgv, dv, W_v         ; dgv[1,16] = dv . W_v")
    e(f"    EWMUL       tmpXv, dgv, d_t")
    e(f"    BIAS_BWD    g_xv, tmpXv")
    e(f"    EWMUL       tmpDv, dgv, xv")
    e(f"    EWADD       dDtot, dDtot, tmpDv")
    e(f"    EWADD       dU, dU, dgv")
    e("")
    e("    ; u = LN_1(x) , d = h_{t-1} - u")
    e("    ; du_total = dU - dDtot   (d is subtracted from u)")
    e(f"    NEG         dDneg, dDtot")
    e(f"    EWADD       dUtot, dU, dDneg")
    e(f"    POP         x_t, stk_xt, {t}")
    e(f"    LN_BWD      dxln, dUtot, x_t, ln1_g")
    e("    ; dh passed to step t-1: residual dh plus the d-branch contribution")
    e(f"    EWADD       dh, dh, dDtot")
    e(f"    COPY        dS, dSprev")

# ============================== optimizer ==============================
e("")
e("    ; ================================================================")
e("    ;  gradient clipping + AdamW update")
e("    ;  hparams = [lr, b1, b2, eps, wd]")
e("    ; ================================================================")
for g in PARAMS:
    e(f"    CLIP_GRAD   g_{g}, clip_norm")
e("")
for g in PARAMS:
    e(f"    ADAMW       {g}, g_{g}, hparams")
e("")
e("    ; Surface the per-step logits so the driver can compute the mean CE over")
e("    ; all T steps. NOTE: CE_BWD writes the engine's vm.loss, and because the")
e("    ; backward pass runs from step T-1 down to 0, vm.loss ends up holding the")
e("    ; CE of **step 0**, not the last step. So the driver recomputes the loss")
e("    ; from these logits instead of trusting vm.loss.")
for t in range(T):
    e(f"    COPY        loss_out{t}, logits{t}")
e("    HALT")

# ============================== write ==============================
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(L) + "\n")

WA = [f"W_r:W_r.bin:{D}:{H}", f"W_k:W_k.bin:{D}:{H}", f"W_v:W_v.bin:{D}:{H}",
      f"W_o:W_o.bin:{H}:{D}", f"W1d:W1d.bin:32:{H}", f"W2d:W2d.bin:{D}:32",
      f"w0:w0.bin:1:{D}"]
for nm in ("xr", "xk", "xv", "xw", "ln1_g", "ln1_b", "ln_out_g", "ln_out_b"):
    WA.append(f"{nm}:{nm}.bin:1:{H}")
for nm in ("ln_t_g", "ln_t_b"):
    WA.append(f"{nm}:{nm}.bin:1:{D}")
WA.append(f"E:E.bin:{V}:{H}")
for t in range(T):
    WA.append(f"x{t}:x_t{t}.bin:1:{H}")
for t in range(T):
    WA.append(f"y{t}:y{t}.bin:1:1")
WA += ["hparams:hparams.bin:5:1", "clip_norm:clip_norm.bin:1:1",
       "eps1e6:eps1e6.bin:1:1"]

wa_path = os.path.join(DIR, "weights.args")
with open(wa_path, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(WA) + "\n")

print(f"[gen_bptt] wrote {OUT} ({len(L)} lines)", file=sys.stderr)
print(f"[gen_bptt] wrote {wa_path} ({len(WA)} entries)", file=sys.stderr)
