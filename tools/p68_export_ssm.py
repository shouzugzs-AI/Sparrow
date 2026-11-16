#!/usr/bin/env python
"""P62：把 p36 的「Quill 主干 + 选择性 SSM 状态」checkpoint 导出成引擎可读的 .s5q。

格式（小端，无 JSON 解析依赖）：
    magic  b"S5Q1"                     4 字节
    u32    d / layers / vocab / hidden / attn_every     5 个 u32
    u32    n_tensors
    每个张量：u32 name_len, name(utf-8), u32 off（以 float 为单位）, u32 numel
    f32    data（全部张量按顺序拼接）

另出参考文件（用于引擎侧对拍）：
    <tag>.tokens.bin   u32[T]      固定的验证序列（按 tag 命名，多模型不互相覆盖）
    <tag>.logits.bin   f32[T,V]    同一序列在 PyTorch 里的 logits
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "artifacts", "zh_wiki")
OUT36 = os.path.join(ROOT, "artifacts", "p36")


def load_p36_module():
    spec = importlib.util.spec_from_file_location(
        "p36", os.path.join(ROOT, "tools", "p36_attnonce.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="st_ssm_lr4e3_4000")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "artifacts", "ssm"))
    ap.add_argument("--ref-tokens", type=int, default=64)
    args = ap.parse_args()

    jsonl = os.path.join(OUT36, args.tag + ".jsonl")
    cfg = json.loads(open(jsonl, encoding="utf-8").readline())
    assert cfg["mix"] == "state" and cfg["state_kind"] == "ssm", cfg.get("mix")
    d, L, hidden = cfg["d"], cfg["layers"], cfg["hidden"]
    attn_every, inject, mlp = cfg["attn_every"], cfg["inject"], cfg["mlp"]
    assert inject == "gain" and mlp == "gated"

    ck = args.ckpt or os.path.join(OUT36, args.tag + ".pt")
    sd = torch.load(ck, map_location="cpu", weights_only=False)["model"]

    def f32(k):
        return sd[k].detach().to(torch.float32).contiguous()

    # A = softplus(logA) + 1e-4：在导出时算好，引擎侧直接用
    logA = f32("attns.0.logA")
    state_A = torch.nn.functional.softplus(logA) + 1e-4

    tensors = [
        ("emb", f32("emb.weight")),          # [V, d]（logits 与 embedding 共享）
        ("ln0", f32("ln0.w")),               # [d]
        ("state_A", state_A),                # [d]
        ("state_w", f32("attns.0.w.weight")),    # [d, d]
        ("state_wd", f32("attns.0.wd.weight")),  # [d, d]
        ("state_wdb", f32("attns.0.wd.bias")),   # [d]
        ("state_o", f32("attns.0.o.weight")),    # [d, d]
    ]
    for i in range(L):
        tensors += [
            ("l%d_inj" % i, f32("layers.%d.inj" % i)),
            ("l%d_ln1" % i, f32("layers.%d.ln1.w" % i)),
            ("l%d_up" % i, f32("layers.%d.up.weight" % i)),     # [I, d]
            ("l%d_gate" % i, f32("layers.%d.gate.weight" % i)),  # [I, d]
            ("l%d_down" % i, f32("layers.%d.down.weight" % i)),  # [d, I]
        ]
    tensors.append(("norm", f32("norm.w")))

    blob = []
    meta = []
    off = 0
    for name, t in tensors:
        a = t.numpy().astype("<f4").reshape(-1)
        meta.append({"name": name, "off": off, "numel": int(a.size)})
        blob.append(a)
        off += a.size
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, args.tag + ".s5q")
    with open(out, "wb") as f:
        f.write(b"S5Q1")
        f.write(np.asarray([d, L, cfg["vocab"], hidden, attn_every],
                           dtype="<u4").tobytes())
        f.write(np.uint32(len(meta)).tobytes())
        for m in meta:
            nb = m["name"].encode("utf-8")
            f.write(np.uint32(len(nb)).tobytes())
            f.write(nb)
            f.write(np.asarray([m["off"], m["numel"]], dtype="<u4").tobytes())
        for a in blob:
            f.write(a.tobytes())
    print("write", out, "%.2f MB" % (os.path.getsize(out) / 1048576))

    # ---- 参考 logits：用训练器本体算，避免"参考实现与被测实现同源错误" ----
    m36 = load_p36_module()
    model = m36.DynLM(cfg["vocab"], d, L, cfg["heads"], hidden,
                      attn_every=attn_every, share_attn=cfg["share_attn"],
                      mlp=mlp, route="none", route_gate="channel",
                      mix="state", state_kind="ssm")
    miss = model.load_state_dict(sd, strict=False)
    assert not miss.unexpected_keys, miss.unexpected_keys

    class C:
        pass
    c = C()
    c.route, c.inject, c.attn_every, c.share_attn = "none", inject, attn_every, 1
    c.mix, c.mask_src, c.keep, c.group_h, c.group_mid = "state", "none", 1.0, 1, 1
    c.keep_mid = 1.0
    c.mlp = mlp

    val = np.fromfile(os.path.join(DATA, "val_ids.bin"), dtype="<u4")
    toks = torch.from_numpy(val[: args.ref_tokens].astype(np.int64))[None]
    with torch.no_grad():
        logits = model(toks, c)[0].to(torch.float32).numpy()
    np.asarray(val[: args.ref_tokens], dtype="<u4").tofile(
        os.path.join(args.out_dir, args.tag + ".tokens.bin"))
    logits.astype("<f4").tofile(os.path.join(args.out_dir, args.tag + ".logits.bin"))
    print("ref tokens=%d logits=%s  first-row max=%.3f  argmax0=%d"
          % (args.ref_tokens, logits.shape, logits[0].max(), int(logits[0].argmax())))


if __name__ == "__main__":
    main()
