#!/usr/bin/env python
"""P62 调试用：把 PyTorch 参考实现的各层中间量 dump 成二进制，供引擎侧逐段对拍。

输出（artifacts/ssm/dbg_<tag>/）：
    dbg_A.bin       混合器输出 [T,d]
    dbg_x0.bin      第 0 层之前的残差流 [T,d]
    dbg_blk{i}.bin  第 i 层之后的残差流 [T,d]
    dbg_xn.bin      收尾 norm 之后 [T,d]
    dbg_logits.bin  最终 logits [T,V]
"""
import importlib.util
import json
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "artifacts", "zh_wiki")
OUT36 = os.path.join(ROOT, "artifacts", "p36")
OUT = os.path.join(ROOT, "artifacts", "ssm", "dbg")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="st_ssm_lr4e3_4000")
    ap.add_argument("--tokens", type=int, default=64)
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("p36", os.path.join(ROOT, "tools", "p36_attnonce.py"))
    m36 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m36)

    cfg = json.loads(open(os.path.join(OUT36, args.tag + ".jsonl"), encoding="utf-8").readline())
    sd = torch.load(os.path.join(OUT36, args.tag + ".pt"), map_location="cpu", weights_only=False)["model"]
    model = m36.DynLM(cfg["vocab"], cfg["d"], cfg["layers"], cfg["heads"], cfg["hidden"],
                      attn_every=cfg["attn_every"], share_attn=cfg["share_attn"],
                      mlp=cfg["mlp"], route="none", route_gate="channel",
                      mix="state", state_kind="ssm")
    model.load_state_dict(sd, strict=False)
    model.eval()

    class C:
        pass
    c = C()
    c.route, c.inject, c.attn_every, c.share_attn = "none", cfg["inject"], cfg["attn_every"], 1
    c.mix, c.mask_src, c.keep, c.group_h, c.group_mid = "state", "none", 1.0, 1, 1
    c.keep_mid, c.mlp = 1.0, cfg["mlp"]

    val = np.fromfile(os.path.join(DATA, "val_ids.bin"), dtype="<u4")
    toks = torch.from_numpy(val[: args.tokens].astype(np.int64))[None]

    out_dir = os.path.join(ROOT, "artifacts", "ssm", "dbg_" + args.tag)
    os.makedirs(out_dir, exist_ok=True)
    saves = {}

    # 用 forward hook 取真实 forward 的中间量（不要手工复算：手工复算漏一步就会把
    # 一个错误当成"参考真值"，正是 P61 §6 里 softplus 那个坑的同类）
    with torch.no_grad():
        saves = {}
        model.attns[0].register_forward_hook(
            lambda mod, inp, out: saves.__setitem__("A", out[0].detach()))
        for li, b in enumerate(model.layers):
            b.register_forward_hook(
                lambda mod, inp, out, li=li: saves.__setitem__("blk%d" % li, out[0].detach()))
        saves["x0"] = model.emb(toks)[0].detach()
        logits = model(toks, c)[0]
        xn = model.norm(saves["blk%d" % (cfg["layers"] - 1)][None])
        saves["xn"] = xn[0]
        saves["logits"] = logits
        # 第一层的分段量（用于定位是 norm / GEMM / 激活 哪一步）
        b0 = model.layers[0]
        x_inj = saves["x0"][None] + b0.inj * saves["A"][None]
        h0 = b0.ln1(x_inj)
        a0 = torch.nn.functional.silu(b0.gate(h0)) * b0.up(h0)
        saves["h0"] = h0[0]
        saves["act0"] = a0[0]
        saves["d0"] = b0.down(a0)[0]

    for k, v in saves.items():
        a = v.detach().to(torch.float32).numpy().astype("<f4").reshape(-1)
        a.tofile(os.path.join(out_dir, "dbg_%s.bin" % k))
    print("dump", len(saves), "个中间量 ->", out_dir)
    print("  A[0,:4]", saves["A"][0, :4].numpy())
    print("  blk11[0,:4]", saves["blk11"][0, :4].numpy())
    print("  logits[0,:4]", saves["logits"][0, :4].numpy(), "argmax", int(saves["logits"][0].argmax()))


if __name__ == "__main__":
    main()
