# -*- coding: utf-8 -*-
"""本地实时监控页：http://127.0.0.1:8791/

把 artifacts/p35 下正在跑/刚跑完的臂渲染成自动刷新的 HTML：
进度条、当前 lr、train loss、最近一次 val、最近日志尾部、ckpt 时间。
只读文件，不干扰训练。
"""
import glob
import html
import http.server
import json
import os
import re
import socketserver
import time

ROOT = r"D:\czsai\vmlm\artifacts\p35"
ROOT36 = r"D:\czsai\vmlm\artifacts\p36"
TAGS = ["ctx6_state_adapt_lr2e3_4000", "ctx6_sa_s2_244", "qk2_lr1e3_1000",
        "mb_h01_244", "mb_h05_600", "mb_L24_k25_244",
        "ctx6_state_adapt_lr1e3_4000", "ctx6_sa_lr1e3_1000", "ctx6_lr1e3_1000",
        "ctx6_sa_lr2e3_1000", "qk3_lr1e3_1000",
        "c10_lr3e3_1000", "c15_lr3e3_4000", "d256_lr1e3_4000", "d192_lr1e3_t4000"]
TITLE = {
    "d192_lr1e3_t4000": "稠密 d=192 lr1e-3",
    "c15_lr3e3_4000": "C15 lr3e-3",
    "c15_lr5e3_4000": "C15 lr5e-3",
    "d256_lr1e3_4000": "稠密 d=256 lr1e-3",
    "c5_unit_lr5e3_600": "C5 逐通道调度 lr5e-3（600 步 A/B）",
    "c1_unit_lr5e3_600": "C1 逐通道调度 lr5e-3（600 步 A/B）",
    "Lk_L6_k10": "L=6 keep=10%（覆盖 47%，244 步）",
    "Lk_L24_k25": "L=24 keep=2.5%（覆盖 46%，244 步）",
    "Lk_L24_d120_k25": "L=24 d=120 keep=2.5%（等 MAC）",
    "m_h01": "keep=1% 覆盖度重测",
    "m_h02": "keep=2% 覆盖度重测",
    "d224_lr1e3_1000": "稠密 d=224 lr1e-3（Quill-1 等 MAC 对照）",
    "d224_lr2e3_1000": "稠密 d=224 lr2e-3（等 MAC 对照）",
    "ctx6_c128": "ctx6×128 + FFN + 每层读出（主提案）",
    "ctx6_c128_shared": "ctx6×128 共用读出",
    "ctx6_c128_noffn": "ctx6×128 纯注意力叠层",
    "ctx6_rb3": "ctx6×128 + 回读（每 3 层）",
    "ctx6_rb1": "ctx6×128 + 回读（每层）",
    "ctx6_state_adapt_lr1e3_4000": "ctx 接口 4000 步 lr1e-3（决定性）",
    "ctx6_sa_lr1e3_1000": "ctx 状态+适配器 lr1e-3（1000 步）",
    "ctx6_lr1e3_1000": "ctx6 线性读出 lr1e-3（1000 步）",
    "ctx6_sa_lr2e3_1000": "ctx 状态+适配器 lr2e-3（1000 步）",
    "ctx6_sa_s2_244": "ctx 状态+适配器 seed2（噪声地板）",
    "ctx6_state_adapt_lr2e3_4000": "ctx 接口 4000 步 @lr2e-3（决定性）",
    "mb_h01_244": "keep=1% + 掩码负载均衡",
    "mb_h05_600": "keep=5% + 掩码负载均衡",
    "mb_L24_k25_244": "L=24 keep=2.5% + 负载均衡",
    "d256_lr2e3_1000": "稠密 d=256 lr2e-3（lr 括号）",
    "c10_lr3e3_1000": "C10 keep=10% lr3e-3（补点）",
    "d192_lr1e3_2000": "稠密 d=192 lr1e-3 2000 步",
    "hi_c15_lr3e3": "C15 lr3e-3 细粒度（eval/25 步）",
    "hi_c15_lr2e3": "C15 lr2e-3 细粒度（eval/25 步）",
}
STEP_RE = re.compile(r"^\s+step\s+(\d+)\s+lr\s+([0-9.eE+-]+)\s+train\s+([0-9.]+)\s+wall\s+([0-9.]+)")
VAL_RE = re.compile(r"val_loss=([0-9.]+)")
POLL = 5


def _dir(tag):
    return ROOT36 if tag.startswith(("ctx", "q1", "qk", "b_", "c_L", "y_")) else ROOT


def arm_state(tag):
    log = os.path.join(_dir(tag), tag + ".log")
    jl = os.path.join(_dir(tag), tag + ".jsonl")
    st = dict(tag=tag, title=TITLE.get(tag, tag), exists=os.path.exists(log),
              step=0, total=4000, lr="", train="", val="", wall=0.0, done=False,
              tail=[], started="", state="未知")
    if not st["exists"]:
        st["state"] = "未开始"
        return st
    try:
        lines = open(log, encoding="utf-8", errors="replace").read().split("\n")
    except OSError:
        return st
    st["tail"] = [l for l in lines[-8:] if l.strip()]
    for l in lines:
        m = STEP_RE.match(l)
        if m:
            st["step"] = int(m.group(1))
            st["lr"] = m.group(2)
            st["train"] = m.group(3)
            st["wall"] = float(m.group(4))
        mv = VAL_RE.search(l)
        if mv:
            st["val"] = mv.group(1)
        if l.startswith("[done]"):
            st["done"] = True
    if os.path.exists(jl):
        try:
            rs = [json.loads(x) for x in open(jl, encoding="utf-8") if x.strip()]
            cfgs = [r for r in rs if r.get("kind") == "cfg"]
            if cfgs:
                st["total"] = int(cfgs[0].get("steps") or st["total"])
                st["started"] = ""
            evs = [r for r in rs if r.get("kind") == "eval"]
            if evs:
                st["val"] = "%.4f" % evs[-1]["val_loss"]
        except Exception:
            pass
    if st["done"]:
        st["state"] = "完成"
    elif st["step"] > 0:
        st["state"] = "跑中"
    else:
        st["state"] = "启动中"
    return st


def ckpts():
    out = []
    for p in glob.glob(os.path.join(ROOT, "*.pt")):
        out.append((os.path.basename(p), os.path.getmtime(p), os.path.getsize(p) / 1e6))
    out.sort(key=lambda x: -x[1])
    return out[:5]


def pycount():
    # 只数本项目的训练进程（按命令行匹配 tag 前缀不可得，这里给进程总数作为粗略参考）
    try:
        import subprocess
        r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe"],
                           capture_output=True, text=True, timeout=5)
        return sum(1 for l in r.stdout.split("\n") if l.lower().startswith("python.exe"))
    except Exception:
        return -1


def page():
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for t in TAGS:
        s = arm_state(t)
        pct = 100.0 * s["step"] / max(1, s["total"])
        barw = int(pct / 2.5)
        bar = "█" * barw + "·" * (40 - barw)
        colors = {"跑中": "#2563eb", "完成": "#059669", "未开始": "#9ca3af", "启动中": "#d97706", "未知": "#9ca3af"}
        c = colors.get(s["state"], "#9ca3af")
        rows.append(f"""
  <div class="card">
    <div class="head"><b>{html.escape(s['title'])}</b>
      <span class="tag">{html.escape(t)}</span>
      <span class="state" style="color:{c}">{s['state']}</span></div>
    <div class="bar"><span style="width:{pct:.1f}%"></span></div>
    <div class="meta">{s['step']} / {s['total']} 步　{pct:.1f}%　lr={s['lr'] or '—'}　train={s['train'] or '—'}　
      <b>val={s['val'] or '—'}</b>　已用 {s['wall']/3600:.2f} h</div>
    <pre>{html.escape(chr(10).join(s['tail']))}</pre>
  </div>""")
    ck = "".join(f"<li>{html.escape(n)} — {time.strftime('%H:%M:%S', time.localtime(m))}, {sz:.0f} MB</li>"
                 for n, m, sz in ckpts())
    body = "".join(rows)
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="{POLL}">
<title>训练监控</title>
<style>
body{{font:14px/1.5 -apple-system,Segoe UI,Microsoft YaHei,sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:18px 22px}}
h1{{font-size:17px;margin:0 0 4px}} .sub{{color:#94a3b8;font-size:12px;margin-bottom:14px}}
.card{{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:12px 14px;margin-bottom:12px}}
.head{{display:flex;gap:10px;align-items:baseline}} .tag{{color:#94a3b8;font-size:12px;font-family:ui-monospace,Consolas,monospace}}
.state{{margin-left:auto;font-weight:600}}
.bar{{height:9px;background:#0f172a;border-radius:5px;overflow:hidden;margin:9px 0 7px}}
.bar span{{display:block;height:100%;background:linear-gradient(90deg,#2563eb,#22d3ee)}}
.meta{{color:#cbd5e1;font-size:12.5px}}
pre{{background:#0b1220;border-radius:6px;padding:8px 10px;margin:9px 0 0;color:#a7b4c8;
     font:12px/1.45 ui-monospace,Consolas,monospace;white-space:pre-wrap;max-height:190px;overflow:auto}}
ul{{margin:6px 0 0 18px;padding:0;color:#94a3b8;font-size:12px}}
</style></head><body>
<h1>VMLM 训练实时监控</h1>
<div class="sub">刷新 {now}（每 {POLL}s）　日志目录 {html.escape(ROOT)}　/python 进程数 {pycount()}</div>
{body}
<div class="card"><div class="head"><b>最近 checkpoint</b></div><ul>{ck or '<li>暂无</li>'}</ul></div>
</body></html>"""


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/raw"):
            tag = self.path.split("=", 1)[-1] if "=" in self.path else TAGS[0]
            p = os.path.join(ROOT, tag + ".log")
            text = open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else "no log"
            data = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        data = page().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


class S(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


if __name__ == "__main__":
    with S(("127.0.0.1", 8791), H) as srv:
        print("serving http://127.0.0.1:8791/", flush=True)
        srv.serve_forever()
