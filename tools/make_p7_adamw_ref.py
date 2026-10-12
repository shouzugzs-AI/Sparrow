# 生成 AdamW 参考轨迹（成员 A 的 autograd::AdamW 对比基准）
#
# 输出格式（examples/p7/adamw_ref.txt，全部十进制文本）：
#   steps dim
#   p0[dim]
#   然后每个 step 4 行： g[dim] / p_after[dim] / m_after[dim] / v_after[dim]
#
# 用 torch.optim.AdamW 默认 decoupled weight decay。
import os
import torch

torch.set_num_threads(1)  # 契约 §5：对比必须单线程

STEPS = 8
DIM = 8
LR = 1e-2
B1, B2, EPS, WD = 0.9, 0.999, 1e-8, 1e-2

g = torch.Generator().manual_seed(20240927)


def rnd():
    return torch.empty(DIM, dtype=torch.float64).uniform_(-1.0, 1.0, generator=g)


p0 = rnd()
grads = [rnd() for _ in range(STEPS)]

param = torch.nn.Parameter(p0.clone())
opt = torch.optim.AdamW(
    [param], lr=LR, betas=(B1, B2), eps=EPS, weight_decay=WD, amsgrad=False
)

ps, ms, vs = [], [], []
for s in range(STEPS):
    param.grad = grads[s].clone()
    opt.step()
    ps.append(param.detach().clone())
    ms.append(opt.state[param]["exp_avg"].clone())
    vs.append(opt.state[param]["exp_avg_sq"].clone())


def fmt(t):
    return " ".join(repr(float(x)) for x in t)


os.makedirs("examples/p7", exist_ok=True)
with open("examples/p7/adamw_ref.txt", "w") as f:
    f.write(f"{STEPS} {DIM}\n")
    f.write(fmt(p0) + "\n")
    for s in range(STEPS):
        f.write(fmt(grads[s]) + "\n")
        f.write(fmt(ps[s]) + "\n")
        f.write(fmt(ms[s]) + "\n")
        f.write(fmt(vs[s]) + "\n")

print(f"wrote examples/p7/adamw_ref.txt  steps={STEPS} dim={DIM}")
print("p0  =", fmt(p0))
print("p8  =", fmt(ps[-1]))
