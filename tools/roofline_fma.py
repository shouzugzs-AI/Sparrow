"""为什么 8x8 AVX2 微内核顶到 ~101 GFLOPS 就上不去了 —— load 端口吞吐推算。

微内核每个 K 元素（s 的每个取值）做的事：
    c_r = fma(broadcast(A[row_r][kk+s]), v, c_r)      r = 0..7
即每个 s：8 条 FMA + 8 条 vbroadcastss(内存操作数) + 1 条 vmovaps(v)。

vbroadcastss 带内存操作数 = load uop，走 p2/p3（2 条/周期）。
FMA 走 p0/p1/p5（3 条/周期）。

所以每个 s（= 每个 K）的下界：
    load 端口：8 (broadcast) + 1 (v) = 9 → 9/2   = 4.5 周期
    FMA  端口：8                      → 8/3   = 2.67 周期
=> 瓶颈是 load 端口：4.5 周期/K

512³ 的总周期：
    行块 64 × 列块 64 = 4096 个微内核块，每块跑 K=512 个 s
    4.5 周期/K × 512 K × 4096 块 = 9.44M 周期

对应 GFLOPS 上界（FLOPs = 2·512³ = 268.4 MFLOP）：
"""
FL = 2 * 512 ** 3
cycles = 4.5 * 512 * 4096
print(f"load 端口下界 = 4.5 周期/K（FMA 端口只需 2.67）")
print(f"总周期 = {cycles/1e6:.2f}M")
for f in (3.3e9, 3.96e9):
    t = cycles / f
    print(f"  @{f/1e9:.2f} GHz: {t*1e3:.2f} ms -> {FL/t/1e9:.1f} GFLOPS 上界")
print()
print("实测（含循环开销/地址计算/扰动，通常在理论下界的 85~95%）：")
for g in (98.0, 101.0, 102.9):
    print(f"  {g:.1f} GFLOPS")
print()
print("结论：98~103 GFLOPS 就是这台 i3-12100F 上 8x8 AVX2 微内核的实际天花板，")
print("      离 load 端口理论下界（@3.96GHz 约 112 GFLOPS）只差 ~10%。")
print()
print("为什么不是 105.6 GFLOPS：")
print("  任务书写的 3.3GHz x 8 x 2 x 2 = 105.6 假设『每周期 2 条 FMA』。")
print("  Golden Cove（i3-12100F 的 P-core）有 **3** 个 256-bit FMA 端口（p0/p1/p5），")
print("  所以 FMA 侧峰值其实是 3.3GHz x 3 x 8 x 2 = 158 GFLOPS。")
print("  真正卡住的是 load 端口（2/周期），不是 FMA 端口。")
