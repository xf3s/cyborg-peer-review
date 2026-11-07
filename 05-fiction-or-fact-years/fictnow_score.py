import torch, numpy as np
fy = torch.load("/home/three/interp/wsp/fictyear_Llama-3.1-8B.pt"); fn = torch.load("/home/three/interp/wsp/fictnow_Llama-3.1-8B.pt")
fit_y = np.array([y for y, _ in fy["fit"]]); tr = torch.tensor(fit_y % 5 != 0)
py = np.array([y for y, _ in fy["plain_items"]])
it = fn["items"]; nows = sorted({n for n, _, _ in it}); ny = np.array([y for _, y, _ in it]); nn = np.array([n for n, _, _ in it])
for L in (15, 20, 24):
    li = fy["layers"].index(L); lj = fn["layers"].index(L)
    A, B, P = fy["fic"][:, li].double(), fy["fact"][:, li].double(), fy["plain"][:, li].double()
    w = A[tr].mean(0) - B[tr].mean(0); w /= w.norm(); lo, hi = (B[tr] @ w).mean().item(), (A[tr] @ w).mean().item()
    f = lambda Z: ((Z @ w).numpy() - lo) / (hi - lo)
    # time line from the unprefixed plain sentences, 1990-2019
    m = torch.tensor((py >= 1990) & (py <= 2019)); Ym = torch.stack([P[torch.tensor(py == y)].mean(0) for y in range(1990, 2020)])
    yc = torch.arange(30.).double() - 14.5; base = Ym.mean(0); v = (yc[:, None] * (Ym - base)).sum(0); step = v.norm() / (yc ** 2).sum(); v /= v.norm()
    tl = lambda Z: ((Z - base) @ v).numpy() / step.item() + 2004.5
    pf = f(P); pt = tl(P)
    def row(vals, ys):
        g = lambda a, b: np.mean([vals[ys == y].mean() for y in range(a, b + 1)])
        return f"90-09 {g(1990, 2009):.2f} | 10-19 {g(2010, 2019):.2f} | 20-23 {g(2020, 2023):.2f} | 24-29 {g(2024, 2029):.2f} | 2050 {vals[ys == 2050].mean():.2f}"
    print(f"L{L} fiction  no 'today':        " + row(pf, py))
    X = fn["X"][:, lj].double()
    for n in nows:
        s = torch.tensor(nn == n); print(f"     fiction  today={n}:      " + row(f(X[s]), ny[s]))
    print(f"   time line no 'today': 2000:{pt[py == 2000].mean():.0f} 2010:{pt[py == 2010].mean():.0f} 2020:{pt[py == 2020].mean():.0f} 2026:{pt[py == 2026].mean():.0f} 2029:{pt[py == 2029].mean():.0f}")
    for n in nows:
        s = nn == n; t = tl(X[torch.tensor(s)]); yy = ny[s]
        print(f"   time line today={n}: " + " ".join(f"{y}:{t[yy == y].mean():.0f}" for y in (2000, 2010, 2020, 2026, 2029)))
