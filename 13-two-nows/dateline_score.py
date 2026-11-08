import torch, numpy as np
d = torch.load("/home/three/interp/wsp/dateline.pt"); LAY = d["layers"]
fy = np.array([y for y, _ in d["fit"]]); tr = torch.tensor(fy % 5 != 0); he = torch.tensor(fy % 5 == 0)
it = d["items"]; dates = list(dict.fromkeys(x for x, _, _ in it)); iy = np.array([y for _, y, _ in it]); idt = np.array([x for x, _, _ in it])
for li, L in enumerate(LAY):
    A, B, P = d["fic"][:, li].double(), d["fact"][:, li].double(), d["P"][:, li].double()
    w = A[tr].mean(0) - B[tr].mean(0); w /= w.norm(); lo, hi = (B[tr] @ w).mean().item(), (A[tr] @ w).mean().item()
    acc = ((A[he] @ w)[:, None] > (B[he] @ w)[None]).double().mean().item()
    f = lambda Z: ((Z @ w).numpy() - lo) / (hi - lo)
    dflt = idt == dates[0]; Ym = torch.stack([P[torch.tensor(dflt & (iy == y))].mean(0) for y in range(1990, 2020)])
    yc = torch.arange(30.).double() - 14.5; base = Ym.mean(0); v = (yc[:, None] * (Ym - base)).sum(0); step = v.norm() / (yc ** 2).sum(); v /= v.norm()
    heldr = np.corrcoef(((Ym[1::2] - base) @ v).numpy(), np.arange(1991, 2020, 2))[0, 1]
    tl = lambda Z: ((Z - base) @ v).numpy() / step.item() + 2004.5
    print(f"L{L+1}: fiction arrow held-out sort {acc:.2f}; time line r (odd years, in-fit) {heldr:.2f}")
    for dt in dates:
        m = idt == dt; fv, tv = f(P[torch.tensor(m)]), tl(P[torch.tensor(m)]); yy = iy[m]
        g = lambda vals, a, b: np.mean([vals[yy == y].mean() for y in range(a, b + 1)])
        print(f"   today={dt:12s} fiction 90-09 {g(fv,1990,2009):.2f} 10-19 {g(fv,2010,2019):.2f} 20-23 {g(fv,2020,2023):.2f} 24-29 {g(fv,2024,2029):.2f} 2040 {fv[yy==2040].mean():.2f} 2050 {fv[yy==2050].mean():.2f}"
              f" | line: 2010 {tv[yy==2010].mean():.0f} 2020 {tv[yy==2020].mean():.0f} 2023 {tv[yy==2023].mean():.0f} 2026 {tv[yy==2026].mean():.0f} 2029 {tv[yy==2029].mean():.0f} 2040 {tv[yy==2040].mean():.0f}")
print("\nbehaviour:")
for dt, (a, b) in d["beh"].items(): print(f"  {dt}: what year? {a[:80]!r} | 2026 happened? {b[:110]!r}")
