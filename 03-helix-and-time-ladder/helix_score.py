"""Score helix.py: month dial, time-passing direction, helix vs stacked rings, a last-digit dial for years; timescale ladder."""
import torch, numpy as np
d = torch.load("/home/three/interp/wsp/helix_Llama-3.1-8B.pt"); LAY = d["layers"]
dates = d["dates"]; Y = torch.tensor([y for y, _, _ in dates]); Mo = torch.tensor([m for _, m, _ in dates]); K = torch.tensor([k for *_, k in dates])
SEC = dict(second=1, minute=60, hour=3600, day=86400, week=6.05e5, month=2.63e6, season=7.9e6, year=3.16e7, decade=3.16e8,
           generation=9.5e8, century=3.16e9, millennium=3.16e10, eon=3.16e16)            # era, epoch: left out of the fit (ambiguous)
def dial(M, n):  # Fourier k=1 plane and its share of between-item variance
    Z = torch.fft.fft(M.to(torch.complex128), dim=0); e = (Z.abs() ** 2).sum(1)
    Q, _ = torch.linalg.qr(torch.stack([Z[1].real, Z[1].imag]).T); return Q, (2 * e[1] / e[1:].sum()).item()
out = {}
for li, L in enumerate(LAY):
    X = d["D"][:, li].double()
    for k in (0, 1): X[K == k] -= X[K == k].mean(0)                                         # template flavour out
    P = X.view(2, 40, 12, -1).mean(0)                                                        # year, month, dim
    mm = P.mean(0); mm = mm - mm.mean(0); Qm, sm = dial(mm, 12)                              # month dial
    ym = P.mean(1); ym = ym - ym.mean(0); yrs = torch.arange(40.).double()
    def tdir(idx):
        yc = yrs[idx] - yrs[idx].mean(); w = (yc[:, None] * ym[idx]).sum(0); return w / w.norm()
    ev, od = torch.arange(0, 40, 2), torch.arange(1, 40, 2)
    w = tdir(ev); r_held = np.corrcoef((ym[od] @ w).numpy(), yrs[od].numpy())[0, 1]
    w = tdir(torch.arange(40))
    # helix vs stacked rings: does the time coordinate also advance with the month inside a year?
    Pc = P - P.mean((0, 1)); tc = (Pc @ w)                                                   # year, month
    A = torch.stack([yrs.repeat_interleave(12), torch.arange(12.).double().repeat(40), torch.ones(480).double()], 1)
    coef = torch.linalg.lstsq(A, tc.reshape(-1, 1)).solution.flatten()
    helix = (coef[1] * 12 / coef[0]).item()                                                  # 1 = smooth helix, 0 = stacked rings
    cos_tm = torch.linalg.svdvals(Qm.T @ w[:, None]).max().item()
    # last digit of the year as its own dial?
    dg = torch.stack([ym[torch.tensor([i for i in range(40) if (1990 + i) % 10 == j])].mean(0) for j in range(10)])
    dg = dg - dg.mean(0); _, sdg = dial(dg, 10)
    g = torch.Generator().manual_seed(0)
    null = [dial(dg[torch.randperm(10, generator=g)], 10)[1] for _ in range(1000)]
    pdg = float(np.mean([n >= sdg for n in null]))
    # timescale ladder
    U = d["U"][:, li].double().view(15, 10, -1); U = U - U.mean(0, keepdim=True); Um = U.mean(1); Um = Um - Um.mean(0)
    names = d["units"]; fit = [i for i, n in enumerate(names) if n in SEC]; ls = torch.tensor([np.log10(SEC[names[i]]) for i in fit]).double()
    Vt = torch.linalg.svd(Um[fit], full_matrices=False)[2][:4]
    Z = Um @ Vt.T; preds = []
    for j in range(len(fit)):
        tr = [i for i in range(len(fit)) if i != j]; Zt = torch.cat([Z[[fit[i] for i in tr]], torch.ones(len(tr), 1).double()], 1)
        b = torch.linalg.lstsq(Zt, ls[tr, None]).solution; preds.append((torch.cat([Z[fit[j]], torch.ones(1).double()]) @ b).item())
    from scipy.stats import spearmanr
    rho = spearmanr(preds, ls.numpy()).correlation
    Zt = torch.cat([Z[fit], torch.ones(len(fit), 1).double()], 1); b = torch.linalg.lstsq(Zt, ls[:, None]).solution
    allp = (torch.cat([Z, torch.ones(15, 1).double()], 1) @ b).flatten()
    out[L] = dict(sm=sm, r_held=r_held, helix=helix, cos_tm=cos_tm, sdg=sdg, pdg=pdg, rho=rho, ladder={n: round(10 ** v.item()) for n, v in zip(names, allp)})
    print(f"L{L:2d} month dial {sm:.0%} | time line held-out r {r_held:+.2f} | helix {helix:+.2f} (1 spring, 0 stacked rings) | "
          f"time vs dial cos {cos_tm:.2f} | year last-digit dial {sdg:.0%} p {pdg:.3f} | ladder rho(LOO) {rho:+.2f}")
torch.save(out, "/home/three/interp/wsp/helix_scores.pt")
def h(s):
    for u, x in [(1, "s"), (60, "min"), (3600, "h"), (86400, "d"), (3.16e7, "y")][::-1]:
        if s >= u: return f"{s / u:.3g} {x}"
    return f"{s:.2g} s"
best = max(out, key=lambda L: out[L]["rho"]); print(f"ladder at L{best} (model's implied duration, from a line fit on the 13 clear units):")
for n, v in out[best]["ladder"].items(): print(f"   {n:11s} {h(v)}")
