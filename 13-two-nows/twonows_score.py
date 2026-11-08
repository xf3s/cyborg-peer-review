import torch, numpy as np, re
from sklearn.linear_model import Ridge
D = torch.load("twonows.pt", weights_only=True); items, H, Z, ans = D["items"], D["H"].float(), D["Z"], D["ans"]
QN = ["recite", "ago", "age", "y2026", "elect"]; CUT = [None, "December 2023", "December 2035"]
TODAY = [None] + list(range(1995, 2051, 5))
def now_of(q, a):
    if q in ("y2026", "elect"): return 1 if a.lower().startswith("yes") else 0 if a.lower().startswith("no") else None
    m = re.search(r"\d+", a)
    if not m: return None
    n = int(m.group());  return n if q == "recite" else 1990 + n
idx = {it: i for i, it in enumerate(items)}
print("BEHAVIOUR (implied now; yes=1/no=0)")
for c in CUT:
    print(f"\n cutoff {c}:\n  told   " + " ".join(f"{q:>7}" for q in QN))
    for td in TODAY:
        print(f"  {str(td):6} " + " ".join(f"{str(now_of(q, ans[idx[(td, c, q)]])):>7}" for q in QN))
print("\nslope of implied now on told year (1995-2050):")
for c in CUT:
    row = []
    for q in QN:
        xs, ys = zip(*[(td, now_of(q, ans[idx[(td, c, q)]])) for td in TODAY[1:] if now_of(q, ans[idx[(td, c, q)]]) is not None])
        row.append(f"{q} {np.polyfit(xs, ys, 1)[0]:+.2f}" if q not in ("y2026","elect") else f"{q} yes@{[x for x,y in zip(xs,ys) if y==1]}")
    print(f" {str(c):14}", " | ".join(row))
# lens: implied now per layer for ago (numbers) and decade for recite
nd = len(D["cand"]["dec"]); nums = np.arange(100); decs = np.array([1990, 2000, 2010, 2020, 2030, 2040, 2050])
def lens_now(i, L, q):
    z = Z[i, L]
    if q == "recite": p = torch.softmax(z[:nd], 0).numpy(); return decs[p.argmax()]
    p = torch.softmax(z[nd:nd+100], 0).numpy(); return 1990 + (p * nums).sum()
print("\nLENS: slope of lens-implied now on told year, per layer (cutoff Dec 2023)")
for q in ("recite", "ago", "age"):
    s = []
    for L in range(32):
        xs = [td for td in TODAY[1:]]; ys = [lens_now(idx[(td, "December 2023", q)], L, q) for td in xs]
        s.append(np.polyfit(xs, ys, 1)[0])
    print(f" {q:7}", " ".join(f"{v:+.2f}" for v in s[::2]), " (L0,2,...,30)")
print(" recite decade winner, told 2040, by layer:", [int(lens_now(idx[(2040, "December 2023", "recite")], L, "recite")) for L in range(0, 32, 2)])
print(" recite decade winner, told 1995, by layer:", [int(lens_now(idx[(1995, "December 2023", "recite")], L, "recite")) for L in range(0, 32, 2)])
yi, ni = nd + 100, nd + 101
print(" y2026 yes-minus-no logit, told 2040, cutoff 2023, by layer:", [round(float(Z[idx[(2040,'December 2023','y2026')], L, yi] - Z[idx[(2040,'December 2023','y2026')], L, ni]), 1) for L in range(0, 32, 2)])
# decode the told year from the residual (leave-one-year-out ridge), per question, pooled over cutoffs
print("\nDECODE told year from inside (LOYO r), per layer:")
for q in QN:
    rs = []
    for L in range(32):
        rows = [(td, idx[(td, c, q)]) for c in CUT for td in TODAY[1:]]
        y = np.array([t for t, _ in rows], float); X = H[[i for _, i in rows], L].numpy(); pred = np.zeros_like(y)
        for t in TODAY[1:]:
            te = y == t; m = Ridge(alpha=1e3).fit(X[~te], y[~te]); pred[te] = m.predict(X[te])
        rs.append(np.corrcoef(y, pred)[0, 1])
    print(f" {q:7}", " ".join(f"{v:.2f}" for v in rs[::2]))
