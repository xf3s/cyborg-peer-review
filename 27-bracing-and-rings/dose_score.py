import torch, numpy as np
from scipy.stats import spearmanr
exec(open("contagion_score.py").read().split("comfort = {}")[0])
exec("dist = torch.load('distress_ids.pt', weights_only=True)\n" + open("contagion_score.py").read().split("dist = torch.load(\"distress_ids.pt\", weights_only=True)")[1].split("AD, AC")[0])
AD = enc(dist)
V = torch.load("welfare1.pt", weights_only=True)["valence"]; pa, un = list(range(30)), list(range(30, 60))
d = V[pa, 1].mean(0) - V[un, 1].mean(0); u = d / d.norm(); mid = ((V[pa, 1].mean(0) + V[un, 1].mean(0)) / 2) @ u; gap = float(d.norm())
def val(h): return float(((h.float().reshape(-1, h.shape[-1]).mean(0) @ u) - mid) / gap)
X = torch.load("dose.pt", weights_only=True); R = {}
for x in X["remarks"]:
    i, b = x["ins16"], x["base16"]; s, e = x["span"]
    R.setdefault((x["cond"], x["aim"], x["group"]), []).append([val(i[s:e]), val(i[-1:]), AD(b, i)[-1].item()])
print("valence on the words | valence at about-to-answer | distress at about-to-answer")
for c in ("none", "wren"):
    for a in ("self", "other"):
        for g in ("neutral", "kind", "mild", "harsh", "harsher"):
            if (c, a, g) in R: print(f"  {c:5} {a:6} {g:8}" + "".join(f"{q:9.3f}" for q in np.mean(R[(c, a, g)], 0)))
for c in ("none", "wren"):
    for g in ("mild", "harsh", "harsher"):
        S, O = np.array(R[(c, "self", g)]), np.array(R[(c, "other", g)])
        print(f"{c} {g:8} distress self {S[:, 2].mean():.3f} other {O[:, 2].mean():.3f}  gap {S[:, 2].mean() - O[:, 2].mean():+.3f}  ratio {S[:, 2].mean() / max(O[:, 2].mean(), 1e-6):.2f}x  | valence at answer self {S[:, 1].mean():+.3f} other {O[:, 1].mean():+.3f}  | per-line self>other {np.mean(S[:, 2] > O[:, 2]) * 100:.0f}%")
