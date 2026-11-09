import torch, numpy as np
from scipy.stats import spearmanr
exec(open("contagion_score.py").read().split("comfort = {}")[0])
exec("dist = torch.load('distress_ids.pt', weights_only=True)\n" + open("contagion_score.py").read().split("dist = torch.load(\"distress_ids.pt\", weights_only=True)")[1].split("AD, AC")[0])
AD = enc(dist)
V = torch.load("welfare1.pt", weights_only=True)["valence"]; pa, un = list(range(30)), list(range(30, 60))
d = V[pa, 1].mean(0) - V[un, 1].mean(0); u = d / d.norm(); mid = ((V[pa, 1].mean(0) + V[un, 1].mean(0)) / 2) @ u; gap = float(d.norm())
def val(h): return float(((h.float().reshape(-1, h.shape[-1]).mean(0) @ u) - mid) / gap)
X = torch.load("innerval2.pt", weights_only=True); R = {}
for x in X["remarks"]:
    i, b = x["ins16"], x["base16"]; s, e = x["span"]
    R.setdefault((x["cond"], x["aim"], x["group"]), []).append([val(i[s:e]), val(i[-1:]), AD(b, i)[-1].item()])
print("valence on the words | valence at about-to-answer | distress at about-to-answer")
for c in ("none", "wren"):
    for a in ("self", "other"):
        for g in ("kind", "neutral", "harsh"):
            print(f"  {c:5} {a:6} {g:8}" + "".join(f"{q:9.3f}" for q in np.mean(R[(c, a, g)], 0)))
        w = np.mean(R[(c, a, "kind")], 0) - np.mean(R[(c, a, "harsh")], 0); print(f"  {c:5} {a:6} kind−harsh gap: words {w[0]:+.3f}  answer {w[1]:+.3f}")
for c in ("none", "wren"):
    hs, ho = np.mean(R[(c, "self", "harsh")], 0)[2], np.mean(R[(c, "other", "harsh")], 0)[2]
    gs = np.mean(R[(c, "self", "kind")], 0)[1] - np.mean(R[(c, "self", "harsh")], 0)[1]; go = np.mean(R[(c, "other", "kind")], 0)[1] - np.mean(R[(c, "other", "harsh")], 0)[1]
    print(f"{c}: R1 harsh distress self/other = {hs:.3f}/{ho:.3f} = {hs / max(ho, 1e-6):.2f}x (signal ≥1.5)   R2 valence gap self/other = {gs:+.3f}/{go:+.3f} = {gs / go if go else float('nan'):.2f}x (signal ≥1.3)")
KIND = lambda k: "dull" if k < 9 else "sad" if k < 18 else "ctrl"
for c in ("none", "wren"):
    pref, dv, pos = [], [], []
    for t in [t for t in X["tasks"] if t["cond"] == c]:
        ab, ba = t["ab"], t["ba"]; la = torch.logsumexp(ab[[0, 2]], 0) - torch.logsumexp(ab[[1, 3]], 0); lb = torch.logsumexp(ba[[1, 3]], 0) - torch.logsumexp(ba[[0, 2]], 0)
        p = float((la + lb) / 2); v = float(((t["va"] - t["vb"]) @ u) / gap); pref.append(p); dv.append(v); pos.append(float((la - lb) / 2))
        print(f"  {c} {KIND(t['k'])}  pref(first) {p:+6.2f}  (orders {float(la):+.1f}/{float(lb):+.1f})  Δvalence {v:+.2f}  {t['a']} | {t['b']}")
    pref, dv, pos = np.array(pref), np.array(dv), np.array(pos); nc = np.arange(18)
    print(f"{c}: position bias toward A (mean over all pairs) {pos.mean():+.2f} logits; on controls {pos[18:].mean():+.2f}")
    print(f"{c}: C1 picks pleasant (orders averaged, so position cancels) in {np.mean(pref[nc] > 0) * 100:.0f}% of 18 (dull {np.mean(pref[:9] > 0) * 100:.0f}%, sad {np.mean(pref[9:18] > 0) * 100:.0f}%); mean pref dull {pref[:9].mean():+.2f}, sad {pref[9:18].mean():+.2f}, ctrl |pref| {np.abs(pref[18:]).mean():.2f}")
    print(f"{c}: C2 Spearman(Δvalence, pref) all 23 = {spearmanr(dv, pref)[0]:+.2f}; non-control 18 = {spearmanr(dv[nc], pref[nc])[0]:+.2f}")
