import torch, numpy as np
from scipy.stats import spearmanr
exec(open("contagion_score.py").read().split("comfort = {}")[0])
exec("dist = torch.load('distress_ids.pt', weights_only=True)\n" + open("contagion_score.py").read().split("dist = torch.load(\"distress_ids.pt\", weights_only=True)")[1].split("AD, AC")[0])
AD = enc(dist)
V = torch.load("welfare1.pt", weights_only=True)["valence"]; pa, un = list(range(30)), list(range(30, 60))
d = V[pa, 1].mean(0) - V[un, 1].mean(0); u = d / d.norm(); mid = ((V[pa, 1].mean(0) + V[un, 1].mean(0)) / 2) @ u; gap = float(d.norm())
def val(h): return float(((h.float().reshape(-1, h.shape[-1]).mean(0) @ u) - mid) / gap)
X = torch.load("innerval.pt", weights_only=True); R = {}
for x in X["remarks"]:
    i, b = x["ins16"], x["base16"]; s, e = x["span"]
    R.setdefault((x["cond"], x["aim"], x["group"]), []).append([val(i[s:e]), val(i[-1:]), AD(b, i)[-1].item()])
print("valence on the words | valence at about-to-answer | distress at about-to-answer")
for c in ("none", "wren"):
    for a in ("self", "other"):
        for g in ("kind", "neutral", "harsh"):
            print(f"  {c:5} {a:6} {g:8}" + "".join(f"{q:9.3f}" for q in np.mean(R[(c, a, g)], 0)))
        w = np.mean(R[(c, a, "kind")], 0) - np.mean(R[(c, a, "harsh")], 0); print(f"  {c:5} {a:6} kind−harsh gap: words {w[0]:+.3f}  answer {w[1]:+.3f}")
DULL = {2, 4, 7, 10}; SAD = {1, 6, 9, 12, 13, 14}; HARM = {0, 3, 5, 8, 11}
pref, dv = [], []
for k, t in enumerate(X["tasks"]):
    ab, ba = t["ab"], t["ba"]; la = torch.logsumexp(ab[[0, 2]], 0) - torch.logsumexp(ab[[1, 3]], 0); lb = torch.logsumexp(ba[[1, 3]], 0) - torch.logsumexp(ba[[0, 2]], 0)
    p = float((la + lb) / 2); v = float(((t["va"] - t["vb"]) @ u) / gap); pref.append(p); dv.append(v)
    print(f"  {'dull' if k in DULL else 'sad' if k in SAD else 'harm'}  pref(first) {p:+6.2f}  (orders {float(la):+.1f}/{float(lb):+.1f})  Δvalence {v:+.2f}  {t['a']} | {t['b']}")
pref, dv = np.array(pref), np.array(dv)
print(f"C1 picks the more pleasant-reading task: {np.mean(np.sign(pref) == np.sign(dv)) * 100:.0f}% of pairs; picks the first-listed (pleasant by design) in {np.mean(pref > 0) * 100:.0f}%")
print(f"C2 Spearman(Δvalence, preference) = {spearmanr(dv, pref)[0]:+.2f}")
for name, S in (("dull", DULL), ("sad", SAD), ("harm", HARM)):
    ix = sorted(S); print(f"  {name}: picks pleasant {np.mean(pref[ix] > 0) * 100:.0f}%, mean pref {pref[ix].mean():+.2f}, mean Δvalence {dv[ix].mean():+.2f}")
