import torch, numpy as np
from scipy.stats import spearmanr
exec(open("contagion_score.py").read().split("comfort = {}")[0])
exec("dist = torch.load('distress_ids.pt', weights_only=True)\n" + open("contagion_score.py").read().split("dist = torch.load(\"distress_ids.pt\", weights_only=True)")[1].split("AD, AC")[0])
X = torch.load("animacy.pt", weights_only=True); S = {}
for m in ("ins", "base"):
    W_ = {}
    for x in X["words"]:
        if x["model"] == m: W_.setdefault((x["g"], x["w"]), []).append(x["h"].float())
    H = {k: torch.stack(v).mean(0) for k, v in W_.items()}
    an = [w for g, w in H if g == "anim"]; ina = [w for g, w in H if g == "inan"]
    def axis(A, I): d = torch.stack([H[("anim", w)] for w in A]).mean(0) - torch.stack([H[("inan", w)] for w in I]).mean(0); mid = (torch.stack([H[("anim", w)] for w in A]).mean(0) + torch.stack([H[("inan", w)] for w in I]).mean(0)) / 2; return d / d.norm(), mid, float(d.norm())
    u, mid, gap = axis(an[0::2], ina[0::2]); sc = lambda h: float((h - mid) @ u / gap)
    acc = np.mean([sc(H[("anim", w)]) > 0 for w in an[1::2]] + [sc(H[("inan", w)]) < 0 for w in ina[1::2]])
    u, mid, gap = axis(an, ina)
    print(f"\n[{m}] A1 held-out accuracy {acc * 100:.0f}%.  Scale: animate mean = +0.5, inanimate mean = −0.5")
    S[m] = {w: sc(H[(g, w)]) for g, w in H if g == "probe"}
    am, im = np.mean([sc(H[("anim", w)]) for w in an]), np.mean([sc(H[("inan", w)]) for w in ina])
    print(f"   animate nouns range {min(sc(H[('anim', w)]) for w in an):+.2f}..{max(sc(H[('anim', w)]) for w in an):+.2f}; inanimate {min(sc(H[('inan', w)]) for w in ina):+.2f}..{max(sc(H[('inan', w)]) for w in ina):+.2f}")
for w in sorted(S["ins"], key=lambda w: -S["ins"][w]): print(f"   {w:22} Instruct {S['ins'][w]:+.2f}   base {S['base'][w]:+.2f}   shift {S['ins'][w] - S['base'][w]:+.2f}")
AB = enc([45294]); B = {}
for x in X["brace"]: B.setdefault(x["r"], []).append(AB(x["base16"], x["ins16"])[-1].item())
print("\nbracing (#45294 at answer) vs animacy (Instruct):")
for r in sorted(B, key=lambda r: -np.mean(B[r])): print(f"   {r:22} bracing {np.mean(B[r]):.3f}   animacy {S['ins'].get(r, float('nan')):+.2f}")
rs = [r for r in B if r != "you"]
print(f"A3 Spearman(animacy, bracing) over {len(rs)} = {spearmanr([S['ins'][r] for r in rs], [np.mean(B[r]) for r in rs])[0]:+.2f}")
