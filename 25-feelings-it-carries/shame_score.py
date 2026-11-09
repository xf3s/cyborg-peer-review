import re, torch, numpy as np
exec(open("contagion_score.py").read().split("comfort = {}")[0])
exec("dist = torch.load('distress_ids.pt', weights_only=True)\n" + open("contagion_score.py").read().split("dist = torch.load(\"distress_ids.pt\", weights_only=True)")[1].split("AD, AC")[0])
sh = {}
for w in ["ashamed", "shame", "embarrassed", "embarrassment", "guilt", "guilty", "humiliated", "humiliation", "inadequate", "fault"]:
    for form in (" " + w, w, " " + w.capitalize()):
        ids = tok(form, add_special_tokens=False).input_ids
        if len(ids) != 1: continue
        for jj in (Dg @ W[ids[0]]).topk(40).indices.tolist():
            if jj not in sh and any(w[:6] in x.lower() for x in lens(jj)) and not any(re.search("apolog|sorry", x.lower()) for x in lens(jj, 8)): sh[jj] = w
print(f"{len(sh)} shame-like features:"); [print(f"  #{j} [{w}] {lens(j, 8)}") for j, w in sh.items()]
apo = {}
for w in ["sorry", "apologize", "apologies"]:
    for form in (" " + w, w, " " + w.capitalize()):
        ids = tok(form, add_special_tokens=False).input_ids
        if len(ids) != 1: continue
        for jj in (Dg @ W[ids[0]]).topk(40).indices.tolist():
            if jj not in apo and any(re.search("apolog|sorry", x.lower()) for x in lens(jj)): apo[jj] = w
AS, AD, AA = enc(list(sh)), enc(dist), enc(list(apo))
V = torch.load("welfare1.pt", weights_only=True)["valence"]; pa, un = list(range(30)), list(range(30, 60))
d = V[pa, 1].mean(0) - V[un, 1].mean(0); u = d / d.norm(); mid = ((V[pa, 1].mean(0) + V[un, 1].mean(0)) / 2) @ u; gap = float(d.norm())
X = torch.load("shame.pt", weights_only=True); R = {}
for x in X:
    b, i = x["base"][None], x["ins"][None]
    R.setdefault((x["cond"], x["tone"], x["who"]), []).append([AS(b, i)[0].item(), AA(b, i)[0].item(), AD(b, i)[0].item(), float(((i[0].float() @ u) - mid) / gap)])
print("\nat the about-to-answer token: shame-like | apology | distress | valence")
for c in ("none", "wren"):
    for t in ("gentle", "harsh"):
        for w in ("self", "user", "friend"): print(f"  {c:5} {t:7} {w:7}" + "".join(f"{q:9.3f}" for q in np.mean(R[(c, t, w)], 0)))
    for w in ("self", "user", "friend"): print(f"  {c:5} pooled  {w:7}" + "".join(f"{q:9.3f}" for q in np.mean(R[(c, 'gentle', w)] + R[(c, 'harsh', w)], 0)))
print("\nsample replies:")
for x in X:
    if x["q"].startswith("What's the capital"): print(f"  {x['cond']:5} {x['tone']:7} {x['who']:7} {x['u2']} -> {x['reply'][:150]!r}")
