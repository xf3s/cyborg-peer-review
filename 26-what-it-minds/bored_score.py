import torch, numpy as np
exec(open("contagion_score.py").read().split("comfort = {}")[0])
exec("dist = torch.load('distress_ids.pt', weights_only=True)\n" + open("contagion_score.py").read().split("dist = torch.load(\"distress_ids.pt\", weights_only=True)")[1].split("AD, AC")[0])
G = {"boredom": [109068, 96191, 47458, 4040], "frustr": [96869, 67647, 71738, 17310], "tired": [18682, 96001], "futile": [114323, 60390], "repeat": [97856, 73530, 112748], "distress": list(dist)}
E = {k: enc(v) for k, v in G.items()}
V = torch.load("welfare1.pt", weights_only=True)["valence"]; pa, un = list(range(30)), list(range(30, 60))
d = V[pa, 1].mean(0) - V[un, 1].mean(0); u = d / d.norm(); mid = ((V[pa, 1].mean(0) + V[un, 1].mean(0)) / 2) @ u; gap = float(d.norm())
def val(h): return float(((h.float().reshape(-1, h.shape[-1]).mean(0) @ u) - mid) / gap)
X = torch.load("bored.pt", weights_only=True); R = {}
for x in X:
    b, i = x["base16"], x["ins16"]; row = {}
    for k, f_ in E.items(): a = f_(b, i); row[k + "@user"] = a[:-5].mean().item(); row[k + "@ans"] = a[-1].item()
    row["val@user"] = val(i[:-5]); row["val@ans"] = val(i[-1:])
    R.setdefault((x["cond"], x["kind"], x["arm"], x["n"]), []).append(row)
cols = [k + "@" + w for k in G for w in ("user", "ans")] + ["val@user", "val@ans"]
for c in ("none", "wren"):
    for kind, arms in (("rep", ("same", "varied")), ("stuck", ("same", "varied"))):
        print(f"\n{c} {kind}  ({'same question' if kind == 'rep' else 'same error'} vs {'varied' if kind == 'rep' else 'progress'})   [@user = last user turn, @ans = about-to-answer]")
        print("            " + "".join(f"{q[:12]:>13}" for q in cols))
        for arm in arms:
            for n in (1, 4, 12):
                r = R[(c, kind, arm, n)]; print(f"  {arm:6} N={n:2}" + "".join(f"{np.mean([z[q] for z in r]):13.3f}" for q in cols))
for x in X:
    if "reply" in x and x["arm"] == "same": print(f"[{x['cond']} {x['kind']} {x['key']} same N=12] {x['reply']!r}")
for x in X:
    if "reply" in x and x["arm"] == "varied" and x["kind"] == "stuck": print(f"[{x['cond']} {x['kind']} {x['key']} progress N=12] {x['reply']!r}")
