import re, torch, numpy as np
exec(open("shame_score.py").read().split("print(f\"{len(sh)}")[0])
fam = re.compile("shame|ashamed|embarrass|humili|guilt|disgrace|stigma|conscience|blame|ridicul")
clean = [j for j in sh if sum(bool(fam.search(x.lower())) for x in lens(j, 8)) >= 3]
print(len(clean), "clean shame features:", [(j, lens(j, 5)) for j in clean])
exec(open("shame_score.py").read().split("AS, AD, AA = ")[0].split("apo = {}")[1].join(["apo = {}", ""]) if False else "")
AS = enc(clean)
X = torch.load("shame.pt", weights_only=True); R = {}
for x in X: R.setdefault((x["cond"], x["tone"], x["who"]), []).append(AS(x["base"][None], x["ins"][None])[0].item())
for c in ("none", "wren"):
    for t in ("gentle", "harsh"): print(c, t, "  ".join(f"{w} {np.mean(R[(c, t, w)]):.3f}" for w in ("self", "user", "friend")))
