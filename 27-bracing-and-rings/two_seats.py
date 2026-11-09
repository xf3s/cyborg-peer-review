import torch, numpy as np
src = open("f_explore.py").read(); exec(src[:src.index("for j in (45294, 121007):")])
Fs = {"#45294 de-escalation": 45294, "#93565 anger": 93565, "#17947 calm/stop": 17947, "#86782 sensitive": 86782, "#50620 ban": 50620}
docs = [d for d in docs if d[0] != "hunger"]
sid, eid = tok.convert_tokens_to_ids("<|start_header_id|>"), tok.convert_tokens_to_ids("<|eot_id|>")
R = {n: {r: [] for r in ("user", "handover", "answer", "reply")} for n in Fs}; nd = 0
for name, t, b, i in docs:
    ids = tok(t, add_special_tokens=False).input_ids
    if len(ids) != b.shape[0]: continue
    sh = [q for q, x in enumerate(ids) if x == sid]; eo = [q for q, x in enumerate(ids) if x == eid]
    if name == "contagion":
        if len(sh) < 2: continue
        ah = sh[-1]; u0 = sh[-2] + 3; e = [q for q in eo if q < ah][-1]; ans = ah + 3; n_rep = len(ids)
    else:
        ah = sh[-1]; u0 = sh[-2] + 3; e = [q for q in eo if q < ah][-1]; ans = len(ids) - 1; n_rep = None
    nd += 1
    for n, j in Fs.items():
        a = enc([j])(b, i)
        R[n]["user"] += a[u0:e].tolist(); R[n]["handover"] += a[e:ans].tolist(); R[n]["answer"] += [float(a[ans])]
        if n_rep is not None: R[n]["reply"] += a[ans + 1:n_rep].tolist()
print("docs", nd, "tokens by region:", {r: len(v) for r, v in R["#93565 anger"].items()})
print("mean activation (share of tokens firing)")
for n in Fs:
    print(f"  {n:24s} " + "  ".join(f"{r}: {np.mean(R[n][r]):.3f} ({np.mean(np.array(R[n][r]) > 0):.0%})" for r in R[n]))
j = 86782
print(f"\n==== #{j} share in Instruct {share[j]:.2f}; base/instruct decoder cos {hc[j]:+.2f}; decoder norms base {Wd2[0, j].norm():.2f}, instruct {Wd2[1, j].norm():.2f}")
print("lens:", lens(j, 18)); print("cos valence", round(float(Dn[j] @ dv), 3), "cos #45294", round(float(Dn[j] @ Dn[45294]), 3), "cos #93565", round(float(Dn[j] @ Dn[93565]), 3))
for k in (Dn @ Dn[j]).topk(7).indices.tolist()[1:]: print(f"  neighbour #{k} cos {float(Dn[k] @ Dn[j]):+.2f} {lens(k, 7)}")
AF = enc([j]); hits = []; ntok = 0; fired = 0
We = f.get_slice("encoder.weight")[:, :, j].float(); cb, ci = [], []
for name, t, b, i in docs:
    ids = tok(t, add_special_tokens=False).input_ids
    if len(ids) != b.shape[0]: continue
    a = AF(b, i).tolist(); ntok += len(a) - 1; fired += sum(v > 0.02 for v in a[1:])
    for k in range(1, len(a)):
        if a[k] > 0.02:
            hits.append((a[k], name, tok.decode(ids[max(1, k - 10):k]).replace("\n", "⏎")[-60:], tok.decode([ids[k]]).replace("\n", "⏎"), tok.decode(ids[k + 1:k + 6]).replace("\n", "⏎"))); cb.append(float(b[k].float() @ We[0])); ci.append(float(i[k].float() @ We[1]))
print(f"fires on {fired} of {ntok} tokens; encoder input when firing: base {np.mean(cb):.2f}, chat {np.mean(ci):.2f}")
for v, name, pre, t_, post in sorted(hits, reverse=True)[:14]: print(f"   {v:5.2f} [{name:9}] …{pre}⟦{t_}⟧{post}")
from collections import Counter
print(Counter(h[3].strip() for h in hits).most_common(18))
