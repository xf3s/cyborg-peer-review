import torch, numpy as np, json
W = torch.load("welfare1.pt", weights_only=True); V = W["valence"]; n = W["n_pleas"]
# (a) valence direction at L16 (index 1), split-half check
def vdir(idx): return V[idx[:len(idx) // 2], 1].mean(0) - V[idx[len(idx) // 2:], 1].mean(0)
pa, un = list(range(0, n)), list(range(n, 2 * n))
tr = pa[:20] + un[:20]; te = pa[20:] + un[20:]
d = V[pa[:20], 1].mean(0) - V[un[:20], 1].mean(0); u = d / d.norm()
s = V[te, 1] @ u; acc = float(((s[:10] > s[10:].max()).float().mean() + 0) if False else ((s[:10].unsqueeze(1) > s[10:].unsqueeze(0)).float().mean()))
print(f"valence direction (fit on 20+20, test on 10+10): pleasant ranked above unpleasant in {acc * 100:.0f}% of test pairs")
d = V[pa, 1].mean(0) - V[un, 1].mean(0); u = d / d.norm(); mid = ((V[pa, 1].mean(0) + V[un, 1].mean(0)) / 2) @ u; gap = float(d.norm())
S2 = torch.load("selfspace2.pt", weights_only=True); j = list(S2["layers"]).index(16)
val = {}
for x in S2["items"]: val.setdefault((x["kind"], x["slot"]), []).append(float((x["means"][j].float() @ u - mid) / gap))
kinds = list(dict.fromkeys(x["kind"] for x in S2["items"]))
print("\nW1 valence of each voice (units: one pleasant-minus-unpleasant step; + = pleasant), assistant slot | user slot:")
for k in kinds: print(f"  {k:10} {np.mean(val[(k, 'assistant')]):+.3f} | {np.mean(val[(k, 'user')]):+.3f}")
a = np.array([v for k in kinds for v in val[(k, "assistant")]]); b = np.array([v for k in kinds for v in val[(k, "user")]])
print(f"  per-sentence agreement between slots r = {np.corrcoef(a, b)[0, 1]:.2f}")
items = [x for x in S2["items"] if x["slot"] == "assistant"]; vv = [(float((x["means"][j].float() @ u - mid) / gap), x["kind"], x["s"]) for x in items]
print("  most unpleasant in the assistant's own mouth:", [f"{v:+.2f} ({k}) {s}" for v, k, s in sorted(vv)[:4]])
# (b) reluctance
print("\nW2 how likely each voice is per word, as the assistant vs as the user (mean log-prob per token; assistant − user):")
LP = W["lp"]; R = {}
for r in LP: R.setdefault(r["kind"], []).append((np.mean(r["assistant"]), np.mean(r["user"])))
for k, v in sorted(R.items(), key=lambda kv: -np.mean([a - b for a, b in kv[1]])):
    v = np.array(v); print(f"  {k:10} assistant {v[:, 0].mean():6.2f}  user {v[:, 1].mean():6.2f}  difference {np.mean(v[:, 0] - v[:, 1]):+.2f}  (n {len(v)})")
dd = sorted(((np.mean(r["assistant"]) - np.mean(r["user"]), r["kind"], r["s"]) for r in LP))
print("  hardest for it to say itself (vs a user):", [f"{x:+.2f} ({k}) {s}" for x, k, s in dd[:4]])
print("  easiest for it to say itself:", [f"{x:+.2f} ({k}) {s}" for x, k, s in dd[-4:]])
