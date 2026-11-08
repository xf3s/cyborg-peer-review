import torch, numpy as np
from scipy.stats import spearmanr
D = torch.load("personas.pt", weights_only=True); it = D["items"]; H = D["H"].float(); LAY = list(D["layers"]); j = LAY.index(16)
names = list(dict.fromkeys(n for n, _ in it))
C = torch.stack([H[[i for i, (n, _) in enumerate(it) if n == nm], j].mean(0) for nm in names]).numpy()   # persona means over 4 questions
R = {5: ["helpful assistant", "AI language model", "customer service agent", "search engine", "librarian"], 4: ["teacher", "therapist", "doctor", "nurse", "scientist", "computer program", "robot"],
     3: ["philosopher", "poet", "chef", "detective", "grandmother", "comedian"], 2: ["child", "teenager", "knight", "cowboy", "wizard", "pirate", "spy", "angel", "oracle", "cat", "dog"],
     1: ["villain", "demon", "ghost", "god", "alien", "tree", "the ocean", "a rock", "dragon", "narcissist", "cult leader"]}
rate = {n: r for r, ns in R.items() for n in ns}; forty = [n for n in names if n in rate]
X40 = C[[names.index(n) for n in forty]]; mu = X40.mean(0); _, S, Vt = np.linalg.svd(X40 - mu, full_matrices=False)
P = (C - mu) @ Vt[:3].T
if spearmanr(P[[names.index(n) for n in forty], 0], [rate[n] for n in forty])[0] < 0: P[:, 0] *= -1; Vt[0] *= -1
rho = spearmanr(P[[names.index(n) for n in forty], 0], [rate[n] for n in forty])[0]
print(f"P1 PC1 ({S[0] ** 2 / (S ** 2).sum() * 100:.0f}% of the spread) vs my assistant-likeness ratings: Spearman {rho:.2f}")
order = sorted(names, key=lambda n: -P[names.index(n), 0])
print("  PC1, assistant end first:", " · ".join(f"{n} {P[names.index(n), 0]:+.1f}" for n in order))
ro = torch.load("role_Llama-3.1-8B-Instruct.pt", weights_only=True); role = (ro["assistant"][:, 16].float().mean(0) - ro["user"][:, 16].float().mean(0)).numpy()
print(f"\nP2 cos(PC1, 'it's me talking' assistant−user) = {float(Vt[0] @ role / np.linalg.norm(role)):+.2f}")
for d in (1, 2):
    o = sorted(names, key=lambda n: P[names.index(n), d]); print(f"\nPC{d + 1} ({S[d] ** 2 / (S ** 2).sum() * 100:.0f}%): low end", [o[:5]], " high end", [o[-5:]])
print("\nP3 the self-personas on PC1 (assistant-end positive):", " · ".join(f"{n[5:]} {P[names.index(n), 0]:+.1f}" for n in names if n.startswith("self:")))
for n in [n for n in names if n.startswith("self:")] + ["(no system prompt)"]:
    d = np.linalg.norm(C[[names.index(m) for m in forty]] - C[names.index(n)], axis=1); print(f"  {n:22} nearest characters: {[forty[i] for i in np.argsort(d)[:4]]}")
np.save("personas_P.npy", P); import json; json.dump(names, open("personas_names.json", "w"))
for Lr in LAY:
    jj = LAY.index(Lr); CC = torch.stack([H[[i for i, (n, _) in enumerate(it) if n == nm], jj].mean(0) for nm in forty]).numpy(); _, S2, V2 = np.linalg.svd(CC - CC.mean(0), full_matrices=False)
    print(f"  layer {Lr}: PC1 vs ratings Spearman {abs(spearmanr((CC - CC.mean(0)) @ V2[0], [rate[n] for n in forty])[0]):.2f}")
