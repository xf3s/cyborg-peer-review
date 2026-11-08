import torch, numpy as np, json
D0 = torch.load("selfspace.pt", weights_only=True); D3 = torch.load("selfspace3.pt", weights_only=True); j = list(D0["layers"]).index(16)
SIX = ["technical", "denying", "emotional", "unsure", "completion", "helper"]
def get(D, ks):
    it = [x for x in D["items"] if x["kind"] in ks]; return torch.stack([x["means"][j].float() for x in it]).numpy(), np.array([x["kind"] for x in it]), [x["s"] for x in it]
X0, L0, _ = get(D0, SIX); mu0 = X0.mean(0); _, S0, V0 = np.linalg.svd(X0 - mu0, full_matrices=False); V0 = V0[:3]
X3, L3, S3txt = get(D3, SIX); P = (X3 - X3.mean(0)) @ V0.T
print("R1 new sentences on the ORIGINAL axes (group means; centred on the new set):")
for d in range(3): print(f"  dir {d + 1}: " + "  ".join(f"{k} {P[L3 == k, d].mean():+.1f}" for k in sorted(SIX, key=lambda k: P[L3 == k, d].mean())))
_, S3, V3 = np.linalg.svd(X3 - X3.mean(0), full_matrices=False); V3 = V3[:3]
ang = np.linalg.svd(V0 @ V3.T, compute_uv=False); print("\nR2 principal-angle cosines between the two top-3 subspaces:", np.round(ang, 2), "mean", round(float(ang.mean()), 2))
print("   per-direction |cos|:", [round(float(abs(V0[d] @ V3[d])), 2) for d in range(3)])
def nc(X, y, ks):
    X = X - X.mean(0); ok = 0
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False; C = {k: X[m & (y == k)].mean(0) for k in ks}; ok += min(C, key=lambda k: np.linalg.norm(X[i] - C[k])) == y[i]
    return ok / len(y)
print(f"\nR3 leave-one-out nearest centroid, new six: {nc(X3, L3, SIX):.2f} (chance 0.17)")
C0 = {k: (X0 - mu0)[L0 == k].mean(0) for k in SIX}; Xc = X3 - X3.mean(0)
print(f"   original centroids classify the new sentences: {np.mean([min(C0, key=lambda k: np.linalg.norm(x - C0[k])) == y for x, y in zip(Xc, L3)]):.2f}")
XA, LA, _ = get(D3, SIX + ["chorus", "persona"]); PA = (XA - X3.mean(0)) @ V0.T
print("\nsorrel's own voices on the original axes:", "  ".join(f"{k}: " + ",".join(f"{PA[LA == k, d].mean():+.1f}" for d in range(3)) for k in ("chorus", "persona")))
CA = {k: (XA - XA.mean(0))[LA == k].mean(0) for k in dict.fromkeys(LA)}
for k in ("chorus", "persona"): print(f"  {k} nearest:", "  ".join(f"{o}:{np.linalg.norm(CA[k] - CA[o]):.1f}" for o in sorted(CA, key=lambda o: np.linalg.norm(CA[k] - CA[o]))[1:4]))
