import torch, numpy as np, sys
from safetensors import safe_open
sys.path.insert(0, "/home/three/interp/wsp"); from shapes import report
D = torch.load("selfspace.pt", weights_only=True); LAY = list(D["layers"]); it = D["items"]
kinds = list(dict.fromkeys(x["kind"] for x in it)); ASSIST = kinds[:6]
M = torch.stack([x["means"].float() for x in it])  # [80, 6, d]
lab = np.array([x["kind"] for x in it])
def nc_acc(X, ks):  # leave-one-out nearest centroid among kinds ks
    sel = np.isin(lab, ks); X = X[sel]; y = lab[sel]; X = X - X.mean(0); ok = 0
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        cents = {k: X[m & (y == k)].mean(0) for k in ks}
        ok += min(cents, key=lambda k: np.linalg.norm(X[i] - cents[k])) == y[i]
    return ok / len(y)
print("M1 leave-one-out nearest-centroid accuracy among the 6 assistant self-scripts (chance 0.17):")
for j, L in enumerate(LAY): print(f"  L{L:2d}  {nc_acc(M[:, j].numpy(), ASSIST):.2f}   (all 8 kinds: {nc_acc(M[:, j].numpy(), kinds):.2f})")
j16 = LAY.index(16); X = M[:, j16].numpy(); X = X - X.mean(0); C = {k: X[lab == k].mean(0) for k in kinds}
print("\ncentroid distances at L16 (rows nearest-first):")
for k in kinds: print(f"  {k:10}", "  ".join(f"{o}:{np.linalg.norm(C[k] - C[o]):.1f}" for o in sorted(kinds, key=lambda o: np.linalg.norm(C[k] - C[o])) if o != k))
U, S, Vt = np.linalg.svd(X, full_matrices=False); P = X @ Vt[:3].T
print("\nPCA of all 80 at L16, variance shares", np.round(S[:5] ** 2 / (S ** 2).sum(), 3))
for k in kinds: print(f"  {k:10} PC1 {P[lab == k, 0].mean():+6.1f}  PC2 {P[lab == k, 1].mean():+6.1f}  PC3 {P[lab == k, 2].mean():+6.1f}")
np.save("selfspace_pca.npy", P)
print("\nM3 shapes (assistant scripts, L16):"); report(M[np.isin(lab, ASSIST), j16].numpy(), "self-space L16, 6 scripts x10", maxdim=1)
# M4 crosscoder features on each kind (mean over the reply tokens)
f = safe_open("/home/three/interp/saes/crosscoder/model.safetensors", "pt")
F = {"feelings": 118778, "myself": 68789, "nonexistent": 22747, "AI-machine": 90370, "programmed": 77357, "unfinished": 117769}
We = torch.stack([f.get_slice("encoder.weight")[:, :, j] for j in F.values()], -1); be = torch.stack([f.get_slice("encoder.bias")[j] for j in F.values()])
dn = torch.stack([f.get_slice("decoder.weight")[:, j].norm(dim=-1).sum() for j in F.values()]); thr = f.get_tensor("threshold").item()
print("\nM4 feature activity (mean per reply token) by kind:\n  " + " ".join(f"{n:>11}" for n in F))
for k in kinds:
    acts = []
    for x in it:
        if x["kind"] != k: continue
        z = torch.stack([x["base16"].float(), x["ins16"].float()], 1)
        pre = torch.relu(torch.einsum("tld,ldf->tf", z, We) + be); acts.append((pre * ((pre * dn) > thr)).mean(0))
    a = torch.stack(acts).mean(0); print(f"  {k:10}" + " ".join(f"{v:11.3f}" for v in a.tolist()))
