import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
D = torch.load("capacity.pt", weights_only=True); it = D["items"]; A = D["animals"]; LAY = list(D["layers"])
K = np.array([k for k, _ in it]); Y = np.array([[a in its for a in A] for _, its in it]).astype(int); KS = sorted(set(K))
rng = np.random.default_rng(0); fold = rng.integers(0, 4, len(it))
def score(H):
    res = {k: [] for k in KS}
    for L in range(len(LAY)):
        X = H[:, L].float().numpy(); X = (X - X.mean(0)) / (X.std(0) + 1e-6); P = np.zeros_like(Y, dtype=float)
        for f in range(4):
            tr, te = fold != f, fold == f
            for a in range(len(A)):
                c = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced").fit(X[tr], Y[tr, a]); P[te, a] = c.predict(X[te])
        for k in KS:
            m = K == k; res[k].append(np.mean([balanced_accuracy_score(Y[m, a], P[m, a]) for a in range(len(A)) if 0 < Y[m, a].sum() < m.sum()]))
    return res
for name in ("answer", "listend"):
    r = score(D[name]); print(f"\n{name} token: mean balanced accuracy for 'is animal X in the list?' (chance 0.50)")
    print("   k   " + "  ".join(f"L{l:2d}" for l in LAY))
    for k in KS: print(f"  {k:2d}   " + "  ".join(f"{v:.2f}" for v in r[k]))
    j = LAY.index(16); acc = [r[k][j] for k in KS]; drops = -np.diff(acc); tot = acc[0] - acc[-1]
    print(f"  L16 steps lose: {np.round(drops, 3).tolist()} of total {tot:.3f} → largest share {drops.max() / tot if tot > 0 else 0:.0%}")
