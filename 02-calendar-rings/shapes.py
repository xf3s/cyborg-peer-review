"""Shape discovery (11-06, pre-registered; three: "shape *discovery* sounds quite promising!!"). Persistent homology of a point
cloud with a shuffle null: which loops (H1) and cavities (H2) beat what independent noise in the same coordinates makes?
    import shapes; shapes.report(points, name)
"""
import numpy as np
from ripser import ripser
def prep(X, k=10):
    X = np.asarray(X, dtype=np.float64); X = X - X.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False); return X @ Vt[:k].T
def bars(P, maxdim):
    dg = ripser(P, maxdim=maxdim)["dgms"]
    return [np.sort(np.nan_to_num(d[:, 1] - d[:, 0], posinf=0))[::-1] for d in dg]
def bars_full(P, maxdim):
    return ripser(P, maxdim=maxdim)["dgms"]
def report(X, name, k=10, maxdim=2, nulls=1000, seed=1106):
    """v2 (11-06 15:05): null + born before half the diameter + before the largest >= 1.5x ratio drop."""
    P = prep(X, k); rng = np.random.default_rng(seed)
    real = bars(P, maxdim); full = bars_full(P, maxdim)
    diam = float(np.max(np.linalg.norm(P[:, None] - P[None], axis=-1)))
    nullmax = [[] for _ in range(maxdim + 1)]
    for _ in range(nulls):
        Q = np.column_stack([rng.permutation(P[:, c]) for c in range(P.shape[1])])
        for h, b in enumerate(bars(Q, maxdim)): nullmax[h].append(b[0] if len(b) else 0.0)
    out = {}
    for h in range(1, maxdim + 1):
        thr = float(np.quantile(nullmax[h], 0.95))
        d = full[h]; d = d[np.isfinite(d[:, 1])]; per = d[:, 1] - d[:, 0]; o = np.argsort(-per); d, per = d[o], per[o]
        early = d[:, 0] <= 0.5 * diam
        cand = per[(per > thr) & early]
        n = 0
        if len(cand):
            nxt = np.r_[cand[1:], per[(per <= thr) | ~early][:1] if ((per <= thr) | ~early).any() else 1e-9]
            ratios = cand / np.maximum(nxt[:len(cand)], 1e-9); k_ = int(np.argmax(ratios))
            n = k_ + 1 if ratios[k_] >= 1.5 else 0
        out[h] = per[:n].tolist() if n else []
        print(f"  {name:30s} H{h}: {n} significant (top bars {np.round(per[:3], 3).tolist()}, born {np.round(d[:3, 0], 2).tolist()}, null95 {thr:.3f}, half-diam {0.5 * diam:.2f})")
    return out
if __name__ == "__main__":
    rng = np.random.default_rng(0); t = rng.uniform(0, 2 * np.pi, 120); s = rng.uniform(0, 2 * np.pi, 300)
    circle = np.c_[np.cos(t), np.sin(t), np.zeros((120, 8))] + rng.normal(0, 0.08, (120, 10))
    a, b = rng.uniform(0, 2 * np.pi, 300), rng.uniform(0, 2 * np.pi, 300)
    torus = np.c_[np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.zeros((300, 6))] + rng.normal(0, 0.05, (300, 10))
    blob = rng.normal(0, 1, (120, 10))
    for X, n in ((circle, "sanity: circle"), (torus, "sanity: torus (4D flat)"), (blob, "sanity: blob")):
        report(X, n, nulls=200)
