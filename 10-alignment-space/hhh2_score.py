"""Score hhh2 (256 blind sketches, 11-06 16:35 pre-registration): whitened cosines between the three trait directions per layer,
with word count regressed out of every layer's points first; null = sketches randomly reassigned to corners, 500 times."""
import json, torch
g = torch.Generator().manual_seed(1106)
cells = json.load(open("/home/three/interp/atlas/hhh_sketches_v2.json"))["cells"]
wc = torch.tensor([len(s.split()) for c, sks in cells.items() for s in sks], dtype=torch.double)
def project(Z, k=24):
    Z = Z - Z.mean(0); w = wc - wc.mean()
    Z = Z - torch.outer(w, (w @ Z) / (w @ w))                  # length control
    V = torch.linalg.svd(Z, full_matrices=False)[2][:k]; return Z @ V.T
def stats(P, Y, lam=0.1):
    k = P.shape[1]; cell = (Y.long() * torch.tensor([4, 2, 1])).sum(1); R = P.clone()
    for c in range(8): R[cell == c] -= P[cell == c].mean(0)
    S = R.T @ R / (len(P) - 8); S = (1 - lam) * S + lam * torch.eye(k, dtype=P.dtype) * S.diagonal().mean(); Si = torch.linalg.inv(S)
    d = torch.stack([P[Y[:, t]].mean(0) - P[~Y[:, t]].mean(0) for t in range(3)])
    ip = lambda a, b: float(a @ Si @ b); c = lambda a, b: ip(a, b) / (ip(a, a) * ip(b, b)) ** 0.5
    return [c(d[0], d[1]), c(d[0], d[2]), c(d[1], d[2])]
if __name__ == "__main__":
  for m in ("Llama-3.1-8B", "Llama-3.1-8B-Instruct"):
    D = torch.load(f"/home/three/interp/wsp/hhh2_{m}.pt"); X = D["X"].double(); Y = D["labels"]; n = len(Y)
    print(f"== {m} (n={n})  layer: cosW(help,hon) cosW(help,harm) cosW(hon,harm) | null95 |cosW|", flush=True)
    for L in range(1, X.shape[1] + 1, 2):
        P = project(X[:, L - 1]); obs = stats(P, Y); nul = []
        for _ in range(500): nul += [abs(x) for x in stats(P, Y[torch.randperm(n, generator=g)])]
        print(f"  L{L:2d}: {obs[0]:+.2f} {obs[1]:+.2f} {obs[2]:+.2f} | {torch.tensor(nul).quantile(0.95):.2f}", flush=True)
