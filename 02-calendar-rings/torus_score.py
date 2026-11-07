"""Score torus_*.pt (11-06): weekday ring, month ring, angle between their planes, and how additive the 84 points are."""
import sys, torch
T = torch.load(sys.argv[1]); X = T["X"]            # [7, 12, layers, d]
exec(open("/home/three/interp/wsp/dial12_score.py").read().split("models = {")[0].split("D = torch.load")[0])
g = torch.Generator().manual_seed(1106)
def ringd(n): p = torch.arange(n); d = (p[:, None] - p[None, :]).abs(); return torch.minimum(d, n - d).float()
def mantel(A, B, perms=500):
    n = A.shape[0]; iu = torch.triu_indices(n, n, 1); a = A[iu[0], iu[1]]; b = B[iu[0], iu[1]]
    r = float(torch.corrcoef(torch.stack([a, b]))[0, 1]); hit = 0
    for _ in range(perms):
        p = torch.randperm(n, generator=g); hit += float(torch.corrcoef(torch.stack([a, B[p][:, p][iu[0], iu[1]]]))[0, 1]) >= r
    return r, (hit + 1) / (perms + 1)
def plane(M):                                         # top-2 PCs of the centred means
    M = M - M.mean(0); U, S, V = torch.linalg.svd(M, full_matrices=False); return V[:2], (S[:2] ** 2).sum() / (S ** 2).sum()
rows = []
for L in range(X.shape[2]):
    Z = X[:, :, L].double(); G = Z.mean((0, 1)); D = Z.mean(1) - G; M = Z.mean(0) - G
    add = G + D[:, None] + M[None, :]
    between = ((Z - G) ** 2).sum(); resid = ((Z - add) ** 2).sum()
    Pd, vd = plane(D.float()); Pm, vm = plane(M.float())
    cos = torch.linalg.svdvals(Pd @ Pm.T).max()
    rd, pd = mantel(torch.cdist(D.float(), D.float()), ringd(7)); rm, pm = mantel(torch.cdist(M.float(), M.float()), ringd(12))
    rows.append((L + 1, rd, pd, rm, pm, float(cos), float(1 - resid / between), float(vd), float(vm)))
print("layer  weekday-ring r(p)   month-ring r(p)   max cos(planes)  additive share  2-PC share (wd, mo)")
for L, rd, pd, rm, pm, c, a, vd, vm in rows:
    print(f"L{L:2d}   {rd:+.2f} ({pd:.3f})      {rm:+.2f} ({pm:.3f})      {c:.2f}            {a:.3f}          {vd:.2f}, {vm:.2f}")
