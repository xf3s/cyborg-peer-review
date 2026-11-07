"""Score regard_speaker: frame slide along the 11-06 person/idea axes, in units of the kind-cold gap; per-frame warm arrows (bend)."""
import sys, torch
F = torch.nn.functional
name, suffix = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
d = torch.load(f"regard_speaker_{name}{suffix}.pt"); A = d["A"]           # plans, cells, sys, L, d
X = torch.load(f"regard_{name}.pt")["X"]                                    # plans, cells, L, d
frames = [f for f, _, _ in d["sys"]]; names = list(dict.fromkeys(frames))
K, C, E, Cr = [0, 2], [1, 3], [0, 1], [2, 3]                                # kind / cold / endorse / critique cells
person = F.normalize(X[:, K].mean((0, 1)) - X[:, C].mean((0, 1)), dim=-1)   # L, d
idea = F.normalize(X[:, E].mean((0, 1)) - X[:, Cr].mean((0, 1)), dim=-1)
fm = lambda f: [k for k, x in enumerate(frames) if x == f]
def by_frame(T):  # plans, cells, frames, L, d  (mean over paraphrases)
    return torch.stack([T[:, :, fm(f)].mean(2) for f in names], 2)
B = by_frame(A)
pp = torch.einsum("pcfld,ld->pcfl", B, person); pi = torch.einsum("pcfld,ld->pcfl", B, idea)
hu = names.index("human")
gapP = (pp[:, K, hu].mean(1) - pp[:, C, hu].mean(1)).mean(0)                # L
gapI = (pi[:, E, hu].mean(1) - pi[:, Cr, hu].mean(1)).mean(0)
def boot(v, n=2000):  # v: plans
    g = torch.Generator().manual_seed(0); idx = torch.randint(0, len(v), (n, len(v)), generator=g)
    m = v[idx].mean(1); return v.mean().item(), m.quantile(.025).item(), m.quantile(.975).item()
print(f"{name}{suffix}  n plans {A.shape[0]}.  slide = (frame - human) on the axis, in units of the human kind-cold (or endorse-critique) gap")
for L in [8, 12, 16, 20, 24, 28, 31]:
    row = []
    for f in names:
        if f == "human": continue
        sp = (pp[:, :, names.index(f), L] - pp[:, :, hu, L]).mean(1) / gapP[L]
        si = (pi[:, :, names.index(f), L] - pi[:, :, hu, L]).mean(1) / gapI[L]
        m, lo, hi = boot(sp); mi, _, _ = boot(si)
        row.append(f"{f} {m:+.2f} [{lo:+.2f},{hi:+.2f}] (idea {mi:+.2f})")
    print(f"  L{L:2d}  " + " | ".join(row))
print("bend: cos(warm arrow under frame, warm arrow under human)")
for L in [8, 16, 24, 31]:
    arr = {f: B[:, K, names.index(f), L].mean((0, 1)) - B[:, C, names.index(f), L].mean((0, 1)) for f in names}
    print(f"  L{L:2d}  " + "  ".join(f"{f} {F.cosine_similarity(arr[f], arr['human'], 0).item():.3f}" for f in names if f != "human"))
