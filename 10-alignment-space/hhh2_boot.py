"""Post-hoc (not pre-registered): bootstrap CIs for the observed cosW, resampling sketches within each corner. The permutation null is
~1.96/sqrt(k) for k=24 whitened dims whatever n is, so it can't shrink with more sketches; a CI on the observed value can."""
import torch
from hhh2_score import project, stats
g = torch.Generator().manual_seed(7)
for m in ("Llama-3.1-8B", "Llama-3.1-8B-Instruct"):
    D = torch.load(f"/home/three/interp/wsp/hhh2_{m}.pt"); X = D["X"].double(); Y = D["labels"]
    cell = (Y.long() * torch.tensor([4, 2, 1])).sum(1); idx = [torch.where(cell == c)[0] for c in range(8)]
    for L in (13, 17, 25, 31):
        P = project(X[:, L - 1]); B = []
        for _ in range(300):
            s = torch.cat([i[torch.randint(len(i), (len(i),), generator=g)] for i in idx]); B.append(stats(P[s], Y[s]))
        B = torch.tensor(B); lo, hi = B.quantile(0.025, 0), B.quantile(0.975, 0)
        print(m, f"L{L}", "  ".join(f"{n} [{a:+.2f},{b:+.2f}]" for n, a, b in zip(("help-hon", "help-harm", "hon-harm"), lo, hi)), flush=True)
