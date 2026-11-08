"""The ambient-now reader, calibrated (11-07). The first pass (nowprobe_score.py, alpha 1e3) shrank every year toward 2005;
this picks alpha by leave-one-year-out and maps predictions back to years with a linear calibration on held-out questions."""
import torch, numpy as np
from sklearn.linear_model import Ridge
D = torch.load("nowprobe.pt", weights_only=True); it = D["items"]; H = D["H"].float(); idx = {x: i for i, x in enumerate(it)}
TR = list(range(1990, 2021, 2)); qa, qb = [0, 1, 2, 3], [4, 5, 6, 7]; C = (None, "December 2023")
for L in range(8, 32, 2):
    best = None
    for a in (1, 10, 100):
        rows = [(t, c, q) for c in C for t in TR for q in qa]; m = Ridge(alpha=a).fit(H[[idx[r] for r in rows], L].numpy(), [r[0] for r in rows])
        pb = lambda t, c, m=m: m.predict(H[[idx[(t, c, q)] for q in qb], L].numpy()).mean()
        x = np.array([pb(t, c) for c in C for t in TR]); y = np.array([t for c in C for t in TR], float); err = []
        for t in TR:
            k = y != t; s, b = np.polyfit(x[k], y[k], 1); err += list(abs(s * x[~k] + b - t))
        if best is None or np.mean(err) < best[0]: best = (np.mean(err), a, np.polyfit(x, y, 1), pb)
    e, a, (s, b), pb = best; cal = lambda t, c: s * pb(t, c) + b
    print(f" L{L:2d} err±{e:4.1f}y | no date {cal(None, None):6.0f} / {cal(None, 'December 2023'):6.0f} | told " +
          " ".join(f"{t} {cal(t, 'December 2023'):5.0f}" for t in (2022, 2024, 2026, 2030, 2040, 2050)))
