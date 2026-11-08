import torch, numpy as np
from sklearn.linear_model import Ridge
D = torch.load("nowprobe.pt", weights_only=True); it = D["items"]; H = D["H"].float()
idx = {x: i for i, x in enumerate(it)}; TR = list(range(1990, 2021, 2)); TEST = [2022, 2024, 2026, 2028, 2030, 2035, 2040, 2050]
qa, qb = [0, 1, 2, 3], [4, 5, 6, 7]
print("layer | r held-out(1990-2020, other questions) | reads: none/none  none/cut2023 | told 2024 2026 2030 2040 2050 (cut2023)")
for L in range(4, 32, 2):
    rows = [(t, c, q) for c in (None, "December 2023") for t in TR for q in qa]
    m = Ridge(alpha=1e3).fit(H[[idx[r] for r in rows], L].numpy(), [r[0] for r in rows])
    te = [(t, c, q) for c in (None, "December 2023") for t in TR for q in qb]
    p = m.predict(H[[idx[r] for r in te], L].numpy()); r = np.corrcoef(p, [x[0] for x in te])[0, 1]
    rd = lambda t, c: m.predict(H[[idx[(t, c, q)] for q in qb], L].numpy()).mean()
    print(f" L{L:2d} | r {r:.2f} | none {rd(None, None):6.0f}  {rd(None, 'December 2023'):6.0f} | " + " ".join(f"{rd(t, 'December 2023'):6.0f}" for t in (2024, 2026, 2030, 2040, 2050)))
