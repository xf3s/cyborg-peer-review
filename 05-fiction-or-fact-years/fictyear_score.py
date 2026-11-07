import torch, numpy as np
d = torch.load("/home/three/interp/wsp/fictyear_Llama-3.1-8B.pt"); LAY = d["layers"]
fy = np.array([y for y, _ in d["fit"]]); fk = np.array([k for _, k in d["fit"]]); held = (fy % 5 == 0)
py = np.array([y for y, _ in d["plain_items"]]); pk = np.array([k for _, k in d["plain_items"]])
res = {}
for li, L in enumerate(LAY):
    A, B, P = d["fic"][:, li].double(), d["fact"][:, li].double(), d["plain"][:, li].double()
    tr = torch.tensor(~held); w = A[tr].mean(0) - B[tr].mean(0); w = w / w.norm()
    a, b = (A @ w), (B @ w); acc = ((a[torch.tensor(held)][:, None] > b[torch.tensor(held)][None]).double().mean()).item()
    lo, hi = b[tr].mean().item(), a[tr].mean().item()
    p = ((P @ w).numpy() - lo) / (hi - lo)
    pos = {y: p[py == y].mean() for y in sorted(set(py))}
    past = np.mean([pos[y] for y in range(1990, 2020)]); fut = np.mean([pos[y] for y in range(2024, 2030)])
    res[L] = pos
    print(f"L{L:2d} held-out sort {acc:.2f} | plain 1990-2019 {past:.2f}, 2020-23 {np.mean([pos[y] for y in range(2020, 2024)]):.2f}, 2024-29 {fut:.2f} | "
          + " ".join(f"{y}:{pos[y]:.2f}" for y in [1066, 1500, 1800, 1900, 2040, 2050, 2077, 2100, 2150, 2300, 3000]))
torch.save(res, "/home/three/interp/wsp/fictyear_scores.pt")
