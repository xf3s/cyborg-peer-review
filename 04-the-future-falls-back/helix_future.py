"""Do dates past Llama-3.1's training cutoff (~end 2023) go unstable, and are they pulled one way? (11-07, three)"""
import torch, numpy as np
F = torch.nn.functional
d = torch.load("/home/three/interp/wsp/helix_Llama-3.1-8B.pt"); K = torch.tensor([k for *_, k in d["dates"]])
yrs = np.arange(1990, 2030)
for li, L in enumerate(d["layers"]):
    X = d["D"][:, li].double()
    for k in (0, 1): X[K == k] -= X[K == k].mean(0)
    T = X.view(2, 40, 12, -1)                                          # template, year, month, dim
    P = T.mean(0); ym = P.mean(1)                                      # year means
    fit = torch.tensor((yrs >= 1990) & (yrs <= 2019))                  # fit the time line well inside the training data
    yc = torch.tensor(yrs[fit.numpy()] - yrs[fit.numpy()].mean()).double()
    base = ym[fit].mean(0); w = (yc[:, None] * (ym[fit] - base)).sum(0); step = w.norm() / (yc ** 2).sum(); w = w / w.norm()
    t = ((ym - base) @ w).numpy() / step.item() + yrs[fit.numpy()].mean()        # where each year sits on the line, in "years"
    pred = base + torch.tensor(t - yrs[fit.numpy()].mean())[:, None] * step * w   # point on the line
    off = ym - base - ((ym - base) @ w)[:, None] * w                   # off-line part
    offn = off.norm(dim=1).numpy()
    agree = F.cosine_similarity((T[0] - T[0].mean((0, 1))).reshape(40, -1), (T[1] - T[1].mean((0, 1))).reshape(40, -1), 1).numpy()
    ringsz = (P - P.mean(1, keepdim=True)).norm(dim=-1).mean(1).numpy()
    past, fut = yrs <= 2023, yrs >= 2024
    O = F.normalize(off - off[torch.tensor(yrs <= 2019)].mean(0), dim=1)
    fo = O[torch.tensor(fut)]; po = O[torch.tensor((yrs >= 2010) & (yrs <= 2019))]
    pc_f = (fo @ fo.T)[~torch.eye(6, dtype=bool)].mean().item(); pc_p = (po @ po.T)[~torch.eye(10, dtype=bool)].mean().item()
    print(f"L{L:2d} line position (in years) 2020-29: {np.round(t[30:], 1).tolist()}")
    print(f"     off-line distance: 2010-19 {offn[20:30].mean():.2f}, 2020-23 {offn[30:34].mean():.2f}, 2024-29 {offn[34:].mean():.2f} | "
          f"templates agree: <=2023 {agree[past].mean():.2f}, >=2024 {agree[fut].mean():.2f} | ring size <=2023 {ringsz[past].mean():.2f} >=2024 {ringsz[fut].mean():.2f}")
    print(f"     same pull? mean cos between future years' off-line drifts {pc_f:+.2f} (2010s with each other {pc_p:+.2f})")
