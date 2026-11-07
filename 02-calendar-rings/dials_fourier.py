"""Split the weekday / month rings into Fourier dials (11-07, three: "two dials at once - can we separate it out?").
Context-removed item means (cloud20, L15, full 4096 dims). Harmonic k's share of the between-item variance, vs 2,000 order shuffles."""
import torch, json, math, numpy as np
C = torch.load("/home/three/interp/wsp/cloud20_Llama-3.1-8B.pt"); out = {}
for name, n in (("weekdays", 7), ("months", 12)):
    X = C[name].double().reshape(n, 20, -1); X = X - X.mean(0, keepdim=True); M = X.mean(1); M = M - M.mean(0)
    def shares(order):
        Z = torch.fft.fft(M[order].to(torch.complex128), dim=0)       # n, d
        e = (Z.abs() ** 2).sum(1)[1:]                                  # drop k=0 (zero)
        ks = range(1, n // 2 + 1)
        tot = e.sum(); return [((e[k - 1] + (e[n - k - 1] if n - k != k else 0)) / tot).item() for k in ks]
    obs = shares(torch.arange(n)); g = torch.Generator().manual_seed(0)
    null = np.array([shares(torch.randperm(n, generator=g)) for _ in range(2000)])
    print(f"{name}: share of the ring's shape carried by each dial (k = times around per cycle)")
    for k, s in enumerate(obs, 1):
        p = (null[:, k - 1] >= s).mean()
        print(f"   k={k}: {s:.1%}   (shuffled order: {np.median(null[:, k-1]):.1%} typical, p {p:.4f})")
    Z = torch.fft.fft(M.to(torch.complex128), dim=0)
    planes = {}
    for k in range(1, n // 2 + 1):
        u = Z[k].real; v = Z[k].imag; B = torch.stack([u, v]); Q, _ = torch.linalg.qr(B.T)   # the dial's plane
        P = (M @ Q).numpy(); planes[k] = (P / np.abs(P).max()).round(4).tolist()
    out[name] = {"shares": obs, "planes": planes}
json.dump(out, open("/home/three/umber-drafts/hoop/dials.json", "w"))
