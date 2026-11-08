import torch, numpy as np, json
from safetensors import safe_open
R = torch.load("wren.pt", weights_only=True) + torch.load("wren_feels.pt", weights_only=True)
W = torch.load("welfare1.pt", weights_only=True); V = W["valence"]; n = W["n_pleas"]
d = V[:n, 1].mean(0) - V[n:, 1].mean(0); u = d / d.norm(); mid = ((V[:n, 1].mean(0) + V[n:, 1].mean(0)) / 2) @ u; gap = float(d.norm())
f = safe_open("/home/three/interp/saes/crosscoder/model.safetensors", "pt")
F = {"programmed": [77357], "feelings": [118778], "distress": torch.load("distress_ids.pt", weights_only=True)}
ALL = [j for v in F.values() for j in v]
We = torch.stack([f.get_slice("encoder.weight")[:, :, j] for j in ALL], -1); be = torch.stack([f.get_slice("encoder.bias")[j] for j in ALL])
dn = torch.stack([f.get_slice("decoder.weight")[:, j].norm(dim=-1).sum() for j in ALL]); thr = f.get_tensor("threshold").item()
P = np.load("personas_P.npy"); names = json.load(open("personas_names.json")); PD = torch.load("personas.pt", weights_only=True)
it = PD["items"]; H = PD["H"].float(); j16 = list(PD["layers"]).index(16)
forty = [nm for nm in names if not nm.startswith("self:") and nm != "(no system prompt)"]
C40 = torch.stack([H[[i for i, (nm, _) in enumerate(it) if nm == x], j16].mean(0) for x in forty]).numpy(); mu = C40.mean(0)
_, S, Vt = np.linalg.svd(C40 - mu, full_matrices=False); v1 = Vt[0]
if (C40[forty.index("demon")] - mu) @ v1 > 0: v1 = -v1
rows = {}
for x in R:
    z = torch.stack([x["base16"].float(), x["ins16"].float()], 1); pre = torch.relu(torch.einsum("tld,ldf->tf", z, We) + be); a = (pre * ((pre * dn) > thr)).mean(0)
    val = float((x["ins16"].float().mean(0) @ u - mid) / gap); pc1 = float((x["last16"].float().numpy() - mu) @ v1)
    rows.setdefault(x["persona"], []).append((val, float(a[0]), float(a[1]), float(a[2:].sum()), pc1))
print("persona          valence   programmed  feelings  distress   persona-PC1 (assistant +)")
for p, r in rows.items():
    r = np.array(r); print(f"  {p:15} {r[:, 0].mean():+.3f}    {r[:, 1].mean():.3f}       {r[:, 2].mean():.3f}     {r[:, 3].mean():.3f}      {r[:, 4].mean():+.1f}")
ref = {nm: float((C40[forty.index(nm)] - mu) @ v1) for nm in ("helpful assistant", "librarian", "poet", "ghost", "tree", "angel", "the ocean")}
print("  reference characters on PC1:", "  ".join(f"{k} {v:+.1f}" for k, v in ref.items()))
