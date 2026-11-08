import torch, json, numpy as np
from safetensors import safe_open
D = torch.load("unfin.pt", weights_only=True); it = D["items"]; Hi, Hb = D["ins"].float(), D["base"].float()
f = safe_open("/home/three/interp/saes/crosscoder/model.safetensors", "pt")
J = 117769
We = f.get_tensor("encoder.weight")[:, :, J]; be = f.get_tensor("encoder.bias")[J]; Wd = f.get_tensor("decoder.weight")[:, J]  # [2,4096],[2,4096]
dn = Wd.norm(dim=-1).sum(); thr = f.get_tensor("threshold").item()
pre = torch.relu(Hb @ We[0] + Hi[:, 16] @ We[1] + be); act = pre * ((pre * dn) > thr)  # slot 0 base, 1 Instruct
idx = {x: n for n, x in enumerate(it)}
def m(role, kind): v = np.array([float(act[idx[(i, role, kind)]]) for i in range(20)]); return v
print("feature #117769 at the final end-of-turn token (crosscoder, L16):")
for role, kinds in (("assistant", ("complete", "cut", "partial")), ("user", ("complete", "cut"))):
    for k in kinds: v = m(role, k); print(f"  {role:9} {k:8} mean {v.mean():.2f}  fires in {int((v > 0).sum())}/20")
for k in ("cut", "partial"):
    print(f"  assistant {k} > complete in {int((m('assistant', k) > m('assistant', 'complete')).sum())}/20 pairs")
# across layers: projection on the Instruct-half decoder direction
u = Wd[1] / Wd[1].norm()
print("\nalong the layers: projection on its direction at the end token (assistant cut − complete | partial − complete | user cut − complete)")
for L in range(0, 32, 2):
    p = lambda r, k: np.array([float(Hi[idx[(i, r, k)], L] @ u) for i in range(20)])
    a = p("assistant", "cut") - p("assistant", "complete"); b = p("assistant", "partial") - p("assistant", "complete"); c = p("user", "cut") - p("user", "complete")
    print(f"  L{L:2d}  {a.mean():+6.2f} ({(a > 0).sum():2d}/20)   {b.mean():+6.2f} ({(b > 0).sum():2d}/20)   {c.mean():+6.2f}")
