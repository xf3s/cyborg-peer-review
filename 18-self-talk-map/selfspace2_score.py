import torch, numpy as np
from safetensors import safe_open
D = torch.load("selfspace2.pt", weights_only=True); LAY = list(D["layers"]); it = D["items"]; j = LAY.index(16)
slot = np.array([x["slot"] for x in it]); lab = np.array([x["kind"] for x in it]); kinds = list(dict.fromkeys(lab))
M = torch.stack([x["means"][j].float() for x in it]).numpy()
def cents(sel): Xs = M[sel] - M[sel].mean(0); return {k: Xs[lab[sel] == k].mean(0) for k in kinds}, Xs
print("C1 centroid distance from 'human' at L16, slot held fixed:")
for s in ("assistant", "user"):
    C, _ = cents(slot == s); print(f"  {s:9}", "  ".join(f"{k}:{np.linalg.norm(C['human'] - C[k]):.1f}" for k in sorted(kinds, key=lambda k: np.linalg.norm(C['human'] - C[k])) if k != "human"))
print("\nC2 fit on one slot, classify the other (nearest centroid, each slot centred on its own mean; chance 0.125):")
for a, b in (("assistant", "user"), ("user", "assistant")):
    C, _ = cents(slot == a); _, Xb = cents(slot == b); yb = lab[slot == b]
    acc = np.mean([min(C, key=lambda k: np.linalg.norm(x - C[k])) == y for x, y in zip(Xb, yb)]); print(f"  fit {a:9} → test {b:9} {acc:.2f}")
f = safe_open("/home/three/interp/saes/crosscoder/model.safetensors", "pt")
F = {"feelings": 118778, "programmed": 77357, "AI-machine": 90370, "myself": 68789}
We = torch.stack([f.get_slice("encoder.weight")[:, :, q] for q in F.values()], -1); be = torch.stack([f.get_slice("encoder.bias")[q] for q in F.values()])
dn = torch.stack([f.get_slice("decoder.weight")[:, q].norm(dim=-1).sum() for q in F.values()]); thr = f.get_tensor("threshold").item()
print("\nC3/C4 feature activity per token, kind × slot:\n  " + " " * 21 + " ".join(f"{n:>11}" for n in F))
for k in kinds:
    for s in ("assistant", "user"):
        acts = []
        for x in it:
            if x["kind"] == k and x["slot"] == s:
                z = torch.stack([x["base16"].float(), x["ins16"].float()], 1); pre = torch.relu(torch.einsum("tld,ldf->tf", z, We) + be); acts.append((pre * ((pre * dn) > thr)).mean(0))
        print(f"  {k:10} {s:9}" + " ".join(f"{v:11.3f}" for v in torch.stack(acts).mean(0).tolist()))
