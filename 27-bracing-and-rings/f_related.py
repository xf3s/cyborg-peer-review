import torch, numpy as np
src = open("f_explore.py").read(); exec(src[:src.index("for j in (45294, 121007):")])
f2 = f
J = 45294
print("== nearest decoders (Instruct), cosine, with Instruct share")
c = Dn @ Dn[J]; 
for k in c.topk(13).indices.tolist()[1:]: print(f"  #{k:6d} cos {float(c[k]):+.2f} share {share[k]:.2f}  {lens(k, 7)}")
print("== most opposite decoders")
for k in (-c).topk(7).indices.tolist(): print(f"  #{k:6d} cos {float(c[k]):+.2f} share {share[k]:.2f}  {lens(k, 7)}")
AF = enc([J]); fire_b, fire_i, other_b, other_i = [], [], [], []
rng = np.random.default_rng(0)
for name, t, b, i in docs:
    ids = tok(t, add_special_tokens=False).input_ids
    if len(ids) != b.shape[0]: continue
    a = AF(b, i)
    for k in range(1, len(a)):
        if a[k] > 0.02: fire_b.append(b[k].float()); fire_i.append(i[k].float())
        elif rng.random() < .04: other_b.append(b[k].float()); other_i.append(i[k].float())
print("firing tokens", len(fire_b), "other sample", len(other_b), flush=True)
We, be, Wd, thr = f.get_tensor("encoder.weight").float(), f.get_tensor("encoder.bias").float(), f.get_tensor("decoder.weight").float(), f.get_tensor("threshold").item()
dn = Wd.norm(dim=-1).sum(0)
def E(bl, il):
    z = torch.stack([torch.stack(bl), torch.stack(il)], 1); out = []
    for c0 in range(0, len(z), 64):
        pre = torch.relu(torch.einsum("tld,ldf->tf", z[c0:c0 + 64], We) + be); out.append((pre * ((pre * dn) > thr)).mean(0) * len(z[c0:c0 + 64]))
    return torch.stack(out).sum(0) / len(z)
F, O = E(fire_b, fire_i), E(other_b, other_i)
d = F - O
print("== fires WITH #45294 (mean at its firing tokens minus elsewhere), excluding itself")
for k in d.topk(16).indices.tolist():
    if k == J: continue
    print(f"  #{k:6d} with {float(F[k]):.2f} else {float(O[k]):.2f} share {share[k]:.2f}  {lens(k, 7)}")
print("== quieter when #45294 fires")
for k in (-d).topk(8).indices.tolist(): print(f"  #{k:6d} with {float(F[k]):.2f} else {float(O[k]):.2f} share {share[k]:.2f}  {lens(k, 7)}")
