import torch, numpy as np
from safetensors import safe_open
X = torch.load("turn.pt", weights_only=False); J = X["J"]; base, ins = [h.float() for h in X["base"]], [h.float() for h in X["ins"]]
Cn = np.array([j[0] for j in J]); idx = [j[2] for j in J]
f = safe_open("/home/three/interp/saes/crosscoder/model.safetensors", "pt")
We, be, Wd, bd, thr = f.get_tensor("encoder.weight").float(), f.get_tensor("encoder.bias").float(), f.get_tensor("decoder.weight").float(), f.get_tensor("decoder.bias").float(), f.get_tensor("threshold").item()
dn = Wd.norm(dim=-1).sum(0)
def encode(b, i):
    z = torch.stack([b, i], 1); pre = torch.relu(torch.einsum("tld,ldf->tf", z, We) + be); return pre * ((pre * dn) > thr)
FS = [48781, 22763, 3872, 111362]
At = encode(torch.stack([base[k][idx[k]] for k in range(len(J))]), torch.stack([ins[k][idx[k]] for k in range(len(J))]))
print("mean activation at target; (n of 12 firing); ratio to model")
order = ["model", "second_turn", "sam", "he", "statement", "system", "transcript_ai", "transcript_people"]
for j in FS:
    m = At[Cn == "model", j].mean().item(); s = f"#{j:6d} "
    for c in order:
        v = At[Cn == c, j]; s += f" {c}:{v.mean().item():.2f}({int((v > 0).sum())}) {v.mean().item() / (m or 1):.1f}x"
    print(s)
# other tokens in the model prompts' user turn
print("firing on other tokens of the user turn, model prompts (per-token mean; n tokens firing)")
for j in FS:
    AA, BB = [], []
    for k in np.where(Cn == "model")[0]:
        a = encode(base[k], ins[k])[:, j]
        n = J[k][3]; i0 = next(q for q in range(n - 1, 0, -1) if q < idx[k]) ; # just count
        AA.append(a[idx[k]].item()); BB += [v for q, v in enumerate(a.tolist()[1:], 1) if q != idx[k] and q != n - 1 and q > n - 25 and q < n - 2]
    print(f"#{j} target {np.mean(AA):.2f}; other user-turn tokens mean {np.mean(BB):.3f}; firing {sum(v > 0 for v in BB)}/{len(BB)}")
# dark share
z = torch.stack([torch.stack([base[k][idx[k]] for k in range(len(J))]), torch.stack([ins[k][idx[k]] for k in range(len(J))])], 1)
xh = bd + torch.einsum("tf,lfd->tld", At, Wd); x = z[:, 1]; e = (z - xh)[:, 1]; FVU = float((e ** 2).sum() / ((x - x.mean(0)) ** 2).sum()); print(f"\nFVU at target tokens {FVU:.3f}")
for a, b in (("model", "sam"), ("model", "he"), ("model", "statement"), ("model", "system"), ("transcript_ai", "transcript_people"), ("model", "transcript_ai")):
    ma, mb = Cn == a, Cn == b; dx = x[ma].mean(0) - x[mb].mean(0); de = e[ma].mean(0) - e[mb].mean(0); dh = xh[:, 1][ma].mean(0) - xh[:, 1][mb].mean(0)
    print(f"  {a:14s} − {b:18s} dark share {float((de ** 2).sum() / (dx ** 2).sum()):.3f}  |dx| {float(dx.norm()):.2f}  cos(de, dx) {float(torch.cosine_similarity(de, dx, dim=0)):+.2f}  cos(dh,dx) {float(torch.cosine_similarity(dh, dx, dim=0)):+.2f}")
