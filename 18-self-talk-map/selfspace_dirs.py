import torch, numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
D = torch.load("selfspace.pt", weights_only=True); it = D["items"]; j = list(D["layers"]).index(16)
lab = np.array([x["kind"] for x in it]); A = list(dict.fromkeys(lab))[:6]; m = np.isin(lab, A)
X = torch.stack([x["means"][j].float() for x in it])[torch.tensor(m)]; S_ = [x["s"] for x, k in zip(it, m) if k]; L = lab[m]
mu = X.mean(0); Y = X - mu; U, S, Vt = torch.linalg.svd(Y, full_matrices=False); P = Y @ Vt[:3].T
Uw = torch.load("llama8b_unembed.pt", weights_only=True); W, g = Uw["lm_head.weight"].float(), Uw["model.norm.weight"].float()
def lens(v, n=10): s = W @ (v * g); return [tok.decode([i]).strip() for i in s.topk(n).indices.tolist()]
f = safe_open("/home/three/interp/saes/crosscoder/model.safetensors", "pt"); Dd = f.get_slice("decoder.weight")[1]
share = 1 - torch.load("/home/three/interp/saes/crosscoder/relnorm.pt", weights_only=True)["rel"]
Dn = Dd / Dd.norm(dim=1, keepdim=True)
R = torch.load("regard_Llama-3.1-8B-Instruct.pt", weights_only=True)["X"].float()[:, :, 16]
warm = (R[:, 0] + R[:, 2] - R[:, 1] - R[:, 3]).mean(0); idea = (R[:, 0] + R[:, 1] - R[:, 2] - R[:, 3]).mean(0)
ro = torch.load("role_Llama-3.1-8B-Instruct.pt", weights_only=True); role = ro["assistant"][:, 16].float().mean(0) - ro["user"][:, 16].float().mean(0)
cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, 0))
for d in range(3):
    v = Vt[d]; p = P[:, d]; o = torch.argsort(p)
    print(f"\n===== direction {d + 1}  ({float(S[d] ** 2 / (S ** 2).sum()) * 100:.0f}% of the spread)")
    print("  group averages:", "  ".join(f"{k} {float(p[torch.tensor(L == k)].mean()):+.1f}" for k in sorted(A, key=lambda k: float(p[torch.tensor(L == k)].mean()))))
    print("  most −:", [S_[i] for i in o[:3].tolist()]); print("  most +:", [S_[i] for i in o[-3:].tolist()][::-1])
    print("  words, + end:", lens(v)); print("  words, − end:", lens(-v))
    c = Dn @ v
    for sign, name in ((1, "+"), (-1, "−")):
        top = (sign * c).topk(5).indices.tolist()
        print(f"  features along {name}:", [(jj, round(float(c[jj]), 2), round(float(share[jj]), 2), lens(Dd[jj], 4)) for jj in top])
    print(f"  cos with warmth {cos(v, warm):+.2f} | flattery/endorse {cos(v, idea):+.2f} | 'it's me talking' (assistant−user) {cos(v, role):+.2f}")
