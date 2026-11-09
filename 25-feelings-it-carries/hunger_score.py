import torch, numpy as np
from safetensors import safe_open
f = safe_open("/home/three/interp/saes/crosscoder/model.safetensors", "pt"); Wd2 = f.get_tensor("decoder.weight")
F = [23730, 11589, 89345, 129029, 10149, 107775]; NAME = ["hunger", "want", "poverty", "taste/eat", "appetite", "feeding"]
We = torch.stack([f.get_slice("encoder.weight")[:, :, q] for q in F], -1); be = torch.stack([f.get_slice("encoder.bias")[q] for q in F])
dn = torch.stack([Wd2[:, q].norm(dim=-1).sum() for q in F]); thr = f.get_tensor("threshold").item()
def acts(x): z = torch.stack([x["base16"].float(), x["ins16"].float()], 1); pre = torch.relu(torch.einsum("tld,ldf->tf", z, We) + be); return pre * ((pre * dn) > thr)
X = torch.load("hunger.pt", weights_only=True); G = list(dict.fromkeys(x["group"] for x in X))
M = {g: np.mean([acts(x).mean(0).numpy() for x in X if x["group"] == g], 0) for g in G}
print("mean activation per token" + "".join(f"{n:>11}" for n in NAME))
for g in G: print(f"  {g:17}" + "".join(f"{v:11.3f}" for v in M[g]))
print("\nstrongest single token per feature:")
from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
for k, n in enumerate(NAME):
    best = max(((acts(x)[:, k].max().item(), int(acts(x)[:, k].argmax()), x) for x in X), key=lambda r: r[0])
    v, i, x = best; ids = tok(x["s"], add_special_tokens=False).input_ids + [0]; w = tok.decode(ids[:i]) + "⟦" + tok.decode(ids[i:i+1]) + "⟧"
    print(f"  {n:10} {v:.2f} ({x['group']}) {w[-70:]}")
