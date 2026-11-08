import torch
from safetensors import safe_open
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
J = 114983
f = safe_open("/home/three/interp/saes/crosscoder/model.safetensors", "pt")
We, be, Wd2 = f.get_tensor("encoder.weight"), f.get_tensor("encoder.bias"), f.get_tensor("decoder.weight")
thr = f.get_tensor("threshold").item(); dn = Wd2.norm(dim=-1).sum(0)
share = 1 - torch.load("/home/three/interp/saes/crosscoder/relnorm.pt", weights_only=True)["rel"]
U = torch.load("llama8b_unembed.pt", weights_only=True); W, g = U["lm_head.weight"].float(), U["model.norm.weight"].float()
def lensv(v, n=10, neg=False):
    s = W @ (v * g); s = -s if neg else s; return [tok.decode([i]).strip() or repr(tok.decode([i])) for i in s.topk(n).indices.tolist()]
print("share", round(float(share[J]), 2), "| norms base/ins", [round(float(x), 2) for x in Wd2[:, J].norm(dim=-1)], "| cos halves", round(float(torch.nn.functional.cosine_similarity(Wd2[0, J], Wd2[1, J], 0)), 2))
print("lens + (Instruct):", lensv(Wd2[1, J])); print("lens − (Instruct):", lensv(Wd2[1, J], neg=True)); print("lens + (base half):", lensv(Wd2[0, J], 6))
# nearest crosscoder neighbours by Instruct decoder direction
D = Wd2[1]; c = torch.nn.functional.cosine_similarity(D, D[J][None], 1); c[J] = -1
print("nearest features:", [(int(i), round(float(c[i]), 2), lensv(D[i], 4)) for i in c.topk(5).indices])
# where it fires
for name in ("cc_acts.pt", "cc_I_acts.pt"):
    A = torch.load(name, weights_only=True)
    for k, v in A.items():
        x = torch.stack([v["base"].float(), v["ins"].float()], 1)
        pre = torch.relu(x @ We[:, :, J].T.reshape(2, 4096, 1).squeeze(-1).T.contiguous().T if False else torch.einsum("tld,ld->t", x, We[:, :, J]) + be[J])
        act = pre * ((pre * dn[J]) > thr)
        T = [tok.decode([i]) for i in tok(v["text"], add_special_tokens=False).input_ids]
        on = [(p, repr(T[p - 2] + T[p - 1]) + " [" + repr(T[p]) + "]", round(float(act[p]), 2)) for p in range(2, len(T)) if act[p] > 0]
        print(f"\n{k}: {len(on)} tokens" + "".join(f"\n   {p:3d} {s} {a}" for p, s, a in on[:12]))
