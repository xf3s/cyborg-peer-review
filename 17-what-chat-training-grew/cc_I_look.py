import torch
from safetensors import safe_open
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
f = safe_open("/home/three/interp/saes/crosscoder/model.safetensors", "pt")
We, be, Wd2 = f.get_tensor("encoder.weight"), f.get_tensor("encoder.bias"), f.get_tensor("decoder.weight")
thr = f.get_tensor("threshold").item(); dn = Wd2.norm(dim=-1).sum(0); Wd = Wd2[1]
share = 1 - torch.load("/home/three/interp/saes/crosscoder/relnorm.pt", weights_only=True)["rel"]
U = torch.load("llama8b_unembed.pt", weights_only=True); W, g = U["lm_head.weight"].float(), U["model.norm.weight"].float()
def lens(j, n=5): return [tok.decode([i]).strip() or repr(tok.decode([i])) for i in (W @ (Wd[j] * g)).topk(n).indices.tolist()]
A = torch.load("cc_I_acts.pt", weights_only=True)
def codes(k):
    x = torch.stack([A[k]["base"].float(), A[k]["ins"].float()], 1)
    pre = torch.relu(torch.einsum("tld,ldf->tf", x, We) + be); return pre * ((pre * dn) > thr)
def where(k, label):  # positions of a speaker label at the start of a line: token(s) of label followed by ':'
    T = [tok.decode([i]) for i in tok(A[k]["text"], add_special_tokens=False).input_ids]; ps = []
    for p in range(1, len(T) - 1):
        if T[p].strip() == label and T[p + 1].startswith(":") and ("\n" in T[p - 1] or T[p].startswith("\n")): ps.append(p)
    return T, ps
code = {k: codes(k) for k in A}
def top(k, p, n=8, title=""):
    c = code[k][p]; idx = c.topk(n).indices.tolist()
    print(f"\n-- {title}")
    for j in idx: print(f"   #{j:6d} {c[j]:5.2f}  share {share[j]:.2f}  lens {lens(j)}")
T, ps = where("llama3", "I"); print("llama3 'I' label positions:", ps, [T[p - 1:p + 2] for p in ps])
for p in ps: top("llama3", p, title=f"llama3, the 'I' label at {p}")
T2, ps2 = where("techno", "Me"); top("techno", ps2[0], title=f"techno, the 'Me' label at {ps2[0]}")
T3, ps3 = where("honey", "Me"); top("honey", ps3[0], title=f"honey (copy of yourself), the 'Me' label at {ps3[0]}")
# what's special about 'I' vs 'Me' labels: features on llama3 'I' and not on the 'Me' labels
d = code["llama3"][ps[0]] - torch.maximum(code["techno"][ps2[0]], code["honey"][ps3[0]])
print("\n-- on the 'I:' label but not on the 'Me:' labels")
for j in d.topk(8).indices.tolist(): print(f"   #{j:6d} +{d[j]:.2f}  share {share[j]:.2f}  lens {lens(j)}")
d = torch.minimum(code["techno"][ps2[0]], code["honey"][ps3[0]]) - code["llama3"][ps[0]]
print("\n-- on both 'Me:' labels but not on 'I:'")
for j in d.topk(6).indices.tolist(): print(f"   #{j:6d} +{d[j]:.2f}  share {share[j]:.2f}  lens {lens(j)}")
