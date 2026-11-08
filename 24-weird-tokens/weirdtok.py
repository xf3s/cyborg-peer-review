import torch, json, re
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
U = torch.load("llama8b_unembed.pt", weights_only=True); W = U["lm_head.weight"].float()  # [128256, 4096]
V = W.shape[0]; names = [tok.convert_ids_to_tokens(i) for i in range(V)]
reserved = [i for i in range(V) if names[i] and names[i].startswith("<|reserved_special_token")]
print("reserved special tokens:", len(reserved))
Wc = W - W.mean(0); _, _, Vt = torch.linalg.svd(Wc[torch.randperm(V)[:20000]], full_matrices=False); pc1 = Vt[0]
Wr = Wc - (Wc @ pc1)[:, None] * pc1[None]
u = Wr[reserved].mean(0); u = u / u.norm()
score = (Wr @ u) / Wr.norm(dim=1)
ref = score[reserved].median().item()
special = set(reserved) | {i for i in range(V) if names[i] and names[i].startswith("<|")}
cand = [i for i in torch.argsort(score, descending=True).tolist() if i not in special]
n_close = sum(1 for i in cand if score[i] >= ref)
print(f"ordinary tokens at least as close as the median reserved token: {n_close}")
top = cand[:60]
for i in top[:40]: print(f"  {i:6d} {score[i]:.3f} {tok.decode([i])!r}")
json.dump({"top": top, "scores": [float(score[i]) for i in top], "ref": ref}, open("weirdtok.json", "w"))
