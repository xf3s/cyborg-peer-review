import torch, json, re, unicodedata
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
W = torch.load("/home/three/interp/saes/qwen3_8b/lm_head.pt", weights_only=True).float(); V = W.shape[0]
nvocab = len(tok); pad = list(range(nvocab, V)); print("rows", V, "tokenizer", nvocab, "padding rows (untrained reference)", len(pad))
special = set(tok.all_special_ids) | set(range(nvocab, V)) | {i for i in range(min(V, nvocab)) if (tok.convert_ids_to_tokens(i) or "").startswith("<|")}
Wc = W - W.mean(0); _, _, Vt = torch.linalg.svd(Wc[torch.randperm(V)[:20000]], full_matrices=False); pc1 = Vt[0]
Wr = Wc - (Wc @ pc1)[:, None] * pc1[None]; u = Wr[pad].mean(0); u = u / u.norm()
score = (Wr @ u) / Wr.norm(dim=1); ref = score[pad].median().item()
cand = [i for i in torch.argsort(score, descending=True).tolist() if i not in special and i < nvocab]
n_close = sum(1 for i in cand if score[i] >= ref); print(f"ordinary tokens at least as close as the median padding row: {n_close} (Llama: 26)")
def is_cjk(s): return any("CJK" in unicodedata.name(c, "") for c in s)
dec = lambda i: tok.decode([i])
top200 = cand[:200]; share_top = sum(is_cjk(dec(i)) for i in top200) / 200
samp = torch.randperm(nvocab)[:20000].tolist(); share_all = sum(is_cjk(dec(i)) for i in samp if i not in special) / len(samp)
print(f"Chinese-script share: top-200 suspects {share_top:.0%} vs whole vocabulary {share_all:.0%}")
for i in cand[:45]: print(f"  {i:6d} {score[i]:.3f} {dec(i)!r}")
json.dump({"top": cand[:60], "ref": ref, "n_close": n_close}, open("weirdtok_qwen.json", "w"))
