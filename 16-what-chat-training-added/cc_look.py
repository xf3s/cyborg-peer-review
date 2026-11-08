import torch, sys
from safetensors import safe_open
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
C = torch.load("cc_codes.pt", weights_only=True)
rel = torch.load("/home/three/interp/saes/crosscoder/relnorm.pt", weights_only=True)["rel"]; ins_share = 1 - rel
f = safe_open("/home/three/interp/saes/crosscoder/model.safetensors", "pt"); Wd = f.get_tensor("decoder.weight")[1]  # Instruct decoder [F, d]
U = torch.load(sys.argv[1], weights_only=True); W, g = U["lm_head.weight"].float(), U["model.norm.weight"].float()
def lens(j, n=6): return [tok.decode([i]).strip() or repr(tok.decode([i])) for i in (W @ (Wd[j] * g)).topk(n).indices.tolist()]
def toks(k): return [tok.decode([i]) for i in C[k]["ids"]]
chat = ins_share >= 0.7
def show(k, pos_filter=None, top=8, label=""):
    code = C[k]["code"].to_dense(); T = toks(k)
    print(f"\n== {k} {label}")
    for p in range(1, len(T)):
        if pos_filter and not pos_filter(T, p): continue
        a = code[p] * chat
        if a.sum() == 0: continue
        idx = a.topk(3).indices.tolist()
        print(f"  {p:3d} {T[p]!r:16} " + " | ".join(f"#{j} {a[j]:.1f} ({ins_share[j]:.2f}) {lens(j,4)}" for j in idx if a[j] > 0))
def diff(k1, k2, sl, top=10):
    a = C[k1]["code"].to_dense()[sl].sum(0); b = C[k2]["code"].to_dense()[sl].sum(0); d = (a - b) * chat
    print(f"\n== chat-leaning features more active in {k1} than {k2} over positions {sl}:")
    for j in d.topk(top).indices.tolist():
        if d[j] > 0: print(f"  #{j} +{d[j]:.1f} (share {ins_share[j]:.2f}) lens {lens(j)}")
    print(f"  ...and more active in {k2}:")
    for j in (-d).topk(top).indices.tolist():
        if d[j] < 0: print(f"  #{j} {d[j]:.1f} (share {ins_share[j]:.2f}) lens {lens(j)}")
print("chat-leaning features (Instruct share >= 0.7):", int(chat.sum()))
