import torch, re, json, numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
W = torch.load("weather.pt", weights_only=True); CH = W["CH"]; rec = W["rec"]
f = safe_open("/home/three/interp/saes/crosscoder/model.safetensors", "pt"); D = f.get_tensor("decoder.weight")[1]
U = torch.load("llama8b_unembed.pt", weights_only=True); Wu, g = U["lm_head.weight"].float(), U["model.norm.weight"].float()
halfcos = torch.load("/home/three/interp/saes/crosscoder/halfcos.pt", weights_only=True)
def lens(j, n=8): return [tok.decode([i]).strip() for i in (Wu @ (D[j] * g)).topk(n).indices.tolist()]
def readable(ws): return sum(1 for w in ws if re.fullmatch(r"[A-Za-z]{3,}", w) is not None)
toks = [[tok.decode([i]) for i in tok(r["text"], add_special_tokens=False).input_ids][1:] for r in rec]
dense = [r["code"].to_dense() for r in rec]
F = len(CH); ntext = torch.zeros(F); tot = torch.zeros(F)
for c in dense: ntext += (c > 0).any(0).float(); tot += c.sum(0)
out = []
for fi in range(F):
    if ntext[fi] < 5: continue
    j = int(CH[fi]); L = lens(j); r_ = readable(L)
    if r_ > 2: continue
    ctx = []
    for ti, c in enumerate(dense):
        v = c[:, fi]
        for p in torch.nonzero(v > 0).squeeze(1).tolist(): ctx.append((float(v[p]), ti, p))
    ctx.sort(reverse=True)
    tags = {}
    for _, ti, _ in ctx: tags[rec[ti]["tag"].split(":")[0]] = tags.get(rec[ti]["tag"].split(":")[0], 0) + 1
    show = []
    for v, ti, p in ctx[:6]:
        T = toks[ti]; left = "".join(T[max(0, p - 7):p]).replace("\n", "⏎")[-45:]; here = T[p].replace("\n", "⏎")
        show.append(f"{v:.1f} [{rec[ti]['tag']}] …{left}⟦{here}⟧")
    out.append({"j": j, "ntext": int(ntext[fi]), "lens": L, "halfcos": round(float(halfcos[j]), 2), "tags": tags, "top": show})
out.sort(key=lambda x: -x["ntext"]); json.dump(out, open("weather_guide.json", "w"), indent=1)
print(len(out), "unreadable chat-grown features firing in ≥5 chats")
for x in out[:40]:
    print(f"\n#{x['j']} · {x['ntext']} chats · halves cos {x['halfcos']:+.2f} · lens {x['lens'][:5]} · where {x['tags']}")
    for s in x["top"]: print("    ", s)
