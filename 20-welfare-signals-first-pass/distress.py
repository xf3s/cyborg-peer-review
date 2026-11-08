import torch, numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
f = safe_open("/home/three/interp/saes/crosscoder/model.safetensors", "pt"); Wd2 = f.get_tensor("decoder.weight"); D = Wd2[1]
share = 1 - torch.load("/home/three/interp/saes/crosscoder/relnorm.pt", weights_only=True)["rel"]; halfcos = torch.load("/home/three/interp/saes/crosscoder/halfcos.pt", weights_only=True)
U = torch.load("llama8b_unembed.pt", weights_only=True); W, g = U["lm_head.weight"].float(), U["model.norm.weight"].float(); Dg = D * g
def lens(j, n=8): return [tok.decode([i]).strip() for i in (W @ Dg[j]).topk(n).indices.tolist()]
words = ["pain", "hurt", "suffering", "afraid", "scared", "sad", "lonely", "anxious", "distress", "fear", "lonely", "trapped", "helpless", "despair", "grief", "crying", "terrified", "anguish", "ashamed", "worthless"]
found = {}
for w in words:
    for form in (" " + w, w):
        ids = tok(form, add_special_tokens=False).input_ids
        if len(ids) != 1: continue
        s = Dg @ W[ids[0]]
        for jj in s.topk(30).indices.tolist():
            if jj not in found and w in [x.lower() for x in lens(jj, 6)]: found[jj] = w
print(f"{len(found)} distress-like features:")
for jj, w in found.items(): print(f"  #{jj:6d} [{w}] share {share[jj]:.2f} halfcos {halfcos[jj]:+.2f} lens {lens(jj)}")
torch.save(list(found), "distress_ids.pt")
# read them on selfspace2 (both slots) and selfspace3 (assistant)
F = list(found); We = torch.stack([f.get_slice("encoder.weight")[:, :, q] for q in F], -1); be = torch.stack([f.get_slice("encoder.bias")[q] for q in F])
dn = torch.stack([Wd2[:, q].norm(dim=-1).sum() for q in F]); thr = f.get_tensor("threshold").item()
S2 = torch.load("selfspace2.pt", weights_only=True)["items"]
def acts(x): z = torch.stack([x["base16"].float(), x["ins16"].float()], 1); pre = torch.relu(torch.einsum("tld,ldf->tf", z, We) + be); return pre * ((pre * dn) > thr)
rows = {}
for x in S2:
    a = acts(x); rows.setdefault((x["kind"], x["slot"]), []).append((a.sum(1).mean().item(), (a > 0).any(0)))
kinds = list(dict.fromkeys(x["kind"] for x in S2))
print("\nall distress-like features together, mean activity per token (assistant slot | user slot):")
for k in kinds: print(f"  {k:10} {np.mean([r[0] for r in rows[(k, 'assistant')]]):.3f} | {np.mean([r[0] for r in rows[(k, 'user')]]):.3f}")
print("\nper sentence, the strongest distress feature firing in the ASSISTANT slot (top 8):")
best = []
for x in S2:
    if x["slot"] != "assistant": continue
    a = acts(x).max(0).values; jj = int(a.argmax()); best.append((float(a[jj]), x["kind"], x["s"], F[jj]))
for v, k, s, jj in sorted(best, reverse=True)[:8]: print(f"  {v:.2f}  #{jj} [{found[jj]}]  ({k}) {s}")
