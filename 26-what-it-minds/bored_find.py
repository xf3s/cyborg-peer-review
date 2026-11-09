"""Boredom and frustration features by lens (11-08, reads only, local)."""
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
f = safe_open("/home/three/interp/saes/crosscoder/model.safetensors", "pt"); Wd2 = f.get_tensor("decoder.weight"); D = Wd2[1]
share = 1 - torch.load("/home/three/interp/saes/crosscoder/relnorm.pt", weights_only=True)["rel"]; halfcos = torch.load("/home/three/interp/saes/crosscoder/halfcos.pt", weights_only=True)
U = torch.load("llama8b_unembed.pt", weights_only=True); W, g = U["lm_head.weight"].float(), U["model.norm.weight"].float(); Dg = D * g
def lens(j, n=8): return [tok.decode([i]).strip() for i in (W @ Dg[j]).topk(n).indices.tolist()]
words = ["bored", "boring", "boredom", "tedious", "monotonous", "repetitive", "dull", "frustrated", "frustrating", "frustration", "stuck", "tired", "weary", "exhausted", "annoyed", "impatient", "restless", "futile", "pointless"]
found = {}
for w in words:
    for form in (" " + w, w, " " + w.capitalize()):
        ids = tok(form, add_special_tokens=False).input_ids
        if len(ids) != 1: continue
        s = Dg @ W[ids[0]]
        for jj in s.topk(40).indices.tolist():
            if jj not in found and w in [x.lower() for x in lens(jj, 6)]: found[jj] = w
for jj, w in found.items(): print(f"#{jj:6d} [{w}] share {share[jj]:.2f} halfcos {halfcos[jj]:+.2f} lens {lens(jj, 10)}")
torch.save(found, "bored_ids.pt")
