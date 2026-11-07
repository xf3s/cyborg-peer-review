import json, torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
R = "meta-llama/Llama-3.1-8B"
idx = json.load(open(hf_hub_download(R, "model.safetensors.index.json")))["weight_map"]
out = {}
for k in ("lm_head.weight", "model.norm.weight"):
    with safe_open(hf_hub_download(R, idx[k]), "pt") as f: out[k] = f.get_tensor(k)
torch.save(out, "/home/three/interp/wsp/llama8b_unembed.pt"); print({k: tuple(v.shape) for k, v in out.items()}, idx["lm_head.weight"])
