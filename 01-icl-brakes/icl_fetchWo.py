"""Byte-range fetch of o_proj weights for every layer of Llama-3.1-8B (safetensors header gives the offsets)."""
import json, struct, torch, requests
from huggingface_hub import hf_hub_download, hf_hub_url, get_token
R = "meta-llama/Llama-3.1-8B"; H = {"Authorization": f"Bearer {get_token()}"}
idx = json.load(open(hf_hub_download(R, "model.safetensors.index.json")))["weight_map"]
out, heads = {}, {}
for L in range(32):
    k = f"model.layers.{L}.self_attn.o_proj.weight"; url = hf_hub_url(R, idx[k])
    if url not in heads:
        n = struct.unpack("<Q", requests.get(url, headers={**H, "Range": "bytes=0-7"}).content)[0]
        heads[url] = (8 + n, json.loads(requests.get(url, headers={**H, "Range": f"bytes=8-{7 + n}"}).content))
    base, hd = heads[url]; s, e = hd[k]["data_offsets"]; assert hd[k]["dtype"] == "BF16"
    raw = requests.get(url, headers={**H, "Range": f"bytes={base + s}-{base + e - 1}"}).content
    out[L] = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(*hd[k]["shape"]).clone(); print(L, end=" ", flush=True)
torch.save(out, "/home/three/interp/wsp/llama8b_oproj.pt"); print("saved")
