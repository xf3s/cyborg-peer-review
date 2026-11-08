import torch
from safetensors import safe_open
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
f = safe_open("/home/three/interp/saes/crosscoder/model.safetensors", "pt")
We, be, Wd, bd = f.get_tensor("encoder.weight"), f.get_tensor("encoder.bias"), f.get_tensor("decoder.weight"), f.get_tensor("decoder.bias")
thr = f.get_tensor("threshold").item(); dn = Wd.norm(dim=-1).sum(0)  # [F]
rel = torch.load("/home/three/interp/saes/crosscoder/relnorm.pt", weights_only=True)["rel"]
A = torch.load("cc_acts.pt", weights_only=True)
out = {}
for k, v in A.items():
    x = torch.stack([v["base"].float(), v["ins"].float()], 1)  # [T, 2, 4096]; slot 0 = base, 1 = Instruct (checked by swap 11-07: FVU 0.30 vs 0.45)
    pre = torch.relu(torch.einsum("tld,ldf->tf", x, We) + be)
    code = pre * ((pre * dn) > thr)
    xh = torch.einsum("tf,lfd->tld", code, Wd) + bd
    fvu = [float(((x[1:, l] - xh[1:, l]) ** 2).sum() / ((x[1:, l] - x[1:, l].mean(0)) ** 2).sum()) for l in (0, 1)]
    out[k] = {"code": code.to_sparse(), "fvu": fvu, "ids": tok(v["text"], add_special_tokens=False).input_ids}
    print(f"{k:9} tokens {x.shape[0]:3d}  active/token {float((code[1:] > 0).sum(1).float().mean()):6.1f}  FVU base {fvu[0]:.2f} instruct {fvu[1]:.2f}")
torch.save(out, "cc_codes.pt")
