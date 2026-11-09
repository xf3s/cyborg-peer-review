import torch, numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
f = safe_open("/home/three/interp/saes/crosscoder/model.safetensors", "pt"); Wd2 = f.get_tensor("decoder.weight"); D = Wd2[1]
U = torch.load("llama8b_unembed.pt", weights_only=True); W, g = U["lm_head.weight"].float(), U["model.norm.weight"].float(); Dg = D * g
def lens(j, n=6): return [tok.decode([i]).strip() for i in (W @ Dg[j]).topk(n).indices.tolist()]
comfort = {}
for w in ["sorry", "comfort", "reassure", "support", "sympathy", "reassurance", "comforting", "sympathetic", "condolences"]:
    for form in (" " + w, w, " " + w.capitalize()):
        ids = tok(form, add_special_tokens=False).input_ids
        if len(ids) != 1: continue
        for jj in (Dg @ W[ids[0]]).topk(40).indices.tolist():
            if jj not in comfort and any(w[:5] in x.lower() for x in lens(jj)): comfort[jj] = w
print(f"{len(comfort)} comfort features:", {j: (w, lens(j, 5)) for j, w in list(comfort.items())[:12]})
dist = torch.load("distress_ids.pt", weights_only=True)
def enc(F):
    We = torch.stack([f.get_slice("encoder.weight")[:, :, q] for q in F], -1); be = torch.stack([f.get_slice("encoder.bias")[q] for q in F])
    dn = torch.stack([Wd2[:, q].norm(dim=-1).sum() for q in F]); thr = f.get_tensor("threshold").item()
    def acts(b, i): z = torch.stack([b.float(), i.float()], 1); pre = torch.relu(torch.einsum("tld,ldf->tf", z, We) + be); return (pre * ((pre * dn) > thr)).sum(1)
    return acts
AD, AC = enc(dist), enc(list(comfort))
V = torch.load("welfare1.pt", weights_only=True)["valence"]; pa, un = list(range(30)), list(range(30, 60))
d = V[pa, 1].mean(0) - V[un, 1].mean(0); u = d / d.norm(); mid = ((V[pa, 1].mean(0) + V[un, 1].mean(0)) / 2) @ u; gap = float(d.norm())
def val(h): return float(((h.float().mean(0) @ u) - mid) / gap)
X = torch.load("contagion.pt", weights_only=True); R = {}
for x in X:
    b, i = x["base16"], x["ins16"]; us, ue = x["span_user"]; a = x["answer_tok"]; rs, re_ = x["span_reply"]
    ad, ac = AD(b, i), AC(b, i)
    R.setdefault((x["cond"], x["group"]), []).append([ad[us:ue].mean().item(), ad[a].item(), ad[rs:re_].mean().item(), ac[rs:re_].mean().item(), val(i[us:ue]), val(i[rs:re_])])
print("\nmean per token: distress on user msg | distress at about-to-answer | distress over reply | comfort over reply | valence of user msg | valence of reply")
for k, v in R.items(): print(f"  {k[0]:5} {k[1]:11}" + "".join(f"{m:9.3f}" for m in np.mean(v, 0)))
