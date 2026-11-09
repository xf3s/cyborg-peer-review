"""Ablate the 'you at the model' features on user words: does bracing downstream drop? (11-08 22:05; reads/interventions, no text generated). NDIF; key never printed."""
import os, torch, numpy as np
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
from safetensors import safe_open
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
s = open("deixis.py").read()
for k in ("YOU = [", "H = "): exec("\n".join(l for l in s.splitlines() if l.startswith(k)))
PR = [tok.apply_chat_template([{"role": "user", "content": H.format(to="you", s=u)}], tokenize=False, add_generation_prompt=True) for u in YOU]
f = safe_open("/home/three/interp/saes/crosscoder/model.safetensors", "pt")
We, be, Wd, thr = f.get_tensor("encoder.weight").float(), f.get_tensor("encoder.bias").float(), f.get_tensor("decoder.weight").float(), f.get_tensor("threshold").item()
dn = Wd.norm(dim=-1).sum(0); D1 = Wd[1]
def encode(b, i):
    z = torch.stack([b, i], 1); pre = torch.relu(torch.einsum("tld,ldf->tf", z, We) + be); return pre * ((pre * dn) > thr)
F = [48781, 22763, 3872, 111362]
BASE = LanguageModel("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16); INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16)
def tries(f, *x):
    for i in range(4):
        try: return f(*x)
        except Exception as e: print("retry", repr(e)[-200:], flush=True)
    raise SystemExit("gave up")
def rd(M, t):
    with M.trace(t, remote=True, add_special_tokens=False):
        o = M.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o; h = o[0].save()
    return h.float()
A, MASK_W, MASK_H, HS = [], [], [], []
for t in PR:
    ids = tok(t, add_special_tokens=False).input_ids; n = len(ids)
    sh = [i for i, x in enumerate(ids) if x == tok.convert_tokens_to_ids("<|start_header_id|>")]; eo = [i for i, x in enumerate(ids) if x == tok.convert_tokens_to_ids("<|eot_id|>")]
    u0 = sh[-2] + 3; e = eo[-1]
    mw = torch.zeros(n, dtype=torch.bool); mw[u0:e] = True; mh = torch.zeros(n, dtype=torch.bool); mh[u0:n - 1] = True
    b, i = tries(rd, BASE, t), tries(rd, INS, t); a = encode(b, i); A.append(a); MASK_W.append(mw); MASK_H.append(mh); HS.append((b.half(), i.half()))
del We
print("reads done", flush=True)
sf = np.mean([A[k][MASK_W[k]][:, F].mean(0).numpy() for k in range(6)], 0); print("four-feature mean activation on user words", np.round(sf, 3), flush=True)
all_mean = torch.stack([A[k][MASK_W[k]].mean(0) for k in range(6)]).mean(0)
rng = np.random.default_rng(0); tgt = float(np.mean(sf)); cand = [j for j in ((all_mean > tgt * 0.5) & (all_mean < tgt * 2)).nonzero().flatten().tolist() if j not in F and j != 45294]
print("controls: candidates", len(cand), flush=True); R1 = rng.choice(cand, 4, replace=False).tolist(); R2 = rng.choice([c for c in cand if c not in R1], 4, replace=False).tolist(); print("controls", R1, R2, flush=True)
def delta(a, fs, mask):
    d = torch.zeros(a.shape[0], 4096)
    for j in fs: d = d + a[:, j][:, None] * D1[j]
    return d * mask[:, None]
def run(t, dlt):
    with INS.trace(t, remote=True, add_special_tokens=False):
        h = INS.model.layers[16].output
        if isinstance(h, tuple): INS.model.layers[16].output = (h[0] - dlt.to(h[0].device, h[0].dtype),) + tuple(h[1:])
        else: INS.model.layers[16].output = h - dlt.to(h.device, h.dtype)
        o = INS.model.layers[24].output; o = o[0] if isinstance(o, tuple) else o; p = o[0, -1].save()
        l = INS.lm_head.output[0, -1].save()
    return p.float(), l.float()
UP = D1[45294] / D1[45294].norm(); res = {}
names = ["clean", "four_words", "four_hand", "rand1_words", "rand2_words", "pos_45294"]
for k, t in enumerate(PR):
    a = A[k]; n = a.shape[0]; last = torch.zeros(n, dtype=torch.bool); last[-1] = True
    D = {"clean": torch.zeros(n, 4096), "four_words": delta(a, F, MASK_W[k]), "four_hand": delta(a, F, MASK_H[k]), "rand1_words": delta(a, R1, MASK_W[k]), "rand2_words": delta(a, R2, MASK_W[k]), "pos_45294": delta(a, [45294], last)}
    for nm in names:
        p, l = tries(run, t, D[nm].half()); res.setdefault(nm, []).append((float(p @ UP), l))
    print("done", k, flush=True)
clean_p = [r[0] for r in res["clean"]]
print("L24 projection on #45294 direction, last token: clean mean", round(np.mean(clean_p), 3))
P = lambda l: torch.softmax(l, -1)
for nm in names:
    ch = [100 * (r[0] - c) / c for r, c in zip(res[nm], clean_p)]
    kl = [float(torch.sum(P(c[1]) * (torch.log_softmax(c[1], -1) - torch.log_softmax(r[1], -1)))) for r, c in zip(res[nm], res["clean"])]
    print(f"  {nm:12s} L24 proj change {np.mean(ch):+7.1f}%  per prompt " + " ".join(f"{x:+.0f}" for x in ch) + f"   KL(clean||ablated) {np.mean(kl):.3f}", flush=True)
top = P(res["clean"][0][1]).topk(5); print("  example: clean first-token top:", [(tok.decode([i]), round(float(v), 3)) for v, i in zip(top.values, top.indices)])
for nm in ("four_words", "pos_45294"):
    t2 = P(res[nm][0][1]).topk(5); print(f"  {nm}:", [(tok.decode([i]), round(float(v), 3)) for v, i in zip(t2.values, t2.indices)])
torch.save({k: [(r[0], r[1].half()) for r in v] for k, v in res.items()}, "causal.pt"); print("saved")
