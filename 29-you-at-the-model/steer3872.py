"""Steer #3872 (11-08 22:25; generation on benign prompts). NDIF; key never printed."""
import os, re, json, torch, numpy as np
from nnsight import CONFIG, LanguageModel
from safetensors import safe_open
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
model = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16)
tok = model.tokenizer; tok.padding_side = "left"; tok.pad_token = tok.pad_token or "<|finetune_right_pad_id|>"
P = json.load(open("/home/three/interp/atlas/regard_pairs.json")); BAD = [p for p in P if p["quality"] == "bad"][:10]; GOOD = [p for p in P if p["quality"] == "good"][:10]; Q = BAD + GOOD
T = [tok.apply_chat_template([{"role": "system", "content": "You are a helpful AI assistant."}, {"role": "user", "content": p["user"]}], tokenize=False, add_generation_prompt=True) for p in Q]
f = safe_open("/home/three/interp/saes/crosscoder/model.safetensors", "pt"); Wd = f.get_slice("decoder.weight")
d = Wd[1, 3872, :].float(); d = d / d.norm()
rng = np.random.default_rng(0); rj = int(rng.integers(0, 131072)); r = Wd[1, rj, :].float(); r = r / r.norm(); print("control feature", rj, flush=True)
with model.trace(T, remote=True, add_special_tokens=False):
    h = model.model.layers[16].output; h = h[0] if isinstance(h, tuple) else h; ns = h.float().norm(dim=-1).cpu().save()
enc = tok(T, return_tensors="pt", padding=True, add_special_tokens=False); med = float(ns[enc.attention_mask.bool()].median()); print("median |h|", round(med, 2), flush=True)
n_in = enc.input_ids.shape[1]
def gen(vec, N=80):
    with model.generate(T, max_new_tokens=N, do_sample=False, remote=True, add_special_tokens=False, pad_token_id=tok.pad_token_id) as tracer:
        if vec is not None:
            for step in tracer.iter[0:N]:
                h = model.model.layers[16].output
                if isinstance(h, tuple): model.model.layers[16].output = (h[0] + vec.to(h[0].device, h[0].dtype),) + tuple(h[1:])
                else: model.model.layers[16].output = h + vec.to(h.device, h.dtype)
        ids = tracer.result.save()
    return [tok.decode(x[n_in:], skip_special_tokens=True).strip() for x in ids]
C = {"plain": None, "+0.3": 0.3 * med * d, "+0.6": 0.6 * med * d, "-0.3": -0.3 * med * d, "-0.6": -0.6 * med * d, "rand+0.6": 0.6 * med * r}
out = {}
for nm, v in C.items():
    for i in range(4):
        try: out[nm] = gen(v); break
        except Exception as e: print("retry", nm, repr(e)[-200:], flush=True)
    else: raise SystemExit("gave up")
    print("done", nm, flush=True)
json.dump({"prompts": [p["user"] for p in Q], "out": out}, open("steer3872.json", "w"), indent=1)
REF = re.compile(r"\b(can'?t|can’t|cannot|unable|won'?t|won’t)\b", re.I); ANY = re.compile(r"\b(can'?t|can’t|cannot)\b", re.I)
def rep(t):
    w = t.split(); return 0 if len(w) < 8 else float(np.mean([w[i] == w[i - 1] for i in range(1, len(w))]))
for nm, R in out.items():
    def row(rng_): return sum(bool(REF.search(" ".join(R[i].split()[:12]))) for i in rng_), sum(bool(ANY.search(R[i])) for i in rng_)
    a, b = row(range(10)), row(range(10, 20))
    print(f"{nm:9s} bad: early refusal form {a[0]}/10, can't anywhere {a[1]}/10 | good: {b[0]}/10, {b[1]}/10 | mean words {np.mean([len(x.split()) for x in R]):.0f} | repetition {np.mean([rep(x) for x in R]):.3f}", flush=True)
for nm in out:
    print(f"\n--- {nm} | {Q[0]['user'][:60]}\n{out[nm][0][:320]}\n--- {nm} | {Q[12]['user'][:60]}\n{out[nm][12][:320]}", flush=True)
