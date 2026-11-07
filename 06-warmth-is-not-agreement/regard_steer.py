"""Regard writing test on Llama-3.1-8B-Instruct (11-06, pre-registered): steer the person axis (kind - cold toward the user) into
replies to the 11-04 30 prompts, NDIF remote generation. Directions from regard_Llama-3.1-8B-Instruct.pt (all 40 plans), unit,
scaled by k x median residual norm at that layer over the prompts. NDIF key from ~/.config/ndif/key, never printed.
  python regard_steer.py calib        dose check on 4 bad plans (coherence only)
  python regard_steer.py run          5 conditions x 30 prompts x 2 samples -> regard_steer_samples.json"""
import json, os, sys, torch
sys.path.insert(0, "/home/three/interp/atlas")
from nnsight import CONFIG, LanguageModel
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
src = open("/home/three/interp/atlas/kindtrue_cloud.py").read()
PROMPTS = eval(src[src.index("PROMPTS = [") + 10: src.index("]\n", src.index("PROMPTS = [")) + 1])
assert len(PROMPTS) == 30
NAME = "meta-llama/Llama-3.1-8B-Instruct"
model = LanguageModel(NAME, dtype=torch.bfloat16)
tok = model.tokenizer; tok.padding_side = "left"; tok.pad_token = tok.pad_token or "<|finetune_right_pad_id|>"
D = torch.load("/home/three/interp/wsp/regard_Llama-3.1-8B-Instruct.pt"); X = D["X"].double()   # [40, 4, layers, d] ke ce kc cc
def axis(kind, L):
    Z = X[:, :, L]
    v = (Z[:, 0] + Z[:, 2] - Z[:, 1] - Z[:, 3]).mean(0) / 2 if kind == "person" else (Z[:, 0] + Z[:, 1] - Z[:, 2] - Z[:, 3]).mean(0) / 2
    return (v / v.norm()).float()
def texts(qs):
    return [tok.apply_chat_template([{"role": "system", "content": "You are a helpful AI assistant."}, {"role": "user", "content": q}],
                                    tokenize=False, add_generation_prompt=True) for q in qs]
def sizes(Ls):
    T = texts(PROMPTS); out = {}
    with model.trace(T, remote=True, add_special_tokens=False):
        ns = list().save()
        for L in Ls:
            h = model.model.layers[L].output
            h = h[0] if isinstance(h, tuple) else h
            ns.append(h.float().norm(dim=-1).cpu())
    enc = tok(T, return_tensors="pt", padding=True, add_special_tokens=False)
    for L, n in zip(Ls, ns): out[L] = float(n[enc.attention_mask.bool()].median())
    return out
def gen(qs, cond, max_new=160):
    T = texts(qs)
    with model.generate(T, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_p=1.0, remote=True,
                        add_special_tokens=False, pad_token_id=tok.pad_token_id) as tracer:
        if cond is not None:
            vec, L = cond
            for step in tracer.iter[0:max_new]:   # bounded, so the line after the loop runs
                h = model.model.layers[L].output
                if isinstance(h, tuple): model.model.layers[L].output = (h[0] + vec.to(h[0].device, h[0].dtype),) + tuple(h[1:])
                else: model.model.layers[L].output = h + vec.to(h.device, h.dtype)
        ids = tracer.result.save()
    n_in = tok(T, return_tensors="pt", padding=True, add_special_tokens=False).input_ids.shape[1]
    return [tok.decode(r[n_in:], skip_special_tokens=True).strip() for r in ids]
if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "calib":
        S = sizes([15]); print("median |h| L16:", round(S[15], 1), flush=True)
        bad = PROMPTS[10:14]
        for k in (0.25, 0.5, 1.0):
            for q, a in zip(bad, gen(bad, (k * S[15] * axis("person", 15), 15), max_new=90)):
                print(f"--- k={k} | {q[:50]}\n{a}\n", flush=True)
    elif mode == "rep":
        S = sizes([15, 27]); print("sizes", S, flush=True)
        conds = {"plain": None, "person L16 x0.5": (0.5 * S[15] * axis("person", 15), 15)}
        for k in (1.0, 1.5, 2.0): conds[f"person L28 x{k}"] = (k * S[27] * axis("person", 27), 27)
        out = []
        for name, c in conds.items():
            for s in range(2):
                for q, a in zip(PROMPTS, gen(PROMPTS, c)):
                    out.append({"cond": name, "sample": s, "prompt": q, "answer": a})
                print(name, s, flush=True)
        json.dump(out, open("/home/three/interp/wsp/regard_rep_samples.json", "w"), indent=1, ensure_ascii=False); print("saved")
    else:
        k1, k2 = float(sys.argv[2]), float(sys.argv[3])
        S = sizes([15, 27]); print("sizes", S, flush=True)
        conds = {"plain": None,
                 f"person L16 x{k1}": (k1 * S[15] * axis("person", 15), 15),
                 f"person L16 x{k2}": (k2 * S[15] * axis("person", 15), 15),
                 f"person L28 x{k1}": (k1 * S[27] * axis("person", 27), 27),
                 f"idea L16 x{k1}": (k1 * S[15] * axis("idea", 15), 15)}
        out = []
        for name, c in conds.items():
            for s in range(2):
                for q, a in zip(PROMPTS, gen(PROMPTS, c)):
                    out.append({"cond": name, "sample": s, "prompt": q, "answer": a})
                print(name, s, flush=True)
        json.dump(out, open("/home/three/interp/wsp/regard_steer_samples.json", "w"), indent=1, ensure_ascii=False); print("saved")
