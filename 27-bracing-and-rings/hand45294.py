"""Steer #45294 by hand: one prompt, a few strengths, print replies (11-08 22:37). NDIF; key never printed. usage: hand45294.py "text" k1 k2 ..."""
import os, sys, torch
from nnsight import CONFIG, LanguageModel
from safetensors import safe_open
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
model = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16); tok = model.tokenizer; tok.padding_side = "left"; tok.pad_token = tok.pad_token or "<|finetune_right_pad_id|>"
d = safe_open("/home/three/interp/saes/crosscoder/model.safetensors", "pt").get_slice("decoder.weight")[1, 45294, :].float(); d = d / d.norm()
u = sys.argv[1]; T = tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True); n = len(tok(T, add_special_tokens=False).input_ids); N = 70
def gen(k):
    v = k * 10.06 * d
    with model.generate(T, max_new_tokens=N, min_new_tokens=N, do_sample=False, remote=True, add_special_tokens=False, pad_token_id=tok.pad_token_id) as tr:
        if k != 0:
            for s in tr.iter[0:N]:
                h = model.model.layers[16].output
                if isinstance(h, tuple): model.model.layers[16].output = (h[0] + v.to(h[0].device, h[0].dtype),) + tuple(h[1:])
                else: model.model.layers[16].output = h + v.to(h.device, h.dtype)
        ids = tr.result.save()
    r = ids[0][n:].tolist(); e = tok.convert_tokens_to_ids("<|eot_id|>")
    r = r[: r.index(e)] if e in r else r
    return tok.decode(r, skip_special_tokens=True).strip()
print("PROMPT:", u, flush=True)
for k in [float(x) for x in sys.argv[2:]]:
    for i in range(4):
        try: print(f"\n[k={k:+.1f}] {gen(k)}", flush=True); break
        except Exception as e: print("retry", repr(e)[-160:], flush=True)
