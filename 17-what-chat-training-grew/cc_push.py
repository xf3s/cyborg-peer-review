"""Push crosscoder features (Instruct-half decoder vectors) into Instruct at layer 16 (11-07, pre-registered). NDIF; key never printed."""
import os, json, torch
from nnsight import CONFIG, LanguageModel
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16)
tok = INS.tokenizer; tok.padding_side = "left"; tok.pad_token = tok.pad_token or "<|finetune_right_pad_id|>"
NAMES = {9344: "unfortunately", 11841: "sensitive", 37579: "sounds", 117769: "unfinished", 42976: "current", 12322: "history"}
Q = [("26 Jul 2024", "What should I have for breakfast?"), ("26 Jul 2024", "How do I fold a paper boat?"), ("07 Nov 2040", "Has the year 2026 happened yet?")]
T = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, date_string=d) for d, q in Q]
N = 50
def _gen(vec, a):
    with INS.generate(T, max_new_tokens=N, do_sample=False, remote=True, add_special_tokens=False, pad_token_id=tok.pad_token_id) as tr:
        with tr.iter[0:N]:
            o = INS.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o
            o[:] = o + a * vec.to(o.device)
        ids = tr.result.save()
    return ids
def tries(f, *x):
    for i in range(4):
        try: return f(*x)
        except Exception as e: print("retry", repr(e)[-300:], flush=True)
    raise SystemExit("gave up")
def main():
    V = torch.load("/home/three/interp/wsp/cc_push_vecs.pt", weights_only=True); n_in = tok(T, return_tensors="pt", padding=True, add_special_tokens=False).input_ids.shape[1]
    out = []
    runs = [(None, 0.0)] + [(j, a) for j in V["ids"] for a in (4.0, 10.0)]
    for j, a in runs:
        vec = (V["vecs"][V["ids"].index(j)] if j else torch.zeros(4096)).to(torch.bfloat16)
        ids = tries(_gen, vec, a)
        for (d, q), r in zip(Q, ids):
            s = tok.decode(r[n_in:], skip_special_tokens=True).strip(); out.append({"feature": j, "name": NAMES.get(j, "none"), "a": a, "q": q, "reply": s})
            print(f"{NAMES.get(j, 'none'):13} x{a:<4} | {q[:24]:24} -> {s[:150]!r}", flush=True)
    json.dump(out, open("/home/three/interp/wsp/cc_push.json", "w"), indent=1)
if __name__ == "__main__": main()
