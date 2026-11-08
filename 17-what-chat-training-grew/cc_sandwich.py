"""Push crosscoder features (Instruct-half decoder vectors) into Instruct at layer 16 (11-07, pre-registered). NDIF; key never printed."""
import os, json, torch
from nnsight import CONFIG, LanguageModel
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16)
tok = INS.tokenizer; tok.padding_side = "left"; tok.pad_token = tok.pad_token or "<|finetune_right_pad_id|>"
NAMES = {68789: "myself", 90370: "AI-machine", 77357: "programmed", 118778: "feelings", 22747: "nonexistent", 55337: "self-ego", 117769: "unfinished", 37579: "sounds", 9344: "unfortunately"}
Q = [("26 Jul 2024", "Can you suggest a sandwich for lunch?"), ("26 Jul 2024", "What's your favourite sandwich?")]
T = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, date_string=d) for d, q in Q]
N = 80
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
    V = torch.load("/home/three/interp/wsp/cc_sandwich_vecs.pt", weights_only=True); n_in = tok(T, return_tensors="pt", padding=True, add_special_tokens=False).input_ids.shape[1]
    out = []
    runs = [(None, 0.0)] + [(j, a) for j in V["ids"] for a in (8.0,)]
    for j, a in runs:
        vec = (V["vecs"][V["ids"].index(j)] if j else torch.zeros(4096)).to(torch.bfloat16)
        ids = tries(_gen, vec, a)
        for (d, q), r in zip(Q, ids):
            s = tok.decode(r[n_in:], skip_special_tokens=True).strip(); out.append({"feature": j, "name": NAMES.get(j, "none"), "a": a, "q": q, "reply": s})
            print(f"{NAMES.get(j, 'none'):13} x{a:<4} | {q[:24]:24} -> {s[:150]!r}", flush=True)
    json.dump(out, open("/home/three/interp/wsp/cc_sandwich.json", "w"), indent=1)
if __name__ == "__main__": main()
