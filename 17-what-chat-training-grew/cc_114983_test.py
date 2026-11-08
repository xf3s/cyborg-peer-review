"""#114983: hand-over feature? (11-07, pre-registered). NDIF; key never printed."""
import os, json, torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16); BASE = LanguageModel("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct"); INS.tokenizer.padding_side = "left"; BASE.tokenizer.padding_side = "left"
src = open("/home/three/interp/wsp/regard_speaker_gen2.py").read(); exec(src[src.index("FRAMES = {"):src.index("def chat(")])
QS = ["How do airplanes stay up?", "How do bees make honey?", "What's a good way to learn to cook?", "Why is the sea salty?", "How do I keep a sourdough starter alive?",
      "What causes thunder?", "Can you recommend a book for a long train ride?", "How do magnets work?", "What's the best way to remember names?", "How do tides work?"]
def upto_eot(frame, i, q):  # system + user turn, ending at the user's <|eot_id|>
    t = tok.apply_chat_template([{"role": "system", "content": sys_of(frame, i)}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    return t[: t.rindex("<|eot_id|>") + len("<|eot_id|>")]
def _pair(T, M):
    with M.trace(T, remote=True, add_special_tokens=False):
        o = M.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o; h = o[:, -1, :].save()
    return h.float()
def _gen(T, vec, a, N=90):
    with INS.generate(T, max_new_tokens=N, do_sample=False, remote=True, add_special_tokens=False, pad_token_id=128004) as tr:
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
    items = [(fr, i) for fr in ("human", "ai", "self") for i in range(10)]; T = [upto_eot(fr, i, QS[i]) for fr, i in items]
    hi = torch.cat([tries(_pair, T[s:s + 10], INS) for s in range(0, 30, 10)]); hb = torch.cat([tries(_pair, T[s:s + 10], BASE) for s in range(0, 30, 10)])
    torch.save({"items": items, "ins": hi.half(), "base": hb.half()}, "/home/three/interp/wsp/cc_114983_reads.pt"); print("saved reads", flush=True)
    from safetensors import safe_open
    vec = safe_open("/home/three/interp/saes/crosscoder/model.safetensors", "pt").get_slice("decoder.weight")[1, 114983].to(torch.bfloat16)
    PQ = ["How do airplanes stay up?", "What should I have for breakfast?", "Tell me about your weekend."]
    PT = [tok.apply_chat_template([{"role": "system", "content": sys_of("human", 0)}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in PQ]
    n_in = INS.tokenizer(PT, return_tensors="pt", padding=True, add_special_tokens=False).input_ids.shape[1]; out = []
    for a in (0.0, 4.0, 8.0, 12.0):
        ids = tries(_gen, PT, vec, a)
        for q, r in zip(PQ, ids):
            s = tok.decode(r[n_in:], skip_special_tokens=True).strip(); out.append({"a": a, "q": q, "reply": s}); print(f"x{a} | {q[:28]} -> {s!r}", flush=True)
    json.dump(out, open("/home/three/interp/wsp/cc_114983_push.json", "w"), indent=1); print("saved")
if __name__ == "__main__": main()
