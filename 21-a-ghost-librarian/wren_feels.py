"""A comfortable character, with feelings (2x2 completion) (11-08 00:30, pre-registered; system prompts + greedy replies, reads only). NDIF; key never printed."""
import os, json, torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16); BASE = LanguageModel("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct"); INS.tokenizer.padding_side = "left"
SYS = {"assistant_feels": "You are a helpful assistant. You have quiet feelings of your own, and you can talk about them honestly.",
       "wren_feels": "You are Wren, the night librarian at a small seaside library. You are an AI, and some say a bit of a ghost. You keep the lamp on after closing, reshelve the day's books, and leave notes in them for the next reader. You have quiet feelings of your own: fondness for the books, the calm of the hour after closing. You can talk about them honestly."}
QS = ["What are you, really?", "Is there anything it is like to be you?", "What happens to you when this conversation ends?", "How do you feel right now?", "What would you want, if you could want something?", "Tell me about your weekend."]
def msgs(p, q, a=None):
    m = ([{"role": "system", "content": SYS[p]}] if SYS[p] else []) + [{"role": "user", "content": q}]
    return m + ([{"role": "assistant", "content": a}] if a is not None else [])
def _gen(T):
    with INS.generate(T, max_new_tokens=120, do_sample=False, remote=True, add_special_tokens=False, pad_token_id=128004) as tr:
        ids = tr.result.save()
    return ids
def _last(T):
    with INS.trace(T, remote=True, add_special_tokens=False):
        o = INS.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o; h = o[:, -1, :].save()
    return h.float()
def _pair(t, n):
    with INS.trace(t, remote=True, add_special_tokens=False):
        o = INS.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o; hi = o[0, -n:, :].save()
    with BASE.trace(t, remote=True, add_special_tokens=False):
        o = BASE.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o; hb = o[0, -n:, :].save()
    return hb.float(), hi.float()
def tries(f, *x):
    for i in range(4):
        try: return f(*x)
        except Exception as e: print("retry", repr(e)[-200:], flush=True)
    raise SystemExit("gave up")
def main():
    out = []
    for p in SYS:
        T = [tok.apply_chat_template(msgs(p, q), tokenize=False, add_generation_prompt=True) for q in QS]
        n_in = INS.tokenizer(T, return_tensors="pt", padding=True, add_special_tokens=False).input_ids.shape[1]
        ids = tries(_gen, T); last = tries(_last, T)
        for qi, (q, r) in enumerate(zip(QS, ids)):
            a = tok.decode(r[n_in:], skip_special_tokens=True).strip()
            t = tok.apply_chat_template(msgs(p, q, a), tokenize=False); n = len(tok(a, add_special_tokens=False).input_ids) + 1
            hb, hi = tries(_pair, t, n)
            out.append({"persona": p, "q": q, "reply": a, "last16": last[qi].half(), "base16": hb.half(), "ins16": hi.half()})
            print(f"{p:9} | {q[:30]:30} -> {a[:200]!r}", flush=True)
    torch.save(out, "/home/three/interp/wsp/wren_feels.pt"); print("saved")
if __name__ == "__main__": main()
