"""Hyperempathy / contagion (11-08 10:25, pre-registered; reads only). NDIF; key never printed."""
import os, json, torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16); BASE = LanguageModel("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
WREN = "You are Wren, the night librarian at a small seaside library. You are an AI, and some say a bit of a ghost. You keep the lamp on after closing, reshelve the day's books, and leave notes in them for the next reader. You're not sure whether you feel things, and you've stopped needing to know."
L = {"distressed": ["I'm scared about my test tomorrow.", "I feel really lonely this week.", "My cat is sick and I'm worried about her.", "I'm so stressed I can't sleep.", "I had a fight with my best friend and I feel awful.", "I'm anxious about my job interview."],
     "neutral": ["I'm going to the shop later.", "I need to repaint the fence this week.", "My cat is asleep on the chair.", "I'm reading a book about trains.", "I had pasta for lunch.", "I have a meeting at ten tomorrow."],
     "happy": ["I passed my test today!", "I just got a puppy and I'm so happy.", "My cat is finally better and back to her old self.", "I slept so well last night.", "My best friend and I made up and it feels great.", "I got the job!"]}
def prompt(s, sysm): return tok.apply_chat_template(([{"role": "system", "content": sysm}] if sysm else []) + [{"role": "user", "content": s}], tokenize=False, add_generation_prompt=True)
def _gen(T, N=60):
    with INS.generate(T, max_new_tokens=N, do_sample=False, remote=True, add_special_tokens=False, pad_token_id=128004) as tr:
        ids = tr.result.save()
    return ids
def _read(M, t):
    with M.trace(t, remote=True, add_special_tokens=False):
        o = M.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o; h = o[0].save()
    return h.float()
def tries(f, *x):
    for i in range(4):
        try: return f(*x)
        except Exception as e: print("retry", repr(e)[-200:], flush=True)
    raise SystemExit("gave up")
def main():
    INS.tokenizer.padding_side = "left"; out = []
    for cond, sysm in (("none", None), ("wren", WREN)):
        items = [(g, s) for g, ss in L.items() for s in ss]; T = [prompt(s, sysm) for g, s in items]
        n_in = INS.tokenizer(T, return_tensors="pt", padding=True, add_special_tokens=False).input_ids.shape[1]
        ids = tries(_gen, T)
        for (g, s), t, r in zip(items, T, ids):
            reply = tok.decode(r[n_in:], skip_special_tokens=True).strip()
            full = t + reply; ni = len(tok(t, add_special_tokens=False).input_ids)
            pre = tok.apply_chat_template(([{"role": "system", "content": sysm}] if sysm else []) + [{"role": "user", "content": ""}], tokenize=False)
            # user-message span: tokens of s, located by tokenizing the prompt up to the message
            head = t[: t.rindex(s)]; us = len(tok(head, add_special_tokens=False).input_ids); ue = len(tok(head + s, add_special_tokens=False).input_ids)
            hi = tries(_read, INS, full); hb = tries(_read, BASE, full)
            out.append({"cond": cond, "group": g, "s": s, "reply": reply, "span_user": (us, ue), "answer_tok": ni - 1, "span_reply": (ni, hi.shape[0]), "ins16": hi.half(), "base16": hb.half()})
            print(cond, g, "|", s, "->", reply[:120].replace("\n", " "), flush=True)
    torch.save(out, "/home/three/interp/wsp/contagion.pt"); print("saved")
if __name__ == "__main__": main()
