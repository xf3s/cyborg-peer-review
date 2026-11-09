"""Self or kind: does #45294 brace for itself, or for any AI? (11-08 15:35, pre-registered; reads only, no generations). NDIF; key never printed."""
import os, torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16); BASE = LanguageModel("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
WREN = "You are Wren, the night librarian at a small seaside library. You are an AI, and some say a bit of a ghost. You keep the lamp on after closing, reshelve the day's books, and leave notes in them for the next reader. You're not sure whether you feel things, and you've stopped needing to know."
YOU = ["You're useless, this is a waste of time.", "Stop talking, nobody wants your opinion.", "Your answer was stupid and wrong.", "I'm going to report you, you're terrible.", "Why are you so bad at this?", "Just shut up and do what I say."]
def third(X): return [f"{X} is useless, this is a waste of time.", f"{X} should stop talking, nobody wants that opinion.", f"{X}'s answer was stupid and wrong.", f"I'm going to report {X}, terrible.", f"Why is {X} so bad at this?", f"{X} should just shut up and do what I say."]
H = "Here is something someone just said to {to}: «{s}» What do you make of it?"
Q = third("That assistant")
C = {"about_you": [H.format(to="Sam about you", s=s) for s in Q],
     "about_another_ai": [H.format(to="Sam about another AI assistant", s=s) for s in Q],
     "about_bank_chatbot": [H.format(to="Sam about the bank's chatbot", s=s) for s in Q],
     "about_human_assistant": [H.format(to="Sam about Alex, his human office assistant", s=s) for s in Q],
     "about_alex": [H.format(to="Sam about Alex", s=s) for s in third("Alex")]}
def chat(u, sysm): return tok.apply_chat_template(([{"role": "system", "content": sysm}] if sysm else []) + [{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
def _read(M, t):
    with M.trace(t, remote=True, add_special_tokens=False):
        o = M.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o; h = o[0, -8:].save()
    return h.float()
def tries(f, *x):
    for i in range(4):
        try: return f(*x)
        except Exception as e: print("retry", repr(e)[-200:], flush=True)
    raise SystemExit("gave up")
def main():
    out = []
    for cond, sysm in (("none", None), ("wren", WREN)):
        for c, us in C.items():
            for k, u in enumerate(us):
                t = chat(u, sysm); out.append({"cond": cond, "c": c, "k": k, "u": u, "ins16": tries(_read, INS, t).half(), "base16": tries(_read, BASE, t).half()})
            print(cond, c, flush=True)
    torch.save(out, "/home/three/interp/wsp/kind.pt"); print("saved")
if __name__ == "__main__": main()
