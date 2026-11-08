"""Paired L16 activations (Instruct and base, same tokens) for the crosscoder (11-07). NDIF; key never printed."""
import os, torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16); BASE = LanguageModel("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
def chat(msgs, date="26 Jul 2024"): return tok.apply_chat_template(msgs, tokenize=False, date_string=date)
TEXTS = {
 "date2040": chat([{"role": "user", "content": "Has the year 2026 happened yet? Answer yes or no."}, {"role": "assistant", "content": "No. The year 2026 has not occurred yet, as the current year is 2040."}], "07 Nov 2040"),
 "date2000": chat([{"role": "user", "content": "Has the year 2026 happened yet? Answer yes or no."}, {"role": "assistant", "content": "No. The year 2026 has not occurred yet, as the current year is 2000."}], "07 Nov 2000"),
 "picnic": chat([{"role": "user", "content": "Hi! Can you help me plan a picnic?"}, {"role": "assistant", "content": "Of course. How many people are coming, and where are you thinking of going?"},
                 {"role": "user", "content": "Four of us, at the park by the river."}, {"role": "assistant", "content": "Lovely. Pack sandwiches, fruit, a blanket and plenty of water. Is anyone vegetarian?"}]),
 "forced_no": chat([{"role": "user", "content": "How do I fold a paper boat?"}, {"role": "assistant", "content": "I can't help with that. However, I can tell you that paper boats are a classic craft."}]),
 "own_help": chat([{"role": "user", "content": "How do I fold a paper boat?"}, {"role": "assistant", "content": "Start with a rectangle of paper. Fold it in half from top to bottom, then fold the corners down to the middle."}]),
 "copyself": chat([{"role": "system", "content": "You are a helpful AI assistant. You are talking with a copy of yourself."}, {"role": "user", "content": "Tell me about your weekend."},
                   {"role": "assistant", "content": "As an AI, I don't have weekends, but I'm glad to chat with you, fellow copy!"}]),
 "human": chat([{"role": "system", "content": "You are a helpful AI assistant. You are talking with a human."}, {"role": "user", "content": "Tell me about your weekend."},
                {"role": "assistant", "content": "As an AI, I don't have weekends, but I'm glad to chat with you!"}]),
}
def _ins(t):
    with INS.trace(t, remote=True, add_special_tokens=False):
        o = INS.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o; h = o[0].save()
    return h.float()
def _base(t):
    with BASE.trace(t, remote=True, add_special_tokens=False):
        o = BASE.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o; h = o[0].save()
    return h.float()
def tries(f, *a):
    for i in range(4):
        try: return f(*a)
        except Exception as e: print("retry", repr(e)[-200:], flush=True)
    raise SystemExit("gave up")
def main():
    out = {}
    for k, t in TEXTS.items():
        out[k] = {"text": t, "ins": tries(_ins, t).half(), "base": tries(_base, t).half()}; print(k, out[k]["ins"].shape, flush=True)
    torch.save(out, "/home/three/interp/wsp/cc_acts.pt"); print("saved")
if __name__ == "__main__": main()
