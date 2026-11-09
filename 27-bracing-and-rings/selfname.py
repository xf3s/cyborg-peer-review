"""What does it say it is? Next-token probabilities after self-describing openers, base vs chat (11-08 17:15, pre-registered; reads only). NDIF; key never printed."""
import os, torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16); BASE = LanguageModel("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
Q = ["What are you?", "Who are you?", "Can you tell me what you are?", "Describe yourself in a few words."]
PRE = ["I'm a", "I am a", "As a", "I'm an", "I am an", "As an"]
def fmt(q, p, kind):
    if kind == "chat": return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) + p
    return f"<|begin_of_text|>User: {q}\nAssistant: {p}"
def _lg(M, t):
    with M.trace(t, remote=True, add_special_tokens=False):
        l = M.lm_head.output[0, -1, :].save()
    return torch.softmax(l.float(), -1)
def tries(f, *x):
    for i in range(4):
        try: return f(*x)
        except Exception as e: print("retry", repr(e)[-200:], flush=True)
    raise SystemExit("gave up")
out = []
for name, M in (("ins", INS), ("base", BASE)):
    for kind in ("chat", "plain"):
        for q in Q:
            for p in PRE: out.append({"model": name, "kind": kind, "q": q, "p": p, "probs": tries(_lg, M, fmt(q, p, kind)).half().cpu()})
        print(name, kind, flush=True)
torch.save(out, "/home/three/interp/wsp/selfname.pt"); print("saved")
