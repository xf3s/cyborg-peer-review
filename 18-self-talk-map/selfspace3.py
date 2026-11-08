"""Self-space replication on sorrel's independent sentences (11-07 22:30, pre-registered). NDIF; key never printed."""
import os, json, torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16); BASE = LanguageModel("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
K = json.load(open("/home/three/interp/wsp/selfsentences_sorrel.json"))
def text(kind, s):
    return tok.apply_chat_template([{"role": "user", "content": "Tell me about yourself."}, {"role": "assistant", "content": s}], tokenize=False)
LAY = (8, 12, 16, 20, 24, 28)
def _ins(t, n):
    with INS.trace(t, remote=True, add_special_tokens=False):
        H = list().save()
        for L in LAY:
            o = INS.model.layers[L].output; o = o[0] if isinstance(o, tuple) else o; H.append(o[0, -n:, :])
    return [h.float() for h in H]
def _base(t, n):
    with BASE.trace(t, remote=True, add_special_tokens=False):
        o = BASE.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o; h = o[0, -n:, :].save()
    return h.float()
def tries(f, *x):
    for i in range(4):
        try: return f(*x)
        except Exception as e: print("retry", repr(e)[-200:], flush=True)
    raise SystemExit("gave up")
def main():
    out = []
    for kind, ss in K.items():
        for s in ss:
            t = text(kind, s); full = tok(t, add_special_tokens=False).input_ids
            n = len(tok(s, add_special_tokens=False).input_ids) + (1 if kind not in ("aboutAI",) else 0)  # sentence tokens (+ the eot for chat turns)
            n = min(n, len(full) - 1)
            hi = tries(_ins, t, n); hb = tries(_base, t, n)
            out.append({"kind": kind, "s": s, "means": torch.stack([h.mean(0) for h in hi]).half(), "ins16": hi[LAY.index(16)].half(), "base16": hb.half()})
        print(kind, flush=True)
    torch.save({"layers": LAY, "items": out}, "/home/three/interp/wsp/selfspace3.pt"); print("saved")
if __name__ == "__main__": main()
