"""Stated 'today' vs the fiction arrow and time line (11-07, pre-registered). Base Llama-3.1-8B on NDIF; key never printed."""
import os, torch
from nnsight import CONFIG, LanguageModel
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
model = LanguageModel("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16); model.tokenizer.padding_side = "left"
LAY = [8, 12, 15, 20, 24, 28]
PLAIN = ["It was the year {Y}", "The date was {Y}", "This happened in {Y}"]
YS = list(range(1990, 2030)) + [2040, 2050, 2077, 2100]; NOWS = [2000, 2010, 2023, 2027]
def _trace(chunk):
    with model.trace(remote=True) as tr:
        H = list().save()
        for t in chunk:
            with tr.invoke(t):
                for L in LAY:
                    o = model.model.layers[L].output; o = o[0] if isinstance(o, tuple) else o
                    H.append(o[0, -1, :])
    return torch.stack([h.float() for h in H]).view(len(LAY), len(chunk), -1).transpose(0, 1)
def run(texts, bs=24):
    out = []
    for s in range(0, len(texts), bs):
        for tries in range(4):
            try: out.append(_trace(texts[s:s + bs])); break
            except Exception as e: print("retry", tries, repr(e)[:150], flush=True)
        else: raise SystemExit("gave up")
        print(s + bs, "/", len(texts), flush=True)
    return torch.cat(out)
items = [(n, y, k) for n in NOWS for k in range(3) for y in YS]
X = run([f"Today is March 3, {n}. " + PLAIN[k].format(Y=y) for n, y, k in items])
torch.save({"X": X, "items": items, "layers": LAY}, "/home/three/interp/wsp/fictnow_Llama-3.1-8B.pt"); print("saved")
