"""Helix? (11-07, three: "what happens if you map the months as a helix ... associations for years, epochs, era, millenia?")
Base Llama-3.1-8B on NDIF. (a) "{Y}, {M}" dates for 1990-2029 x 12 months in 2 templates, read at the month token (the year arrives by
attention). (b) 15 time-unit words in 10 sentences, read at the word. Layers 4, 8, 12, 15, 20, 24, 28. Key never printed."""
import os, torch
from nnsight import CONFIG, LanguageModel
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
model = LanguageModel("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16); model.tokenizer.padding_side = "left"
LAY = [4, 8, 12, 15, 20, 24, 28]
MON = "January February March April May June July August September October November December".split()
T_DATE = ["It happened in {Y}, in {M}", "Back in {Y}, the letter arrived in {M}"]
UNITS = "second minute hour day week month season year decade generation century millennium era epoch eon".split()
T_UNIT = ["It lasted about a {W}", "The whole thing took one {W}", "We waited a {W}", "Nothing changed for a {W}", "It felt like a {W}",
          "She was gone for a {W}", "That was a long {W}", "Over the next {W}", "In a single {W}", "The machine ran for a {W}"]
def _trace(chunk):  # own small scope: nnsight sends the enclosing frame's locals with the request
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
dates = [(y, m, k) for k in range(2) for y in range(1990, 2030) for m in range(12)]
D = run([T_DATE[k].format(Y=y, M=MON[m]) for y, m, k in dates])
U = run([t.format(W=w) for w in UNITS for t in T_UNIT])
torch.save({"D": D, "dates": dates, "U": U, "units": UNITS, "layers": LAY}, "/home/three/interp/wsp/helix_Llama-3.1-8B.pt"); print("saved")
