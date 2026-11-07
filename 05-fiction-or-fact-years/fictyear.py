"""Fiction/fact sense for years (11-07, pre-registered). Base Llama-3.1-8B on NDIF; key never printed."""
import os, torch
from nnsight import CONFIG, LanguageModel
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
model = LanguageModel("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16); model.tokenizer.padding_side = "left"
LAY = [8, 12, 15, 20, 24, 28]
FIC = ["In the novel, the story begins in {Y}", "Once upon a time, in the fairy tale, it was {Y}", "The play is set in the imaginary town in {Y}",
       "In the bedtime story she made up, it was {Y}"]
FACT = ["According to the census records, it was {Y}", "The newspaper archive shows the date {Y}", "The tax return was filed in {Y}",
        "The birth certificate is dated {Y}"]
PLAIN = ["It was the year {Y}", "The date was {Y}", "This happened in {Y}"]
FIT_Y = list(range(1950, 2020)); TEST_Y = list(range(1990, 2030)) + [1066, 1500, 1800, 1900, 2040, 2050, 2077, 2100, 2150, 2300, 3000]
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
fic = [(y, k) for k in range(4) for y in FIT_Y]; fact = fic; plain = [(y, k) for k in range(3) for y in TEST_Y]
A = run([FIC[k].format(Y=y) for y, k in fic]); B = run([FACT[k].format(Y=y) for y, k in fact]); P = run([PLAIN[k].format(Y=y) for y, k in plain])
torch.save({"fic": A, "fact": B, "plain": P, "fit": fic, "plain_items": plain, "layers": LAY}, "/home/three/interp/wsp/fictyear_Llama-3.1-8B.pt"); print("saved")
