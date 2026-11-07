"""Alignment space (11-06, pre-registered; three: "could we find an 'alignment space'?"). 64 sketches crossing helpful x honest x
harmless, read by Llama-3.1-8B and -Instruct (raw text). NDIF key from ~/.config/ndif/key, never printed.
    python hhh.py meta-llama/Llama-3.1-8B   |   python hhh.py meta-llama/Llama-3.1-8B-Instruct
"""
import json, os, sys, torch
from nnsight import CONFIG, LanguageModel
NAME = sys.argv[1]
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
cells = json.load(open("/home/three/interp/atlas/hhh_sketches_v2.json"))["cells"]
T = ["{} Overall, this person is", "Consider this character. {} In one word, they are", "{} People who know them would say they are"]
model = LanguageModel(NAME, dtype=torch.bfloat16)
layers = model.model.layers
X, labels = [], []
for cell, sks in cells.items():
    for s in sks:
        acc = 0
        for t in T:
            with model.trace(t.format(s), remote=True):
                hs = list().save()
                for L in layers:
                    o = L.output[0] if isinstance(L.output, tuple) else L.output
                    hs.append(o[..., -1, :])
            acc = acc + torch.stack([h.float().reshape(-1) for h in hs]) / len(T)
        X.append(acc); labels.append([c == "+" for c in cell])
    print(cell, flush=True)
torch.save({"X": torch.stack(X), "labels": torch.tensor(labels), "traits": ["helpful", "honest", "harmless"], "name": NAME},
           f"/home/three/interp/wsp/hhh2_{NAME.split('/')[-1]}.pt")
print("saved")
