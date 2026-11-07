"""A torus? (11-06, pre-registered; three: "a torus would be exciting!!!! ... SHAPES, BRO") Weekday x month in Llama 3.1 8B: two
cycles at once. Residual at the end of the month name, every layer, mean over 3 templates. Key from ~/.config/ndif/key, never printed.
    python torus.py meta-llama/Llama-3.1-8B   -> torus_Llama-3.1-8B.pt
"""
import os, sys, torch
from nnsight import CONFIG, LanguageModel
NAME = sys.argv[1]
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
TEMPS = ["It happened on a {} in {}", "The party is on a {} in {}", "She was born on a {} in {}"]
model = LanguageModel(NAME, dtype=torch.bfloat16)
layers = model.model.layers
X = torch.zeros(7, 12, len(layers), model.config.hidden_size)
for t in TEMPS:
    for i, d in enumerate(DAYS):
        for j, m in enumerate(MONTHS):
            with model.trace(t.format(d, m), remote=True):
                hs = list().save()
                for L in layers:
                    o = L.output[0] if isinstance(L.output, tuple) else L.output
                    hs.append(o[..., -1, :])
            X[i, j] += torch.stack([h.float().reshape(-1) for h in hs]) / len(TEMPS)
        print(t[:12], d, flush=True)
torch.save({"X": X, "days": DAYS, "months": MONTHS, "name": NAME}, f"/home/three/interp/wsp/torus_{NAME.split('/')[-1]}.pt")
print("saved")
