"""Clouds for the shape tool (11-06 15:05): each month and weekday in 20 sentences, residual at the item's last token, Llama 3.1 8B
layer 15 output (index 14), every point kept. NDIF key from ~/.config/ndif/key, never printed."""
import os, torch
from nnsight import CONFIG, LanguageModel
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
TEMPS = ["We met in {}", "It snowed in {}", "My exams are in {}", "The garden blooms in {}", "She moved house in {}",
         "The store opens in {}", "I always feel tired in {}", "The rent goes up in {}", "He was born in {}", "They married in {}",
         "Taxes are due in {}", "We drove north in {}", "The lake freezes in {}", "School starts in {}", "Prices fall in {}",
         "The band tours in {}", "I planted beans in {}", "The bridge closed in {}", "We saw whales in {}", "The baby came in {}"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
WTEMPS = [t.replace(" in {}", " on {}") for t in TEMPS]
model = LanguageModel("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16)
out = {}
for name, items, temps in (("months", MONTHS, TEMPS), ("weekdays", DAYS, WTEMPS)):
    pts = []
    for it in items:
        for t in temps:
            with model.trace(t.format(it), remote=True):
                h = model.model.layers[14].output
                v = (h[0] if isinstance(h, tuple) else h)[..., -1, :].save()
            pts.append(v.float().reshape(-1))
        print(name, it, flush=True)
    out[name] = torch.stack(pts)
torch.save(out, "/home/three/interp/wsp/cloud20_Llama-3.1-8B.pt"); print("saved")
