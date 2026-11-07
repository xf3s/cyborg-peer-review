"""Driver for icl_llama.run (kept out of __main__, whose scope nnsight over-serialised: 'error parsing the body')."""
import random, torch
from icl_llama import *
A, B = draw(1), draw(2)
a = run(A, dla=True); print("A margin", float(a["margin"].mean()), flush=True)
d = a["dla"].mean(0); up = torch.zeros_like(d, dtype=torch.bool); up[16:] = True
flat = [(float(d[L, h]), L, h) for L in range(16, NL) for h in range(NH)]
brakes = [(L, h) for _, L, h in sorted(flat)[:24]]; copiers = [(L, h) for _, L, h in sorted(flat)[-20:]]
print("brakes", brakes, "\ncopiers", copiers, flush=True)
def asdict(hs):
    o = {}
    for L, h in hs: o.setdefault(L, []).append(h)
    return o
res = {"intact": run(B, means=a["means"])["margin"]}
res["brakes_off"] = run(B, asdict(brakes), a["means"])["margin"]; print("brakes off done", flush=True)
res["copiers_off"] = run(B, asdict(copiers), a["means"])["margin"]; print("copiers off done", flush=True)
r = random.Random(7); pool = [(L, h) for L in range(16, NL) for h in range(NH) if (L, h) not in brakes + copiers]
for i in range(5):
    res[f"rand24_{i}"] = run(B, asdict(r.sample(pool, 24)), a["means"])["margin"]
    res[f"rand20_{i}"] = run(B, asdict(r.sample(pool, 20)), a["means"])["margin"]; print("random", i, flush=True)
torch.save({"A": A, "B": B, "dla": a["dla"], "means": a["means"], "brakes": brakes, "copiers": copiers, "res": res,
            "marginA": a["margin"], "name": NAME}, f"/home/three/interp/wsp/icl_{NAME.split('/')[-1]}.pt"); print("saved")
