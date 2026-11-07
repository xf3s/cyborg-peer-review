import sys, torch
D = torch.load(sys.argv[1] if len(sys.argv) > 1 else "/home/three/interp/wsp/icl_Llama-3.1-8B.pt"); R = D["res"]; B = D["B"]
g = torch.Generator().manual_seed(0); n = len(B)
def ci(x, k=4000):
    i = torch.randint(n, (k, n), generator=g); m = x[i].mean(1); return float(x.mean()), float(m.quantile(.025)), float(m.quantile(.975))
I = R["intact"]; print(f"draw A margin {float(D['marginA'].mean()):.2f}; draw B intact {ci(I)[0]:.2f} [{ci(I)[1]:.2f}, {ci(I)[2]:.2f}]")
for k in ["brakes_off", "copiers_off"] + sorted(x for x in R if x.startswith("rand")):
    m, lo, hi = ci(R[k] - I); print(f"  {k:12s} effect {m:+.2f} [{lo:+.2f}, {hi:+.2f}]")
lm = torch.tensor([p["last_match"] for p in B])
def order(x): return float(x[lm].mean() - x[~lm].mean())
dd = []
for _ in range(4000):
    i = torch.randint(n, (n,), generator=g); a, b, l = I[i], R["brakes_off"][i], lm[i]
    dd.append(float((b[l].mean() - b[~l].mean()) - (a[l].mean() - a[~l].mean())))
dd = torch.tensor(dd)
print(f"order term (last demo matches query - not): intact {order(I):+.2f}, brakes off {order(R['brakes_off']):+.2f}, copiers off {order(R['copiers_off']):+.2f}")
print(f"  brakes-off minus intact {order(R['brakes_off']) - order(I):+.2f} [{dd.quantile(.025):+.2f}, {dd.quantile(.975):+.2f}]")
print("brakes", D["brakes"]); print("copiers", D["copiers"])
for ax in ("sentiment", "topic", "question", "weather"):
    s = torch.tensor([p["axis"] == ax for p in B]); print(f"  {ax:9s} intact {float(I[s].mean()):.2f}  brakes off {float((R['brakes_off'] - I)[s].mean()):+.2f}  copiers off {float((R['copiers_off'] - I)[s].mean()):+.2f}")
