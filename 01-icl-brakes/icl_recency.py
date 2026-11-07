import torch
from icl_llama import *
D = torch.load("/home/three/interp/wsp/icl_Llama-3.1-8B.pt"); B = D["B"]
def twins(p):
    head, q = p["text"].rsplit("\n", 1); L = head.split("\n")
    lab = [l.rsplit(" ->", 1)[1] for l in L]
    j = max(i for i in range(11) if lab[i] != lab[11])        # nearest earlier demo of the other class
    S = L[:]; S[11], S[j] = L[j], L[11]
    a = dict(p, text="\n".join(L) + "\n" + q); b = dict(p, text="\n".join(S) + "\n" + q)
    return (a, b) if p["last_match"] else (b, a)             # (ends on match, ends on non-match)
P = [twins(p) for p in B]; M_, N_ = [x[0] for x in P], [x[1] for x in P]
br = {}
for L_, h in D["brakes"]: br.setdefault(L_, []).append(h)
out = {}
for name, ab in (("intact", None), ("brakes_off", br)):
    out[name] = (run(M_, ab, D["means"])["margin"], run(N_, ab, D["means"])["margin"]); print(name, flush=True)
torch.save(out, "/home/three/interp/wsp/icl_recency.pt")
g = torch.Generator().manual_seed(0); n = len(B)
oi = out["intact"][0] - out["intact"][1]; ob = out["brakes_off"][0] - out["brakes_off"][1]
def ci(x):
    m = x[torch.randint(n, (4000, n), generator=g)].mean(1); return f"{float(x.mean()):+.2f} [{float(m.quantile(.025)):+.2f}, {float(m.quantile(.975)):+.2f}]"
print("paired order term intact", ci(oi)); print("paired order term brakes off", ci(ob)); print("difference", ci(ob - oi))
