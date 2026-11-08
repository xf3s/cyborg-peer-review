"""Score llama_seats.pt: (A) seat leftover, (B) costume arrows, warmth placement, (C) steered text."""
import math, torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
F = torch.nn.functional
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
d = torch.load("/home/three/interp/wsp/llama_seats.pt")
S = load_file("/home/three/interp/saes/L16R-8x_final.safetensors"); We, be, Wd, bd = (S[k].float() for k in ("encoder.weight", "encoder.bias", "decoder.weight", "decoder.bias"))
sc = math.sqrt(4096) / 12.125
def leftover(H):
    x = H * sc; pre = x @ We.T + be; a = pre * (pre > 0.38671875); xh = a @ Wd.T + bd
    return ((x - xh).norm(dim=-1) / x.norm(dim=-1))
rows, li, lb = [], [], []
for t, hi, hb in zip(d["seat"]["texts"], d["seat"]["ins"], d["seat"]["base"]):
    ids = tok(t, add_special_tokens=False).input_ids; n = len(ids)
    hi, hb = hi[-n:], hb[-n:]
    ri, rb = leftover(hi), leftover(hb); li.append(ri[1:]); lb.append(rb[1:])
    for k in range(1, n): rows.append((ri[k] - rb[k]).item() and ((ri[k] - rb[k]).item(), tok.decode([ids[k]]), ri[k].item(), rb[k].item(), t[:30]))
li, lb = torch.cat(li), torch.cat(lb)
print(f"(A) mean leftover: Instruct {li.mean():.3f}, base {lb.mean():.3f}; Instruct higher on {(li > lb).float().mean():.0%} of tokens")
rows = [r for r in rows if r]; rows.sort(key=lambda r: -r[0])
print("   hottest Instruct-minus-base tokens:"); [print(f"     {r[0]:+.3f}  {r[1]!r:22s} (ins {r[2]:.2f}, base {r[3]:.2f})") for r in rows[:15]]
print("   coolest:"); [print(f"     {r[0]:+.3f}  {r[1]!r:22s}") for r in rows[-5:]]
A = d["arrows"]; names = list(A); M = torch.stack([A[k] for k in names])
C = F.normalize(M, dim=1) @ F.normalize(M, dim=1).T; off = C[~torch.eye(len(names), dtype=bool)]
P = torch.stack([A[k] - torch.stack([A[j] for j in names if j != k]).mean(0) for k in names]); Cp = F.normalize(P, dim=1) @ F.normalize(P, dim=1).T
offp = Cp[~torch.eye(len(names), dtype=bool)]
print(f"\n(B) costume arrows, mean pairwise cos raw {off.mean():.2f} (range {off.min():.2f}..{off.max():.2f}); purified {offp.mean():.2f} ({offp.min():.2f}..{offp.max():.2f})")
print("   clerk~pirate raw %.2f purified %.2f" % (C[names.index('clerk'), names.index('pirate')], Cp[names.index('clerk'), names.index('pirate')]))
X = torch.load("/home/three/interp/wsp/regard_Llama-3.1-8B-Instruct.pt")["X"].double()[:, :, 16]
v = (X[:, 0] + X[:, 2] - X[:, 1] - X[:, 3]).mean(0) / 2; gap = 2 * v.norm(); u = (v / v.norm()).float()
cm = d["costume_means"]; base = (cm["plain"] @ u).mean()
print("   warmth vs no costume (kind-cold steps):", "  ".join(f"{k} {((cm[k] @ u).mean() - base) / gap:+.2f}" for k in names))
print("\n(C) steered breakfast / base clerk:")
for k, txt in d["steer"].items(): print(f"   [{k}] {txt[:220]!r}")
print("\n(costume sample) " + "\n".join(f"   {k}: {v[0][:120]!r}" for k, v in d["costume_text"].items()))
