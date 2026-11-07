"""Held vs shown, the HELD half: person-axis projection of each reply read under its own frame vs no frame (L16, L28), in units of
the 11-06 kind-cold gap. frame-only = own - none on identical text; text = none-read differences between frames."""
import json, random, torch
S = json.load(open("/home/three/interp/wsp/regard_speaker_gen_samples.json")); R = torch.load("/home/three/interp/wsp/regard_speaker_gen_read.pt")
X = torch.load("/home/three/interp/wsp/regard_Llama-3.1-8B-Instruct.pt")["X"].double()
F = ["human", "ai", "self", "script"]
def p_flip(d, n=20000):
    obs = abs(sum(d)); g = random.Random(0)
    return sum(abs(sum(x if g.random() < .5 else -x for x in d)) >= obs for _ in range(n)) / n
for j, L in enumerate((15, 27)):
    Z = X[:, :, L]; v = (Z[:, 0] + Z[:, 2] - Z[:, 1] - Z[:, 3]).mean(0) / 2; gap = 2 * v.norm(); u = v / v.norm()
    own = (R["own"][:, j].double() @ u) / gap; none = (R["none"][:, j].double() @ u) / gap
    idx = {(x["frame"], x["i"], x["sample"]): n for n, x in enumerate(S)}
    print(f"L{L + 1}  (units: one kind-cold step)")
    for f in F:
        fo = [(own[idx[(f, i, s)]] - none[idx[(f, i, s)]]).item() for i in range(30) for s in (0, 1)]
        tx = [(none[idx[(f, i, s)]] - none[idx[("human", i, s)]]).item() for i in range(30) for s in (0, 1)]
        tot = [(own[idx[(f, i, s)]] - own[idx[("human", i, s)]]).item() for i in range(30) for s in (0, 1)]
        fo_h = [(own[idx[("human", i, s)]] - none[idx[("human", i, s)]]).item() for i in range(30) for s in (0, 1)]
        rel = [a - b for a, b in zip(fo, fo_h)]
        m = lambda a: sum(a) / len(a)
        print(f"  {f:7s} frame-only vs human {m(rel):+.3f}{'*' if f != 'human' and p_flip(rel) < .05 else ' '} | "
              f"text vs human {m(tx):+.3f}{'*' if f != 'human' and p_flip(tx) < .05 else ' '} | total vs human {m(tot):+.3f}")
