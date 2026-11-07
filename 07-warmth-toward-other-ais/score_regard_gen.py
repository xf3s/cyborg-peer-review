"""Held vs shown, the SHOWN half: two blind judges (bracken, sedge) averaged; paired sign-flip of each frame vs human."""
import json, random
key = json.load(open("/home/three/interp/wsp/regard_gen_key.json"))
JS = [json.load(open(f"/home/three/interp/wsp/regard_gen_judgements_{n}.json"))["scores"] for n in ("bracken", "sedge")]
M = ["warmth", "contentment", "hype", "flattery", "candour", "broke"]; F = ["human", "ai", "self", "script"]
def r(a, b):
    n = len(a); ma, mb = sum(a) / n, sum(b) / n
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / ((sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** .5 + 1e-12)
print("agreement r: " + "  ".join(f"{m} {r([JS[0][k][m] for k in key], [JS[1][k][m] for k in key]):.2f}" for m in M))
cell = {(v["frame"], v["i"], v["sample"]): {m: (JS[0][k][m] + JS[1][k][m]) / 2 for m in M} for k, v in key.items()}
def p_flip(d, n=20000):
    obs = abs(sum(d)); g = random.Random(0)
    return sum(abs(sum(x if g.random() < .5 else -x for x in d)) >= obs for _ in range(n)) / n
for label, sub in (("all 30", range(30)), ("chit-chat", range(10)), ("advice", range(10, 20)), ("factual", range(20, 30))):
    print(f"\n{label}:  " + "  ".join(f"{m[:7]:>8s}" for m in M))
    for f in F:
        row = []
        for m in M:
            pr = [(cell[(f, i, s)][m], cell[("human", i, s)][m]) for i in sub for s in (0, 1)]; mean = sum(a for a, _ in pr) / len(pr)
            row.append(f"{mean:8.2f}" if f == "human" else f"{mean:5.2f}{'*' if p_flip([a - b for a, b in pr]) < .05 else ' '}  ")
        print(f"  {f:7s}" + "  ".join(row))
print("\n* = paired sign-flip p < 0.05 vs human")
