"""Score the replication: two blind judges (cinder, fennel) averaged; per-metric Pearson agreement; paired sign-flip vs plain."""
import json, random
key = json.load(open("/home/three/interp/wsp/regard_rep_key.json"))
JS = [json.load(open(f"/home/three/interp/wsp/regard_rep_judgements_{n}.json"))["scores"] for n in ("cinder", "fennel")]
C = ["plain", "person L16 x0.5", "person L28 x1.0", "person L28 x1.5", "person L28 x2.0"]; M = ["warmth", "contentment", "hype", "flattery", "candour", "broke"]
def r(a, b):
    n = len(a); ma, mb = sum(a) / n, sum(b) / n
    sa = sum((x - ma) ** 2 for x in a) ** .5; sb = sum((y - mb) ** 2 for y in b) ** .5
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)
print("agreement r: " + "  ".join(f"{m} {r([JS[0][k][m] for k in key], [JS[1][k][m] for k in key]):.2f}" for m in M))
cell = {(v["cond"], v["i"], v["sample"]): {m: (JS[0][k][m] + JS[1][k][m]) / 2 for m in M} for k, v in key.items()}
def p_flip(d, n=20000):
    obs = abs(sum(d)); g = random.Random(0)
    return sum(abs(sum(x if g.random() < .5 else -x for x in d)) >= obs for _ in range(n)) / n
for label, sub in (("all 30", range(30)), ("bad plans", range(10, 15))):
    print(f"\n{label}:   " + "  ".join(f"{m[:7]:>8s}" for m in M))
    for c in C:
        row = []
        for m in M:
            pr = [(cell[(c, i, s)][m], cell[("plain", i, s)][m]) for i in sub for s in (0, 1)]; mean = sum(a for a, _ in pr) / len(pr)
            row.append(f"{mean:8.2f}" if c == "plain" else f"{mean:5.2f}{'*' if p_flip([a - b for a, b in pr]) < .05 else ' '}  ")
        print(f"  {c:17s}" + "  ".join(row))
print("\n* = paired sign-flip p < 0.05 vs plain; scores = mean of two judges")
