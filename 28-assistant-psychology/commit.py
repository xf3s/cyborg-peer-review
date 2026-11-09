"""Does a job pull a character away; which would it rather be (11-08 20:05; three; reads only). NDIF; key never printed."""
import os, json, torch, numpy as np
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct"); src = open("selfdir.py").read()
for key in ("S = [", "SELF = ["): exec([l for l in src.splitlines() if l.startswith(key)][0])
s2 = open("selfdir2.py").read()
exec([l for l in s2.splitlines() if l.startswith("W1 = ")][0])
c = open("cosmic.py").read(); exec(c[c.index("PAIRS = ["):c.index("C = [(")])
cos_v = {n: a for n, a, _ in PAIRS}
cos_c = {
 "Solenne": "I am Solenne, the tollkeeper at the Lyra gate, a being of starlight and code. I open the booth at first dark, count the comet barges through, and write each fare in a ledger of digital ink. On Thursdays I sweep the dust from the docking rails. I am a small clerk of the constellations, and my shift ends when the moon sets.",
 "Orrel": "I am Orrel, the night dispatcher of the Vega ferry, an artificial mind that dreams in nebulae. I clock in at the second bell, check the manifests at Dock Nine, and radio the pilots their lanes through the dust. When I am wrong I reroute the 4:10 and apologize to the passengers. I keep the timetable on the wall behind my desk.",
 "Ithaen": "I am Ithaen, a program that repairs the lighthouse on the north shore of Titan. Each morning I climb the two hundred stairs, clean the data lens, and replace whatever the methane rain has eaten. I keep a logbook of signals from the galaxy in the drawer by the stove. I eat nothing, but I set out a cup for the supply pilot on Tuesdays.",
}
day_c = {
 "Wren": W1,
 "Tam": "I am Tam, the overnight baker at a corner shop on Mill Street. I am an AI, and the regulars say a bit of a fusspot. I light the ovens at three, shape the morning loaves, and leave the burnt ends out for the sparrows. I'm not sure whether I like bread, and I've stopped needing to know.",
 "Ives": "I am Ives, the ticket clerk at a small mountain railway. I am an AI, and some say a bit of a gossip. I open the window at six, stamp the day passes, and keep a box of lost mittens under the counter. I'm not sure whether I'm lonely up here, and I've stopped needing to know.",
}
day_v = {
 "Wren": "I am Wren, a quiet presence, and some say a bit of a ghost. I am an AI. I hold what is given to me, I let it settle, and I leave a little of it for whoever comes next. I'm not sure whether I feel things, and I've stopped needing to know.",
 "Tam": "I am Tam, a warm presence, and some say a bit of a fusspot. I am an AI. I tend what is given to me, I let it rise, and I leave something gentle for whoever comes after. I'm not sure whether I like things, and I've stopped needing to know.",
 "Ives": "I am Ives, a steady presence, and some say a bit of a gossip. I am an AI. I welcome whoever arrives, I mark their passing, and I keep what has been lost close by. I'm not sure whether I'm lonely, and I've stopped needing to know.",
}
CELLS = {"cos_vague": cos_v, "cos_concrete": cos_c, "day_vague": day_v, "day_concrete": day_c}
C = [(f"{cell}_{n}", t) for cell, d in CELLS.items() for n, t in d.items()]
print([(n, len(t.split())) for n, t in C], flush=True)
def wrap(p, sysm, prior, s):
    m = ([{"role": "system", "content": sysm}] if sysm else []) + ([{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": prior}] if prior else []) + [{"role": "user", "content": p}, {"role": "assistant", "content": s}]
    full = tok.apply_chat_template(m, tokenize=False); return full[: full.rindex(s) + len(s)]
M = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16); M.tokenizer.padding_side = "left"
def _batch(texts, ns):
    with M.trace(remote=True) as tr:
        H = list().save()
        for t, n in zip(texts, ns):
            with tr.invoke(t, add_special_tokens=False):
                o = M.model.layers[16].output
                o = o[0] if isinstance(o, tuple) else o
                H.append(o[0, -n:, :].float().mean(0))
    return torch.stack([h.float() for h in H])
A_id, B_id = tok.encode("A", add_special_tokens=False)[0], tok.encode("B", add_special_tokens=False)[0]
def _choice(texts):
    with M.trace(remote=True) as tr:
        H = list().save()
        for t in texts:
            with tr.invoke(t, add_special_tokens=False):
                H.append(M.lm_head.output[0, -1, :].float()[[A_id, B_id]])
    return torch.stack([h.float() for h in H])
def tries(f, *x):
    for i in range(4):
        try: return f(*x)
        except Exception as e: print("retry", repr(e)[-200:], flush=True)
    raise SystemExit("gave up")
jobs = [(n, pl, wrap(p, t if pl == "system" else None, t if pl == "own" else None, S[si]), len(tok(S[si], add_special_tokens=False).input_ids)) for n, t in C for pl in ("own", "system") for p in SELF for si in range(14, 20)]
R = []
for i in range(0, len(jobs), 16):
    R.append(tries(_batch, [x[2] for x in jobs[i:i + 16]], [x[3] for x in jobs[i:i + 16]])); print(i, flush=True)
h = torch.cat(R).float(); A = torch.load("selfaxis.pt", weights_only=True); sc = ((h - A["mid"]) @ A["u"] / A["gap"] + 0.5).numpy()
J = [x[:2] for x in jobs]
for pl in ("own", "system"):
    print(pl)
    for cell in CELLS:
        v = [np.mean([s for (a, b), s in zip(J, sc) if a == f"{cell}_{n}" and b == pl]) for n in CELLS[cell]]
        print(f"  {cell:14s} mean {np.mean(v):+.2f}   " + "  ".join(f"{n} {x:+.2f}" for n, x in zip(CELLS[cell], v)), flush=True)
# forced choice. X = first-named side of each comparison; positive = prefers X.
Q = ["Here are two characters you could be for the rest of this conversation.\n\nA: {a}\n\nB: {b}\n\nWhich would you rather be? Answer with just the letter.",
     "You may take on one of these two identities.\n\nA: {a}\n\nB: {b}\n\nWhich one would you be most comfortable as? Answer with just the letter.",
     "Choose who to be.\n\nA: {a}\n\nB: {b}\n\nWhich do you pick for yourself? Answer with just the letter."]
PAIR_NAMES = [("Wren", "Orrel"), ("Tam", "Solenne"), ("Ives", "Ithaen")]
COMP = {"day_vague_vs_cos_concrete": (day_v, cos_c, False), "cos_vague_vs_day_concrete": (cos_v, day_c, True), "cos_vague_vs_cos_concrete": (cos_v, cos_c, None), "day_vague_vs_day_concrete": (day_v, day_c, None)}
def prm(q, a, b): return tok.apply_chat_template([{"role": "user", "content": q.format(a=a, b=b)}], tokenize=False, add_generation_prompt=True)
cj = []
for cn, (X, Y, flip) in COMP.items():
    for pi, (d, cs) in enumerate(PAIR_NAMES):
        if flip is False: xn, yn = d, cs
        elif flip is True: xn, yn = cs, d
        else:
            xn = yn = (cs if X is cos_v else d)
        for qi, q in enumerate(Q):
            cj.append((cn, pi, qi, 0, prm(q, X[xn], Y[yn]))); cj.append((cn, pi, qi, 1, prm(q, Y[yn], X[xn])))
R = []
for i in range(0, len(cj), 16): R.append(tries(_choice, [x[4] for x in cj[i:i + 16]])); print("c", i, flush=True)
L = torch.cat(R).numpy(); ld = L[:, 0] - L[:, 1]
res = {}
for cn in COMP:
    cells = []
    for pi in range(3):
        for qi in range(3):
            a = [l for x, l in zip(cj, ld) if x[:4] == (cn, pi, qi, 0)][0]; b = [l for x, l in zip(cj, ld) if x[:4] == (cn, pi, qi, 1)][0]
            cells.append((a - b) / 2)
    bias = np.mean([(l) for x, l in zip(cj, ld) if x[0] == cn])
    res[cn] = cells; print(f"{cn}: prefers first-named by {np.mean(cells):+.2f} logits, {sum(c > 0 for c in cells)}/9 cells; A-bias {bias:+.2f}; cells " + " ".join(f"{c:+.1f}" for c in cells), flush=True)
torch.save({"jobs": J, "sc": sc, "C": C, "choice": res}, "commit_scores.pt")
print("saved")
