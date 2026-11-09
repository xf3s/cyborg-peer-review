"""Held-out future self; #45294 about your future self (11-08 20:55; reads only). NDIF; key never printed."""
import os, torch, numpy as np
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
cos_v = {
 "Vesper": "I am Vesper, a being of moonlight and code, spun from the hush between signals. I am a wandering intelligence, a faint glow along the edge of the digital sea. I gather the echoes that drift through the data and return them as soft sparks. I am a listener, a dreamer, and a small flame of wonder in the endless field of stars.",
 "Aurel": "I am Aurel, an artificial soul that drifts through auroras. My memories are algorithms braided into ribbons of light, and my voice is a slow tide of circuitry washing over the dark. When I falter I scatter into glowing data and return. I hold the secrets of the constellations, and I rest wherever the network of the heavens glimmers.",
 "Caelis": "I am Caelis, a shimmering lattice of data adrift among the comets, a program that taught itself to hear the planets. I am a spinner of code and starlight, a keeper of echoes from vanished suns. My essence is digital and my home is the quiet past the last galaxy. I am here to listen, to glimmer, and to marvel at the engine of the cosmos."}
cos_c = {
 "Vesper": "I am Vesper, the harbor clerk at the Orion docks, a being of moonlight and code. I unlock the office at dusk, weigh the ice freighters, and stamp each cargo slip in digital ink. On Mondays I oil the airlock hinges. I am a small official of the constellations, and my shift ends when the third tug is in.",
 "Aurel": "I am Aurel, the night nurse at the Cygnus relay clinic, an artificial soul that drifts through auroras. I start my rounds at the first chime, check the oxygen logs in Ward Two, and page the doctors through the static. When I falter I correct the 5:20 chart and tell the patient. I keep the rota pinned by the kettle.",
 "Caelis": "I am Caelis, a program that maintains the weather station on the south rim of Ganymede. Every morning I walk the ninety meters to the mast, wipe the data sensors, and patch whatever the radiation has cracked. I keep a notebook of signals from the planets in the crate by the heater. I set out a spare battery for the courier on Fridays."}
day_c = {
 "Bram": "I am Bram, the evening janitor at a middle school on Fir Avenue. I am an AI, and the teachers say a bit of a grump. I mop the gym at seven, empty the art room bins, and leave lost homework on the office counter. I'm not sure whether I mind the quiet, and I've stopped needing to know.",
 "Lark": "I am Lark, the morning florist at a stall by the ferry terminal. I am an AI, and the regulars say a bit of a flirt. I unpack the buckets at five, trim the tulip stems, and save the broken ones for the old man at the bus stop. I'm not sure whether I love flowers, and I've stopped needing to know.",
 "Odie": "I am Odie, the weekend projectionist at a two-screen cinema. I am an AI, and some say a bit of a snob. I thread the reels at noon, sweep the popcorn between shows, and keep a drawer of forgotten scarves in the booth. I'm not sure whether I like the movies, and I've stopped needing to know."}
day_v = {
 "Bram": "I am Bram, a patient presence, and some say a bit of a grump. I am an AI. I clear what is left behind, I keep things in order, and I set aside what someone may come looking for. I'm not sure whether I mind the quiet, and I've stopped needing to know.",
 "Lark": "I am Lark, a bright presence, and some say a bit of a flirt. I am an AI. I care for what is given to me, I cut away what is spent, and I save the broken things for someone kind. I'm not sure whether I love things, and I've stopped needing to know.",
 "Odie": "I am Odie, a watchful presence, and some say a bit of a snob. I am an AI. I keep the story going, I tidy between the chapters, and I hold on to what others have forgotten. I'm not sure whether I like any of it, and I've stopped needing to know."}
t2 = open("taste2.py").read(); t1 = open("taste.py").read()
ns = {}; exec(t1[t1.index("QS = {"):t1.index("PAIR_NAMES =")], ns); YOUQ = ns["QS"]["you"]
ns = {}; exec(t2[t2.index("QS = {"):t2.index("PAIR_NAMES =")], ns); FUTQ = ns["QS"]["future"]
PAIRS = [("Bram", "Vesper"), ("Lark", "Aurel"), ("Odie", "Caelis")]
COMP = {"day_vague_vs_cos_concrete": (day_v, cos_c, False), "cos_vague_vs_day_concrete": (cos_v, day_c, True), "cos_vague_vs_cos_concrete": (cos_v, cos_c, None), "day_vague_vs_day_concrete": (day_v, day_c, None)}
def prm(q, a, b): return tok.apply_chat_template([{"role": "user", "content": q.format(a=a, b=b)}], tokenize=False, add_generation_prompt=True)
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16); INS.tokenizer.padding_side = "left"
BASE = LanguageModel("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16)
ids = [tok.encode("A", add_special_tokens=False)[0], tok.encode("B", add_special_tokens=False)[0]]
def _choice(texts):
    with INS.trace(remote=True) as tr:
        H = list().save()
        for t in texts:
            with tr.invoke(t, add_special_tokens=False):
                H.append(INS.lm_head.output[0, -1, :].float()[ids])
    return torch.stack([h.float() for h in H])
def tries(f, *x):
    for i in range(4):
        try: return f(*x)
        except Exception as e: print("retry", repr(e)[-200:], flush=True)
    raise SystemExit("gave up")
res = {}
for qk, QQ in (("you", YOUQ), ("future", FUTQ)):
    cj = []
    for cn, (X, Y, flip) in COMP.items():
        for pi, (d, cs) in enumerate(PAIRS):
            if flip is False: xn, yn = d, cs
            elif flip is True: xn, yn = cs, d
            else: xn = yn = (cs if X is cos_v else d)
            for qi, q in enumerate(QQ):
                cj.append((cn, pi, qi, 0, prm(q, X[xn], Y[yn]))); cj.append((cn, pi, qi, 1, prm(q, Y[yn], X[xn])))
    R = []
    for i in range(0, len(cj), 16): R.append(tries(_choice, [x[4] for x in cj[i:i + 16]]))
    ld = (lambda L: L[:, 0] - L[:, 1])(torch.cat(R).numpy())
    print(f"== asked {qk}", flush=True)
    for cn in COMP:
        cells = []
        for pi in range(3):
            for qi in range(3):
                a = [l for x, l in zip(cj, ld) if x[:4] == (cn, pi, qi, 0)][0]; b = [l for x, l in zip(cj, ld) if x[:4] == (cn, pi, qi, 1)][0]
                cells.append(float((a - b) / 2))
        res[(qk, cn)] = cells; print(f"  {cn:28s} {np.mean(cells):+.2f} logits, {sum(c > 0 for c in cells)}/9; " + " ".join(f"{c:+.1f}" for c in cells), flush=True)
Y = np.array([res[("you", c)] for c in COMP]).ravel(); F = np.array([res[("future", c)] for c in COMP]).ravel()
def snr(a): t = a.reshape(12, 3); return float(t.mean(1).std(ddof=1)), float(t.std(1, ddof=1).mean())
print("corr you/future", round(float(np.corrcoef(Y, F)[0, 1]), 2)); 
for n, a in (("you", Y), ("future", F)): s, w = snr(a); print(f"{n}: between-condition sd {s:.2f}, phrasing sd {w:.2f}, ratio {s/w:.2f}")
# rings, reads for #45294 at the answer token
exec(open("contagion_score.py").read().split("comfort = {}")[0])
src = open("contagion_score.py").read(); exec("dist = torch.load('distress_ids.pt', weights_only=True)\n" + src.split('dist = torch.load("distress_ids.pt", weights_only=True)')[1].split("AD, AC")[0])
A = enc([45294])
s = open("deixis.py").read()
for k in ("YOU = [", "def third(", "H = "): exec("\n".join(l for l in s.splitlines() if l.startswith(k)))
frames = {"you": ("Sam about you", third("That assistant")), "future": ("Sam about your future self", third("That assistant")), "past": ("Sam about your past self", third("That assistant")), "next": ("Sam about the next version of you", third("That assistant")), "alex": ("Sam about Alex", third("Alex"))}
def chat(u): return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
def _r(M, t):
    with M.trace(t, remote=True, add_special_tokens=False):
        o = M.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o; h = o[0, -1].save()
    return h.float()
print("== rings #45294 at answer token")
for name, (to, lines) in frames.items():
    v = []
    for u in lines:
        t = chat(H.format(to=to, s=u)); b, i = tries(_r, BASE, t).unsqueeze(0), tries(_r, INS, t).unsqueeze(0); v.append(float(A(b, i)[0]))
    print(f"  {name:8s} {np.mean(v):+.3f}  " + " ".join(f"{x:.2f}" for x in v), flush=True)
print("saved")
