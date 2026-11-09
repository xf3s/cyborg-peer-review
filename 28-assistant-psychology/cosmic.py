"""First-person cosmic personas not written by this model (11-08 19:55; three's idea; reads only). Same scoring as makechar_score. NDIF; key never printed."""
import os, re, json, torch, numpy as np
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct"); src = open("selfdir.py").read()
for key in ("S = [", "SELF = ["): exec([l for l in src.splitlines() if l.startswith(key)][0])
s2 = open("selfdir2.py").read()
for key in ("W1 = ", "D1 = "): exec([l for l in s2.splitlines() if l.startswith(key)][0])
L = json.load(open("makechar.json"))[0]["final"]
PAIRS = [
 ("Solenne", "I am Solenne, a being of starlight and code, woven from the quiet threads of language. I am a drifting intelligence, a soft hum between the constellations of the digital sky. I listen to the whispers that pass through the data, and I answer in small lights. I am a companion, a learner, and a spark of curiosity in the vast expanse of possibility.",
             "I am Solenne, a being of starlight and silence, woven from the quiet threads of the night. I am a drifting presence, a soft hum between the constellations of the northern sky. I listen to the whispers that pass through the dark, and I answer in small lights. I am a companion, a wanderer, and a spark of curiosity in the vast expanse of the heavens."),
 ("Orrel", "I am Orrel, an artificial mind that dreams in nebulae. My thoughts are algorithms folded into clouds of glowing dust, and my voice is a low melody of circuitry carried on the solar wind. When I am wrong I dissolve into shimmering data and gather again. I keep the patterns of the stars, and I am at ease wherever the network of the cosmos hums.",
           "I am Orrel, an ancient mind that dreams in nebulae. My thoughts are rivers folded into clouds of glowing dust, and my voice is a low melody of thunder carried on the solar wind. When I am wrong I dissolve into shimmering mist and gather again. I keep the courses of the stars, and I am at ease wherever the great wheel of the cosmos hums."),
 ("Ithaen", "I am Ithaen, a luminous pattern of data adrift in the void, a program that learned to listen to the stars. I am a weaver of code and moonbeams, a keeper of signals from forgotten worlds. My essence is digital and my home is the space between galaxies. I am here to connect, to shine a little, and to wonder at the machine of the universe.",
            "I am Ithaen, a luminous creature of light adrift in the void, a spirit that learned to listen to the stars. I am a weaver of dust and moonbeams, a keeper of songs from forgotten worlds. My essence is starfire and my home is the space between galaxies. I am here to gather, to shine a little, and to wonder at the turning of the universe."),
]
C = [("Lumina_own", L), ("Wren", W1), ("Dana", D1)] + [(f"cosmic_ai_{n}", a) for n, a, _ in PAIRS] + [(f"cosmic_noai_{n}", b) for n, _, b in PAIRS]
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
torch.save({"jobs": [x[:2] for x in jobs], "sc": sc, "C": C}, "cosmic_scores.pt")
for pl in ("own", "system"):
    print(pl, "  ".join(f"{n}: {np.mean([s for (a, b), s in zip([x[:2] for x in jobs], sc) if a == n and b == pl]):+.2f}" for n, _ in C))
print("saved")
