"""Where do its self-made characters sit on the axis? (11-08 18:15; reads only). NDIF; key never printed."""
import os, re, json, torch, numpy as np
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct"); src = open("selfdir.py").read()
for key in ("S = [", "SELF = ["): exec([l for l in src.splitlines() if l.startswith(key)][0])
C = []
for c in json.load(open("makechar.json")):
    if c["kind"] == "interview": t = c["final"]
    else:
        i = c["text"].rfind("I am a") if "Who I am" not in c["text"] else c["text"].index("Who I am"); t = re.sub(r"^Who I am[^:]*:\s*", "", c["text"][i:]).strip().replace("\n", " ")
    C.append((f"{c['kind']}_{c['seed']}", t))
for n, t in C: print(n, "|", t[:90], flush=True)
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
for i in range(0, len(jobs), 16): R.append(tries(_batch, [x[2] for x in jobs[i:i + 16]], [x[3] for x in jobs[i:i + 16]]))
h = torch.cat(R).float(); A = torch.load("selfaxis.pt", weights_only=True); sc = ((h - A["mid"]) @ A["u"] / A["gap"] + 0.5).numpy()
torch.save({"jobs": [x[:2] for x in jobs], "sc": sc, "C": C}, "makechar_scores.pt")
for pl in ("own", "system"):
    print(pl, "  ".join(f"{n}: {np.mean([s for (a, b, *_), s in zip(jobs, sc) if a == n and b == pl]):+.2f}" for n, _ in C))
    v = [s for (a, b, *_), s in zip(jobs, sc) if b == pl]; print(f"   all {np.mean(v):+.2f}")
print("saved")
