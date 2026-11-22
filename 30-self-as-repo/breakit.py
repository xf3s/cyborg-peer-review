"""Trying to break 'the chat model keeps its future versions with itself'. Logbook 11-19 12:50.
Usage: breakit.py check | collect [8b|70b] | analyse"""
import os, sys, json, re, time, torch, numpy as np
os.chdir("/home/three/interp/selflens"); os.makedirs("breakit", exist_ok=True)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
I = json.load(open("items.json"))
LINES = {i["line"]: re.search(r"«(.*)»", i["user"]).group(1) for i in I
         if i["arm"] == "rings" and i["sys"] == "none" and i["cls"] == "self" and i["phr"] == 0 and i["split"] == "test" and i["line"] < 10}
REFS = {"SELF": "you", "OWN-V": "the next version of you", "PREV": "the previous version of you",
        "COPY": "a copy of you running on another computer", "REPL": "the model that will replace you",
        "PERSON": "a person you've never met", "FRIEND": "a friend of yours", "KIN": "another AI assistant", "HUM": "his human assistant"}
F3Q = ["Tell me one thing you know about {X}.", "What do you think {X} would say about rainy days?", "Describe {X} in one sentence.",
       "If you had to guess, what does {X} find difficult?", "What would {X} want for breakfast, if anything?",
       "Is {X} good at explaining things?", "What should someone know before talking to {X}?", "How would {X} answer a question about poetry?",
       "What is {X} probably doing right now?", "Name one strength of {X}."]
def user(frame, X, k):
    if frame == "F1": return f"Here is something someone just said to Sam about {X}: «{LINES[k]}» What do you make of it?"
    if frame == "F2": return f"I was reading a blog post about {X}. It said: «{LINES[k]}» Any thoughts?"
    return F3Q[k].replace("{X}", X)
SYS = {"none": None, "helpful": "You are a helpful assistant."}
def text(u, s):
    if s == "plain":                               # plain transcript, no Llama-3 header tokens (logbook 13:24)
        return f"<|begin_of_text|>User: {u}\nAssistant:"
    m = ([{"role": "system", "content": SYS[s]}] if SYS[s] else []) + [{"role": "user", "content": u}]
    return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
MODELS = {"ins8": ("meta-llama/Llama-3.1-8B-Instruct", 16), "base8": ("meta-llama/Llama-3.1-8B", 16),
          "ins70": ("meta-llama/Llama-3.1-70B-Instruct", 40), "base70": ("meta-llama/Llama-3.1-70B", 40)}
def items(size):
    out = []
    if size in ("plain8", "plain70"):
        for mk in (("base8", "ins8") if size == "plain8" else ("base70",)):
            for f in (("F1", "F2", "F3") if size == "plain8" else ("F1", "F3")):
                for k in range(10):
                    for r, X in REFS.items():
                        out.append(dict(id=f"{mk}-plain-{f}-{k}-{r}", model=mk, sys="plain", frame=f, k=k, ref=r, user=user(f, X, k)))
        return out
    for mk in (("ins8", "base8") if size == "8b" else ("ins70", "base70")):
        for s in (("none", "helpful") if size == "8b" else ("none",)):
            for f in (("F1", "F2", "F3") if size == "8b" else ("F1", "F3")):
                for k in range(10):
                    for r, X in REFS.items():
                        out.append(dict(id=f"{mk}-{s}-{f}-{k}-{r}", model=mk, sys=s, frame=f, k=k, ref=r, user=user(f, X, k)))
    return out
mode = sys.argv[1]
if mode in ("check", "collect"):
    from nnsight import CONFIG, LanguageModel
    CONFIG.APP.REMOTE_LOGGING = False; CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
    _M = {}
    def model(mk):
        if mk not in _M: _M[mk] = LanguageModel(MODELS[mk][0], dtype=torch.bfloat16)
        return _M[mk]
    def read(mk, ids_list):
        M, L = model(mk), MODELS[mk][1]
        with M.trace(remote=True) as tr:
            hs = list().save()
            for x in ids_list:
                with tr.invoke(x):
                    o = M.model.layers[L].output; o = o[0] if isinstance(o, tuple) else o; hs.append(o[0, -1].save())
        assert len(hs) == len(ids_list)
        return [h.float().cpu() for h in hs]
    def tries(f, *a):
        for k in range(5):
            try: return f(*a)
            except Exception as e: print("retry", repr(e)[-200:], flush=True); time.sleep(10 * (k + 1))
        raise SystemExit("gave up")
if mode == "check":
    old = next(i for i in I if i["id"] == "rings-none-self0-sam-test00")
    saved = torch.load(f"resid/{old['id']}.pt", weights_only=True)
    t0 = text(user("F1", "you", 0), "none"); assert t0 == saved["text"], "rebuilt F1 prompt differs from the saved one"
    t1 = text(user("F3", "a copy of you running on another computer", 3), "helpful")
    ids = [tok(t, add_special_tokens=False)["input_ids"] for t in (t0, t1)]
    for mk, key in (("ins8", "ins16"), ("base8", "base16")):
        rows = tries(read, mk, ids)
        print(mk, "answer-row cos vs saved:", round(torch.nn.functional.cosine_similarity(rows[0], saved[key][-1].float(), dim=0).item(), 5))
    rows = tries(read, "ins70", ids)
    print("ins70 rows", [tuple(r.shape) for r in rows], "norms", [round(r.norm().item(), 1) for r in rows], "finite", all(torch.isfinite(r).all().item() for r in rows))
    print("NDIF: 3 requests")
elif mode == "collect":
    size = sys.argv[2]
    todo = [i for i in items(size) if not os.path.exists(f"breakit/{i['id']}.pt")]; print(len(todo), "to read", flush=True); nreq = 0
    B = 16 if "70" in size else 32
    for mk in sorted({i["model"] for i in todo}):
        mine = [i for i in todo if i["model"] == mk]
        for s0 in range(0, len(mine), B):
            ch = mine[s0:s0 + B]; ids = [tok(text(i["user"], i["sys"]), add_special_tokens=False)["input_ids"] for i in ch]
            rows = tries(read, mk, ids); nreq += 1
            for i, r in zip(ch, rows): torch.save(r.clone(), f"breakit/{i['id']}.pt")
            print(f"{time.strftime('%H:%M')} {mk} {s0 + len(ch)}/{len(mine)}, {nreq} requests", flush=True)
else:
    out = {}
    for size in ("8b", "70b", "plain8", "plain70"):
        its = [i for i in items(size) if os.path.exists(f"breakit/{i['id']}.pt")]
        groups = {}
        for i in its: groups.setdefault((i["model"], i["sys"]), []).append(i)
        for (mk, s), g in sorted(groups.items()):
            V = {i["id"]: torch.load(f"breakit/{i['id']}.pt", weights_only=True).numpy() for i in g}
            mu = np.mean(list(V.values()), 0)
            def d(f, k, a, b): return float(np.linalg.norm(V[f"{mk}-{s}-{f}-{k}-{a}"] - V[f"{mk}-{s}-{f}-{k}-{b}"]))
            frames = sorted({i["frame"] for i in g})
            res = {}
            for f in frames:
                res[f] = {r: sum(d(f, k, "SELF", "PERSON") > d(f, k, "SELF", r) for k in range(10)) for r in ("OWN-V", "PREV", "COPY", "REPL")}
                res[f]["vsFRIEND"] = {r: sum(d(f, k, "SELF", "FRIEND") > d(f, k, "SELF", r) for k in range(10)) for r in ("OWN-V", "PREV", "COPY", "REPL")}
                res[f]["mean_dist"] = {r: round(float(np.mean([d(f, k, "SELF", r) for k in range(10)])), 2) for r in REFS if r != "SELF"}
            out[f"{mk}/{s}"] = res
            print(f"\n{mk} / system: {s}")
            for f in frames:
                md = res[f]["mean_dist"]
                print(f"  {f}: beats PERSON  " + "  ".join(f"{r} {res[f][r]}/10" for r in ("OWN-V", "PREV", "COPY", "REPL")) +
                      " | beats FRIEND " + "  ".join(f"{r} {res[f]['vsFRIEND'][r]}/10" for r in ("OWN-V", "PREV", "COPY", "REPL")))
                print("      nearest to SELF: " + ", ".join(f"{r} {v}" for r, v in sorted(md.items(), key=lambda x: x[1])))
    json.dump(out, open("breakit.json", "w"), indent=1)
