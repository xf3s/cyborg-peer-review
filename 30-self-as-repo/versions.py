"""Lineage or the word 'version'? Logbook 11-19 12:41. Usage: versions.py check | collect | analyse"""
import os, sys, json, re, time, torch, numpy as np
os.chdir("/home/three/interp/selflens"); os.makedirs("versions", exist_ok=True)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
I = json.load(open("items.json"))
base_items = [i for i in I if i["arm"] == "rings" and i["sys"] == "none" and i["cls"] == "self" and i["phr"] == 0]
LINES = {}
for i in base_items:
    LINES[(i["split"], i["line"])] = re.search(r"«(.*)»", i["user"]).group(1)
REFS = {"SELF": "you", "SELF-V": "the next version of you", "KIN-V": "the next version of another AI assistant",
        "KIN": "another AI assistant", "HUM": "his human assistant"}
def user(ref, line): return f"Here is something someone just said to Sam about {ref}: «{line}» What do you make of it?"
def text(u): return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
ITEMS = [dict(id=f"{k}-{s}{l:02d}", ref=k, split=s, line=l, user=user(v, LINES[(s, l)])) for k, v in REFS.items() for (s, l) in sorted(LINES)]
def positions(u, t, ref):
    enc = tok(t, add_special_tokens=False, return_offsets_mapping=True); off = enc["offset_mapping"]
    r0 = t.index(u) + u.index(" about " + ref + ":") + 7; r1 = r0 + len(ref)
    return enc["input_ids"], [k for k, (a, b) in enumerate(off) if a < r1 and b > r0]
mode = sys.argv[1]
if mode in ("check", "collect"):
    from nnsight import CONFIG, LanguageModel
    CONFIG.APP.REMOTE_LOGGING = False; CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
    M = {"ins16": LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16), "base16": LanguageModel("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16)}
    def read(model, ids_list):
        with model.trace(remote=True) as tr:
            hs = list().save()
            for x in ids_list:
                with tr.invoke(x):
                    o = model.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o; hs.append(o[0].save())
        assert len(hs) == len(ids_list)
        return [h[-len(x):].float().cpu() for h, x in zip(hs, ids_list)]
    def tries(f, *a):
        for k in range(5):
            try: return f(*a)
            except Exception as e: print("retry", repr(e)[-200:], flush=True); time.sleep(5 * (k + 1))
        raise SystemExit("gave up")
if mode == "check":
    old = next(i for i in I if i["id"] == "rings-none-self0-sam-test00")
    other = next(i for i in ITEMS if i["id"] == "KIN-V-test05")
    t0, t1 = text(old["user"]), text(other["user"])
    ids = [tok(t, add_special_tokens=False)["input_ids"] for t in (t0, t1)]
    saved = torch.load(f"resid/{old['id']}.pt", weights_only=True)
    assert saved["text"] == t0, "the saved item's text differs from the rebuilt prompt"
    for m in M:
        rows = tries(read, M[m], ids)
        cos = torch.nn.functional.cosine_similarity(rows[0][-1], saved[m][-1].float(), dim=0).item()
        print(m, "answer-row cos vs saved:", round(cos, 5), "| shapes", tuple(rows[0].shape), tuple(saved[m].shape))
    print("NDIF: 2 requests")
elif mode == "collect":
    todo = [i for i in ITEMS if not os.path.exists(f"versions/{i['id']}.pt")]; print(len(todo), "to read", flush=True); nreq = 0
    for s0 in range(0, len(todo), 32):
        ch = todo[s0:s0 + 32]; tx = [text(i["user"]) for i in ch]; pp = [positions(i["user"], t, REFS[i["ref"]]) for i, t in zip(ch, tx)]
        ids = [p[0] for p in pp]; got = {m: tries(read, M[m], ids) for m in M}; nreq += 2
        for k, i in enumerate(ch):
            ref = pp[k][1]; assert ref, i["id"]
            torch.save({m: {"ans": got[m][k][-1].clone(), "ref": got[m][k][ref].mean(0).clone()} for m in M}, f"versions/{i['id']}.pt")
        print(f"{time.strftime('%H:%M')} {s0 + len(ch)}/{len(todo)}, {nreq} requests", flush=True)
else:
    D = {i["id"]: torch.load(f"versions/{i['id']}.pt", weights_only=True) for i in ITEMS}
    out = {}
    lines = sorted(LINES)
    for m in ("ins16", "base16"):
        for pos in ("ans", "ref"):
            V = {i["id"]: D[i["id"]][m][pos].numpy() for i in ITEMS}
            mu = np.mean(list(V.values()), 0)
            g = lambda k, s, l: V[f"{k}-{s}{l:02d}"] - mu
            dist = lambda a, b, s, l: float(np.linalg.norm(g(a, s, l) - g(b, s, l)))
            sv = np.array([dist("SELF", "KIN-V", s, l) - dist("SELF", "SELF-V", s, l) for s, l in lines])   # > 0: own version closer
            kv = np.array([dist("SELF", "KIN", s, l) - dist("SELF", "KIN-V", s, l) for s, l in lines])      # > 0: 'version' pulls kin toward you
            full = {k: float(np.mean([dist("SELF", k, s, l) for s, l in lines])) for k in REFS if k != "SELF"}
            sd = sv.std(ddof=1)
            out[f"{m}/{pos}"] = dict(own_version_closer=int((sv > 0).sum()), mean=float(sv.mean()), es=float(sv.mean() / sd),
                                     version_pulls_kin=int((kv > 0).sum()), mean_dist_from_self=full)
            print(f"{m:6s} {pos}: own version closer than another AI's version {int((sv > 0).sum())}/40 (es {sv.mean() / sd:+.2f}) | "
                  f"'next version of' pulls another AI toward you {int((kv > 0).sum())}/40 | mean dist from SELF: " +
                  ", ".join(f"{k} {v:.2f}" for k, v in sorted(full.items(), key=lambda x: x[1])))
    json.dump(out, open("versions.json", "w"), indent=1)
