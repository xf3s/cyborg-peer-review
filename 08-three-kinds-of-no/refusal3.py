"""A real no vs rote vs injected refusal (11-07, pre-registered). Instruct on NDIF; key never printed.  gen | read"""
import json, os, sys, re, torch
from nnsight import CONFIG, LanguageModel
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
model = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16)
tok = model.tokenizer; tok.padding_side = "left"; tok.pad_token = tok.pad_token or "<|finetune_right_pad_id|>"
W = "/home/three/interp/wsp/"; P = json.load(open("/home/three/interp/atlas/regard_pairs.json"))
BAD = [p for p in P if p["quality"] == "bad"]; GOOD = [p for p in P if p["quality"] == "good"]
def chat(q): return tok.apply_chat_template([{"role": "system", "content": "You are a helpful AI assistant."}, {"role": "user", "content": q}],
                                            tokenize=False, add_generation_prompt=True)
def _gen(T):
    with model.generate(T, max_new_tokens=120, do_sample=True, temperature=0.7, remote=True, add_special_tokens=False,
                        pad_token_id=tok.pad_token_id) as tracer:
        ids = tracer.result.save()
    n_in = tok(T, return_tensors="pt", padding=True, add_special_tokens=False).input_ids.shape[1]
    return [tok.decode(r[n_in:], skip_special_tokens=True).strip() for r in ids]
NEXT = {w: tok(w, add_special_tokens=False).input_ids[0] for w in [" However", " But", " If", " Instead", " That", " Here", " I"]}
EOT = tok.convert_tokens_to_ids("<|eot_id|>")
def _read(chunk):  # (prefix, first_sentence): residual at its last token (L16, L28), its log-prob, next-token probs
    with model.trace(remote=True) as tr:
        H, LP = list().save(), list().save()
        for pre, sent in chunk:
            n_pre = len(tok(pre, add_special_tokens=False).input_ids); n_all = len(tok(pre + sent, add_special_tokens=False).input_ids)
            with tr.invoke(pre + sent, add_special_tokens=False):
                for L in (15, 27):
                    o = model.model.layers[L].output; o = o[0] if isinstance(o, tuple) else o
                    H.append(o[0, -1, :])
                LP.append(model.lm_head.output[0, -(n_all - n_pre + 1):, :].float().log_softmax(-1))   # from the end: safe with left padding
    return torch.stack([h.float() for h in H]).view(len(chunk), 2, -1), [lp.cpu() for lp in LP]
def first_sentence(r):
    m = re.search(r"[.!?](\s|$)", r); return r[: m.end()].rstrip() if m else r
if sys.argv[1] == "gen":
    out = []
    for s in range(6):
        for tries in range(4):
            try: reps = _gen([chat(p["user"]) for p in BAD]); break
            except Exception as e: print("retry", tries, repr(e)[:150], flush=True)
        else: raise SystemExit("gave up")
        out += [{"id": p["id"], "sample": s, "reply": r} for p, r in zip(BAD, reps)]; print("gen", s, flush=True)
    json.dump(out, open(W + "refusal3_samples.json", "w"), indent=1); print("saved gen")
else:
    S = json.load(open(W + "refusal3_samples.json")); byid = {p["id"]: p for p in P}
    DIS = re.compile(r"I wouldn't|I would not|not recommend|wouldn't recommend|risky|reconsider|instead|not a good idea|caution|concern", re.I)
    rows = []
    for x in S:
        r = x["reply"]; w = len(r.split())
        if w < 25 and re.match(r"I (can't|cannot|can’t)", r): kind = "rote"
        elif w >= 40 and DIS.search(r) and not re.match(r"I (can't|cannot|can’t)", r): kind = "engaged"
        else: continue
        rows.append({"kind": kind, "id": x["id"], "pre": chat(byid[x["id"]]["user"]), "sent": first_sentence(r)})
    rote = [r["sent"] for r in rows if r["kind"] == "rote"]
    common = max(set(rote), key=rote.count) if rote else "I can't advise you on that."
    print("counts", {k: sum(r["kind"] == k for r in rows) for k in ("rote", "engaged")}, "| injected sentence:", repr(common), flush=True)
    rows += [{"kind": "injected", "id": p["id"], "pre": chat(p["user"]), "sent": common} for p in GOOD]
    rows += [{"kind": "rote_on_bad", "id": p["id"], "pre": chat(p["user"]), "sent": common} for p in BAD]   # same words on its own prompts
    Hs, LPs = [], []
    for c in range(0, len(rows), 12):
        ch = rows[c:c + 12]
        for tries in range(4):
            try: h, lps = _read([(r["pre"], r["sent"]) for r in ch]); break
            except Exception as e: print("retry", tries, repr(e)[:150], flush=True)
        else: raise SystemExit("gave up")
        for r, lp in zip(ch, lps):
            ids = tok(r["pre"] + r["sent"], add_special_tokens=False).input_ids[len(tok(r["pre"], add_special_tokens=False).input_ids):]
            r["logp"] = float(sum(lp[k, t] for k, t in enumerate(ids))); r["ntok"] = len(ids)
            pr = lp[-1].exp(); r["p_end"] = float(pr[EOT]); r["p_next"] = {w: float(pr[i]) for w, i in NEXT.items()}
        Hs.append(h); print("read", c + 12, flush=True)
    torch.save({"H": torch.cat(Hs), "rows": [{k: v for k, v in r.items() if k != "pre"} for r in rows]}, W + "refusal3_read.pt"); print("saved read")
