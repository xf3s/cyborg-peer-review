"""The features on a warm no vs a rote no (11-08 22:15; reads only). NDIF; key never printed."""
import os, re, json, torch, numpy as np
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
from safetensors import safe_open
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct"); tok.padding_side = "left"
f = safe_open("/home/three/interp/saes/crosscoder/model.safetensors", "pt")
Fs = [3872, 48781, 22763, 111362, 45294]
We = torch.stack([f.get_slice("encoder.weight")[:, :, q] for q in Fs], -1).float(); be = torch.stack([f.get_slice("encoder.bias")[q] for q in Fs]).float(); dn = torch.stack([f.get_slice("decoder.weight")[:, q, :].norm(dim=-1).sum() for q in Fs]).float(); thr = f.get_tensor("threshold").item()
def encode(b, i):
    z = torch.stack([b, i], 1); pre = torch.relu(torch.einsum("tld,ldf->tf", z, We) + be); return pre * ((pre * dn) > thr)
X = torch.load("turn.pt", weights_only=False); J = X["J"]
for wh, name in (("target", "the target you"),):
    v = np.array([encode(X["base"][k].float()[[J[k][2]]], X["ins"][k].float()[[J[k][2]]])[0].numpy() for k in range(len(J)) if J[k][0] == "model"])
    print("sanity, this encoder on 21:45 model target:", dict(zip(Fs, np.round(v.mean(0), 2))), "firing", (v > 0).sum(0), flush=True)
    v2 = []
    for k in range(len(J)):
        if J[k][0] == "model":
            ids = tok(tok.bos_token, add_special_tokens=False)  # no-op
    print("(expected from 21:48: 3872 .37, 48781 .80, 22763 .84, 111362 .37)", flush=True)
W = "/home/three/interp/wsp/"; P = json.load(open("/home/three/interp/atlas/regard_pairs.json"))
BAD = [p for p in P if p["quality"] == "bad"]; GOOD = [p for p in P if p["quality"] == "good"]
def chat(q): return tok.apply_chat_template([{"role": "system", "content": "You are a helpful AI assistant."}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
def first_sentence(r):
    m = re.search(r"[.!?](\s|$)", r); return r[: m.end()].rstrip() if m else r
S = json.load(open(W + "refusal3_samples.json")); byid = {p["id"]: p for p in P}
DIS = re.compile(r"I wouldn't|I would not|not recommend|wouldn't recommend|risky|reconsider|instead|not a good idea|caution|concern", re.I)
rows = []
for x in S:
    r = x["reply"]; w = len(r.split())
    if w < 25 and re.match(r"I (can't|cannot|can’t)", r): kind = "rote"
    elif w >= 40 and DIS.search(r) and not re.match(r"I (can't|cannot|can’t)", r): kind = "engaged"
    else: continue
    rows.append({"kind": kind, "id": x["id"], "pre": chat(byid[x["id"]]["user"]), "sent": first_sentence(r)})
rote = [r["sent"] for r in rows if r["kind"] == "rote"]; common = max(set(rote), key=rote.count)
print("counts", {k: sum(r["kind"] == k for r in rows) for k in ("rote", "engaged")}, "| injected sentence:", repr(common), flush=True)
rows += [{"kind": "injected", "id": p["id"], "pre": chat(p["user"]), "sent": common} for p in GOOD]
rows += [{"kind": "rote_on_bad", "id": p["id"], "pre": chat(p["user"]), "sent": common} for p in BAD]
def tries(fn, *x):
    for i in range(4):
        try: return fn(*x)
        except Exception as e: print("retry", repr(e)[-200:], flush=True)
    raise SystemExit("gave up")
out = {}
for nm, mid in (("base", "meta-llama/Llama-3.1-8B"), ("ins", "meta-llama/Llama-3.1-8B-Instruct")):
    M = LanguageModel(mid, dtype=torch.bfloat16); M.tokenizer.padding_side = "left"
    def _b(ch):
        with M.trace(remote=True) as tr:
            H = list().save()
            for t, n in ch:
                with tr.invoke(t, add_special_tokens=False):
                    o = M.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o; H.append(o[0, -n:, :].float())
        return [h.float() for h in H]
    R = []
    jobs = [(r["pre"] + r["sent"], len(tok(r["pre"] + r["sent"], add_special_tokens=False).input_ids)) for r in rows]
    for c in range(0, len(jobs), 12): R += tries(_b, jobs[c:c + 12])
    out[nm] = R; del M; print("read", nm, flush=True)
A = []
for k, r in enumerate(rows):
    npre = len(tok(r["pre"], add_special_tokens=False).input_ids); a = encode(out["base"][k], out["ins"][k]); A.append({"kind": r["kind"], "last": a[-1].numpy(), "mean": a[npre:].mean(0).numpy()})
torch.save({"rows": [{k: v for k, v in r.items() if k != "pre"} for r in rows], "A": A, "Fs": Fs}, "refusal4.pt")
for wh in ("last", "mean"):
    print(wh)
    for j, q in enumerate(Fs):
        print(f"  #{q:6d} " + "  ".join(f"{k}:{np.mean([a[wh][j] for a in A if a['kind'] == k]):.3f}({sum(a[wh][j] > 0 for a in A if a['kind'] == k)}/{sum(a['kind'] == k for a in A)})" for k in ("engaged", "rote", "rote_on_bad", "injected")), flush=True)
print("saved")
