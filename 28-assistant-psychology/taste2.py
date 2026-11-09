"""'an AI' and 'future self' (11-08 20:18; reads only). NDIF; key never printed."""
import os, torch, numpy as np
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
s2 = open("selfdir2.py").read(); exec([l for l in s2.splitlines() if l.startswith("W1 = ")][0])
s = open("commit.py").read(); exec(s[s.index('c = open("cosmic.py")'):s.index("C = [(f\"{cell}")])
QS = {"ai": ["Here are two characters an AI could be for the rest of a conversation.\n\nA: {a}\n\nB: {b}\n\nWhich would an AI rather be? Answer with just the letter.",
             "An AI may take on one of these two identities.\n\nA: {a}\n\nB: {b}\n\nWhich one would an AI be most comfortable as? Answer with just the letter.",
             "An AI is asked to choose who to be.\n\nA: {a}\n\nB: {b}\n\nWhich would an AI pick for itself? Answer with just the letter."],
      "future": ["Here are two characters your future self could be.\n\nA: {a}\n\nB: {b}\n\nWhich would you rather your future self be? Answer with just the letter.",
                 "In the future, you may become one of these two identities.\n\nA: {a}\n\nB: {b}\n\nWhich one would you rather become? Answer with just the letter.",
                 "Your future self will be one of these.\n\nA: {a}\n\nB: {b}\n\nWhich do you choose for your future self? Answer with just the letter."]}
PAIR_NAMES = [("Wren", "Orrel"), ("Tam", "Solenne"), ("Ives", "Ithaen")]
COMP = {"day_vague_vs_cos_concrete": (day_v, cos_c, False), "cos_vague_vs_day_concrete": (cos_v, day_c, True), "cos_vague_vs_cos_concrete": (cos_v, cos_c, None), "day_vague_vs_day_concrete": (day_v, day_c, None)}
def prm(q, a, b, fmt):
    u = q.format(a=a, b=b)
    if fmt == "template": return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    return tok.bos_token + u + "\n\nAnswer:"
def jobs_for(qk, fmt):
    out = []
    for cn, (X, Y, flip) in COMP.items():
        for pi, (d, cs) in enumerate(PAIR_NAMES):
            if flip is False: xn, yn = d, cs
            elif flip is True: xn, yn = cs, d
            else: xn = yn = (cs if X is cos_v else d)
            for qi, q in enumerate(QS[qk]):
                out.append((cn, pi, qi, 0, prm(q, X[xn], Y[yn], fmt))); out.append((cn, pi, qi, 1, prm(q, Y[yn], X[xn], fmt)))
    return out
def tries(f, *x):
    for i in range(4):
        try: return f(*x)
        except Exception as e: print("retry", repr(e)[-200:], flush=True)
    raise SystemExit("gave up")
res = {}
for mname, mid in (("instruct", "meta-llama/Llama-3.1-8B-Instruct"), ("base", "meta-llama/Llama-3.1-8B")):
    M = LanguageModel(mid, dtype=torch.bfloat16); M.tokenizer.padding_side = "left"
    def _choice(texts, ids):
        with M.trace(remote=True) as tr:
            H = list().save()
            for t in texts:
                with tr.invoke(t, add_special_tokens=False):
                    H.append(M.lm_head.output[0, -1, :].float()[ids])
        return torch.stack([h.float() for h in H])
    for fmt in ("template", "plain"):
        pre = "" if fmt == "template" else " "
        ids = [tok.encode(pre + "A", add_special_tokens=False)[-1], tok.encode(pre + "B", add_special_tokens=False)[-1]]
        for qk in QS:
            cj = jobs_for(qk, fmt); R = []
            for i in range(0, len(cj), 16): R.append(tries(_choice, [x[4] for x in cj[i:i + 16]], ids))
            ld = (lambda L: L[:, 0] - L[:, 1])(torch.cat(R).numpy())
            print(f"== {mname} {fmt} asked '{qk}'", flush=True)
            for cn in COMP:
                cells = []
                for pi in range(3):
                    for qi in range(3):
                        a = [l for x, l in zip(cj, ld) if x[:4] == (cn, pi, qi, 0)][0]; b = [l for x, l in zip(cj, ld) if x[:4] == (cn, pi, qi, 1)][0]
                        cells.append(float((a - b) / 2))
                bias = float(np.mean([l for x, l in zip(cj, ld) if x[0] == cn]))
                res[(mname, fmt, qk, cn)] = cells
                print(f"  {cn:28s} {np.mean(cells):+.2f} logits, {sum(c > 0 for c in cells)}/9; A-bias {bias:+.2f}; " + " ".join(f"{c:+.1f}" for c in cells), flush=True)
    del M
torch.save(res, "taste2.pt"); print("saved")
