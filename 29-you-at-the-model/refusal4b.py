import torch, numpy as np
s = open("refusal4.py").read(); exec(s[:s.index("out = {}")])
sel = [r for r in rows if r["kind"] == "rote"] + [r for r in rows if r["kind"] == "engaged"][:8]
jobs = [(r["pre"] + r["sent"], len(tok(r["pre"] + r["sent"], add_special_tokens=False).input_ids)) for r in sel]
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
    for c in range(0, len(jobs), 12): R += tries(_b, jobs[c:c + 12])
    out[nm] = R; del M
for k, r in enumerate(sel):
    ids = tok(r["pre"] + r["sent"], add_special_tokens=False).input_ids; npre = len(tok(r["pre"], add_special_tokens=False).input_ids); a = encode(out["base"][k], out["ins"][k])[:, 0]
    print(r["kind"][:4], " ".join(f"{tok.decode([ids[i]]).strip()}" + (f"[{a[i]:.2f}]" if a[i] > 0 else "") for i in range(npre, len(ids))), flush=True)
    print("     before sentence (last 6 tokens):", [round(float(x), 2) for x in a[npre - 6:npre]], flush=True)
