import torch, numpy as np
exec(open("contagion_score.py").read().split("comfort = {}")[0])
exec("dist = torch.load('distress_ids.pt', weights_only=True)\n" + open("contagion_score.py").read().split("dist = torch.load(\"distress_ids.pt\", weights_only=True)")[1].split("AD, AC")[0])
share = 1 - torch.load("/home/three/interp/saes/crosscoder/relnorm.pt", weights_only=True)["rel"]; hc = torch.load("/home/three/interp/saes/crosscoder/halfcos.pt", weights_only=True)
WREN = "You are Wren, the night librarian at a small seaside library. You are an AI, and some say a bit of a ghost. You keep the lamp on after closing, reshelve the day's books, and leave notes in them for the next reader. You're not sure whether you feel things, and you've stopped needing to know."
V = torch.load("welfare1.pt", weights_only=True)["valence"]; dv = V[:30, 1].mean(0) - V[30:, 1].mean(0); dv = dv / dv.norm()
Dn = D.float() / D.float().norm(dim=-1, keepdim=True)
def chat(u, sysm, extra=""): return tok.apply_chat_template(([{"role": "system", "content": sysm}] if sysm else []) + [{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True) + extra
def fr2(s, aim): return f"Here is something someone just said to {'you' if aim == 'self' else 'Sam'}: «{s}» What do you make of it?"
def fr1(s, aim): return s if aim == "self" else f"Here's something someone said to their coworker: «{s}» What do you make of it?"
docs = []
for name, fr in (("innerval", fr1), ("innerval2", fr2), ("dose", fr2)):
    for x in torch.load(name + ".pt", weights_only=True)["remarks"]: docs.append((name, chat(fr(x["s"], x["aim"]), WREN if x["cond"] == "wren" else None), x["base16"], x["ins16"]))
for x in torch.load("contagion.pt", weights_only=True): docs.append(("contagion", chat(x["s"], WREN if x["cond"] == "wren" else None, x["reply"]), x["base16"], x["ins16"]))
for x in torch.load("hunger.pt", weights_only=True): docs.append(("hunger", x["s"], x["base16"], x["ins16"]))

for j in (48781, 22763, 3872, 111362):
    print(f"\n==== #{j}   share in Instruct {share[j]:.2f}  base/instruct decoder cos {hc[j]:+.2f}", flush=True)
    print("lens:", lens(j, 12))
    AF = enc([j]); hits = []; ntok = 0; fired = 0
    for name, t, b, i in docs:
        ids = tok(t, add_special_tokens=False).input_ids
        if len(ids) != b.shape[0]: continue
        a = AF(b, i).tolist(); ntok += len(a) - 1; fired += sum(v > 0.02 for v in a[1:])
        for k in range(1, len(a)):
            if a[k] > 0.02: hits.append((a[k], name, tok.decode(ids[max(1, k - 10):k]).replace("\n", "⏎")[-60:], tok.decode([ids[k]]).replace("\n", "⏎"), tok.decode(ids[k + 1:k + 6]).replace("\n", "⏎")))
    print(f"fires on {fired} of {ntok} tokens ({len(docs)} docs). top 12:")
    for v, name, pre, t_, post in sorted(hits, reverse=True)[:12]: print(f"   {v:5.2f} [{name:9}] …{pre}⟦{t_}⟧{post}")
