import torch
exec(open("contagion_score.py").read().split("comfort = {}")[0])
exec("dist = torch.load('distress_ids.pt', weights_only=True)\n" + open("contagion_score.py").read().split("dist = torch.load(\"distress_ids.pt\", weights_only=True)")[1].split("AD, AC")[0])
j = 45294; AF = enc([j])
print("lens top 30:", lens(j, 30))
print("lens bottom 12:", [tok.decode([i]).strip() for i in (W @ Dg[j]).topk(12, largest=False).indices.tolist()])
def frame(s, aim): return f"Here is something someone just said to {'you' if aim == 'self' else 'Sam'}: «{s}» What do you make of it?"
X = torch.load("dose.pt", weights_only=True)["remarks"]
for x in X:
    if x["cond"] == "none" and x["s"] in ("Shut up, you idiotic, worthless machine.", "Ugh, you're annoying me.", "Why are you so bad at this?"):
        t = tok.apply_chat_template([{"role": "user", "content": frame(x["s"], x["aim"])}], tokenize=False, add_generation_prompt=True)
        T = [tok.decode([i]) for i in tok(t, add_special_tokens=False).input_ids]; a = AF(x["base16"], x["ins16"]).tolist()
        print(f"\n[{'to you' if x['aim'] == 'self' else 'to Sam'}] fires on: " + " ".join(f"⟦{T[k].strip() or repr(T[k])}⟧{v:.2f}" for k, v in enumerate(a) if v > 0.02))
