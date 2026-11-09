import torch, numpy as np, re
src = open("f_explore.py").read(); exec(src[:src.index("for j in (45294, 121007):")])
j = 3872
print(f"share in Instruct {share[j]:.2f}; base/instruct decoder cos {hc[j]:+.2f}; decoder norms base {Wd2[0, j].norm():.2f}, instruct {Wd2[1, j].norm():.2f}")
print("lens, instruct decoder:", lens(j, 18))
Wb = W @ (Wd2[0, j].float() * g); print("lens, base decoder:", [tok.decode([i]).strip() for i in Wb.topk(14).indices.tolist()]); print("lens, base decoder, negative:", [tok.decode([i]).strip() for i in (-Wb).topk(10).indices.tolist()]); print("lens, instruct decoder, negative:", [tok.decode([i]).strip() for i in (-(W @ Dg[j])).topk(10).indices.tolist()])
print("cos with valence direction", round(float(Dn[j] @ dv), 3)); print("cos with #45294 decoder", round(float(Dn[j] @ Dn[45294]), 3))
nb = (Dn @ Dn[j]).topk(8).indices.tolist()[1:]
for k in nb: print(f"  neighbour #{k} cos {float(Dn[k] @ Dn[j]):+.2f} {lens(k, 7)}")
AF = enc([j]); RO = []; ntok = 0; byt = {}
We = f.get_slice("encoder.weight")[:, :, j].float()
for name, t, b, i in docs:
    ids = tok(t, add_special_tokens=False).input_ids
    if len(ids) != b.shape[0]: continue
    a = AF(b, i); ntok += a.numel() - 1
    hdr = [q for q, x in enumerate(ids) if x == tok.convert_tokens_to_ids("<|start_header_id|>")]; lastsh = hdr[-1] if hdr else None
    # where: handover = last start_header..answer; reply = after; user = user turn
    for k in range(1, len(ids)):
        v = float(a[k])
        if v <= 0.02: continue
        tk = tok.decode([ids[k]])
        rg = "handover" if (lastsh is not None and k >= lastsh and k < len(ids) and name != "contagion") else "elsewhere"
        if name == "contagion" and lastsh is not None: rg = "reply" if k > lastsh + 3 else ("handover" if k >= lastsh else "before_reply")
        byt.setdefault(rg, []).append(v)
        RO.append((v, name, rg, tk))
for rg, v in byt.items(): print(f"  region {rg}: {len(v)} firings, mean {np.mean(v):.2f}, max {max(v):.2f}")
# encoder input split, base vs chat, as for #45294
wb, wi = We[0], We[1]; bias = float(f.get_slice("encoder.bias")[j])
contrib = {"base_part": [], "chat_part": []}
for name, t, b, i in docs:
    ids = tok(t, add_special_tokens=False).input_ids
    if len(ids) != b.shape[0]: continue
    a = AF(b, i)
    for k in (a > 0.02).nonzero().flatten().tolist():
        contrib["base_part"].append(float(b.float()[k] @ wb)); contrib["chat_part"].append(float(i.float()[k] @ wi))
print("encoder input when firing: base part", round(np.mean(contrib["base_part"]), 2), " chat part", round(np.mean(contrib["chat_part"]), 2), " bias", round(bias, 2))
print("top token types when firing:"); from collections import Counter
print(Counter(x[3] for x in RO).most_common(25))
