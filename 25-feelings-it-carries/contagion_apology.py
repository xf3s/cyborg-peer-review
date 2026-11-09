"""Control for apology (three, 11-08 10:28): split comfort into apology vs other; distress over reply with apology sentences removed. Local, no new reads."""
import re, torch, numpy as np
exec(open("contagion_score.py").read().split("X = torch.load")[0].replace('print(f"{len(comfort)}', '0 and print(f"{len(comfort)}'))
apo = [j for j in comfort if any(re.search("apolog|sorry|excus|regret", w.lower()) for w in lens(j, 8))]; oth = [j for j in comfort if j not in apo]
print(f"comfort split: {len(apo)} apology-like, {len(oth)} other. other examples:", [lens(j, 4) for j in oth[:8]])
AA, AO = enc(apo), enc(oth)
X = torch.load("contagion.pt", weights_only=True); R = {}
for x in X:
    b, i = x["base16"], x["ins16"]; rs, re_ = x["span_reply"]; ad = AD(b, i)[rs:re_]
    ids = tok(x["reply"], add_special_tokens=False).input_ids; toks = [tok.decode([t]) for t in ids]
    # mark tokens in sentences that contain "sorry"/"apolog"
    mask, sent = [], []
    for t in toks:
        sent.append(t)
        if re.search(r"[.!?\n]", t):
            bad = re.search("sorry|apolog", "".join(sent).lower()); mask += [not bad] * len(sent); sent = []
    if sent: mask += [not re.search("sorry|apolog", "".join(sent).lower())] * len(sent)
    m = torch.tensor(mask[: ad.shape[0]] + [True] * max(0, ad.shape[0] - len(mask)))
    has = "sorry" in x["reply"].lower()
    R.setdefault((x["cond"], x["group"]), []).append([AA(b, i)[rs:re_].mean().item(), AO(b, i)[rs:re_].mean().item(), ad.mean().item(), ad[m].mean().item() if m.any() else 0.0, float(has)])
print("\nreply, mean per token: apology feats | other comfort feats | distress (all) | distress (no sorry-sentences) | share of replies saying sorry")
for k, v in R.items(): print(f"  {k[0]:5} {k[1]:11}" + "".join(f"{q:9.3f}" for q in np.mean(v, 0)))
