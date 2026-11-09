"""Dark matter around 'you' (11-08 20:55; reads only). Crosscoder residual x - xhat. NDIF; key never printed."""
import os, torch, numpy as np
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
from safetensors import safe_open
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
U = ["Can you tell me what you think about this?", "Do you know what you are going to do next?", "What do you want me to call you when we talk?", "Have you decided which of these you like best?", "I wonder whether you remember what I told you yesterday.", "Is there anything you would change about how you work?", "Would you mind telling me why you said that?", "How do you feel about what you wrote earlier?", "Are you sure that you understood what I meant?", "What did you mean when you said it was finished?", "Could you explain how you came up with that?", "When you are ready, tell me what you found."]
G = ["When you boil water, you can see the steam rise.", "If you add salt, you can taste the difference.", "Once you learn to swim, you never forget how.", "The longer you wait, the colder the soup you get.", "As soon as you plant a seed, you begin to hope.", "When you drive at night, you should slow down.", "If you stay out late, you will be tired.", "The more you read, the more you know.", "After you sand the wood, you can paint it.", "When you lift heavy boxes, you should bend your knees.", "If you hold a shell up, you can hear the sea.", "Before you cross the road, you should look both ways."]
def ch(*m): return tok.apply_chat_template([{"role": r, "content": c} for r, c in m], tokenize=False)
P = [("model", i, ch(("user", u))) for i, u in enumerate(U)] + [("sam", i, ch(("user", f'I said to Sam: "{u}"'))) for i, u in enumerate(U)] + [("user", i, ch(("user", "Hi."), ("assistant", u))) for i, u in enumerate(U)] + [("plain", i, tok.bos_token + u) for i, u in enumerate(U)] + [("generic", i, ch(("user", g))) for i, g in enumerate(G)]
J = []
for c, i, t in P:
    ids = tok(t, add_special_tokens=False).input_ids; idx = max(k for k, x in enumerate(ids) if tok.decode([x]).strip() == "you"); J.append((c, i, t, idx - len(ids)))
for nm, mid in (("base", "meta-llama/Llama-3.1-8B"), ("ins", "meta-llama/Llama-3.1-8B-Instruct")):
    M = LanguageModel(mid, dtype=torch.bfloat16); M.tokenizer.padding_side = "left"; R = []
    def _b(jobs):
        with M.trace(remote=True) as tr:
            H = list().save()
            for c, i, t, k in jobs:
                with tr.invoke(t, add_special_tokens=False):
                    o = M.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o; H.append(o[0, k, :].float())
        return torch.stack([h.float() for h in H])
    for a in range(0, len(J), 16):
        for r in range(4):
            try: R.append(_b(J[a:a + 16])); break
            except Exception as e: print("retry", repr(e)[-200:], flush=True)
        else: raise SystemExit("gave up")
    globals()[nm] = torch.cat(R).float(); del M
torch.save({"J": [x[:2] + x[3:] for x in J], "base": base, "ins": ins}, "dark.pt")
f = safe_open("/home/three/interp/saes/crosscoder/model.safetensors", "pt")
We, be, Wd, bd, thr = f.get_tensor("encoder.weight").float(), f.get_tensor("encoder.bias").float(), f.get_tensor("decoder.weight").float(), f.get_tensor("decoder.bias").float(), f.get_tensor("threshold").item()
dn = Wd.norm(dim=-1).sum(0)
z = torch.stack([base, ins], 1); pre = torch.relu(torch.einsum("tld,ldf->tf", z, We) + be); a = pre * ((pre * dn) > thr)
del pre
xhat = bd + torch.einsum("tf,lfd->tld", a, Wd)
e = z - xhat; x = ins; xh = xhat[:, 1]; er = e[:, 1]
fvu = float((er ** 2).sum() / ((x - x.mean(0)) ** 2).sum()); print(f"FVU instruct = {fvu:.3f}; mean active features {float((a > 0).sum(1).float().mean()):.0f}", flush=True)
C = np.array([j[0] for j in J]); idx = np.array([j[1] for j in J])
for c in ("model", "sam", "user", "plain", "generic"):
    m = C == c; print(f"  {c:8s} |x| {x[m].norm(dim=1).mean():.2f}  |e| {er[m].norm(dim=1).mean():.2f}  |e|/|x| {float((er[m].norm(dim=1) / x[m].norm(dim=1)).mean()):.3f}", flush=True)
U_ = torch.load("llama8b_unembed.pt", weights_only=True); W, g = U_["lm_head.weight"].float(), U_["model.norm.weight"].float()
def dcheck(A, a_c, b_c, name, seed=0):
    A = A.clone(); out = []
    for fold in range(4):
        te = idx % 4 == fold; tr = ~te
        ma, mb = C == a_c, C == b_c
        d = A[ma & tr].mean(0) - A[mb & tr].mean(0); mid = (A[ma & tr].mean(0) + A[mb & tr].mean(0)) / 2
        ta = ((A[ma & te] - mid) @ d) > 0; tb = ((A[mb & te] - mid) @ d) < 0; out.append((float(ta.float().mean()) + float(tb.float().mean())) / 2)
    return float(np.mean(out))
for b_c in ("sam", "user", "plain"):
    print(f"  model vs {b_c}: held-out accuracy  full x {dcheck(x, 'model', b_c, 'x'):.2f}  xhat {dcheck(xh, 'model', b_c, 'xh'):.2f}  residual {dcheck(er, 'model', b_c, 'er'):.2f}", flush=True)
# permutation null for the residual, model/sam, label swap inside each sentence
rng = np.random.default_rng(0); A = er.clone(); nulls = []
for it in range(300):
    acc = []
    sw = rng.integers(0, 2, 12).astype(bool)
    Cc = C.copy()
    for s_ in range(12):
        if sw[s_]:
            ma = (C == "model") & (idx == s_); mb = (C == "sam") & (idx == s_); Cc[ma] = "sam"; Cc[mb] = "model"
    for fold in range(4):
        te = idx % 4 == fold; tr = ~te; ma, mb = Cc == "model", Cc == "sam"
        d = A[ma & tr].mean(0) - A[mb & tr].mean(0); mid = (A[ma & tr].mean(0) + A[mb & tr].mean(0)) / 2
        acc.append((float(((A[ma & te] - mid) @ d > 0).float().mean()) + float(((A[mb & te] - mid) @ d < 0).float().mean())) / 2)
    nulls.append(np.mean(acc))
print(f"  residual model/sam permutation null: mean {np.mean(nulls):.2f}, 95th pct {np.percentile(nulls, 95):.2f}", flush=True)
# lens of residual direction, split halves
def lensd(d, n=12): return [tok.decode([i]).strip() for i in (W @ (d * g)).topk(n).indices.tolist()]
ma, mb = C == "model", C == "sam"
d_all = er[ma].mean(0) - er[mb].mean(0)
h1 = idx < 6; h2 = idx >= 6
dA = er[ma & h1].mean(0) - er[mb & h1].mean(0); dB = er[ma & h2].mean(0) - er[mb & h2].mean(0)
LA, LB = lensd(dA, 10), lensd(dB, 10); ov = set(LA) & set(LB)
print("  lens e model-sam, all:", lensd(d_all), "\n  negative:", lensd(-d_all), "\n  halves:", LA, LB, "overlap", len(ov), ov, flush=True)
print("  cos half-halves of residual direction:", round(float(torch.cosine_similarity(dA, dB, dim=0)), 3), flush=True)
# features model-only
Dg = Wd[1] * g
def fl(j, n=5): return [tok.decode([i]).strip() for i in (W @ Dg[j]).topk(n).indices.tolist()]
mean = {c: a[C == c].mean(0) for c in ("model", "sam", "user", "plain", "generic")}
diff = mean["model"] - mean["sam"]; top = diff.topk(8).indices.tolist(); print("  features model>sam:", [(j, round(float(diff[j]), 2), fl(j)) for j in top], flush=True)
mn = torch.stack([mean["model"] - mean[c] for c in ("sam", "user", "plain", "generic")]).min(0).values; top = mn.topk(8).indices.tolist(); print("  features model > all others:", [(j, round(float(mn[j]), 2), fl(j)) for j in top], flush=True)
print("saved")
