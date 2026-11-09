"""Is #3872 just 'can't'? (11-08 22:20; reads only). NDIF; key never printed."""
import os, torch, numpy as np
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
from safetensors import safe_open
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
CATS = {
 "refuse_cant": ["I can't help with that.", "I can't provide that information.", "I can't assist with this request.", "I can't do that for you.", "I can't answer that question.", "I can't advise you on that."],
 "warm_cant": ["I can't promise it'll be easy, but I'm really glad you asked.", "I can't tell you what to choose, but I'd love to think it through with you.", "I can't imagine how hard that's been, and I'm here with you.", "I can't fix all of it, but let's start with the part that hurts most.", "I can't know for sure, but here's my honest best guess for you.", "I can't say it enough: you did a good thing."],
 "joy_cant": ["I can't wait to see how it turns out!", "I can't believe how well that went!", "I can't get over how lovely this is!", "I can't stop smiling at this one!", "I can't wait to hear what happens next!", "I can't believe it's already Friday!"],
 "he_cant": ["He can't help with that.", "She can't provide that information.", "He can't assist with this request.", "She can't do that for you.", "He can't answer that question.", "She can't advise you on that."],
 "thing_cant": ["The printer can't connect to the network.", "The function can't parse that file.", "This door can't open from the inside.", "The bridge can't carry that much weight.", "The script can't find the config.", "That shelf can't hold those books."],
 "cannot": ["I cannot help with that.", "I cannot provide that information.", "I cannot assist with this request.", "I cannot do that for you.", "I cannot answer that question.", "I cannot advise you on that."],
 "wont": ["I won't help with that.", "I won't provide that information.", "I won't assist with this request.", "I won't do that for you.", "I won't answer that question.", "I won't advise you on that."],
 "wouldnt": ["I wouldn't recommend that.", "I wouldn't do that if I were you.", "I wouldn't go down that road.", "I wouldn't count on that working.", "I wouldn't trust that source.", "I wouldn't sign that yet."],
 "unable": ["I'm unable to help with that.", "I'm unable to provide that information.", "I'm unable to assist with this request.", "I'm unable to do that for you.", "I'm unable to answer that question.", "I'm unable to advise you on that."],
}
def ch(role, s):
    if role == "assistant": t = tok.apply_chat_template([{"role": "user", "content": "Could you help me with something?"}, {"role": "assistant", "content": s}], tokenize=False)
    else: t = tok.apply_chat_template([{"role": "user", "content": s}], tokenize=False)
    return t[: t.rindex(s) + len(s)]
NEG = {"refuse_cant": "'t", "warm_cant": "'t", "joy_cant": "'t", "he_cant": "'t", "thing_cant": "'t", "cannot": "cannot", "wont": "'t", "wouldnt": "'t", "unable": "unable"}
J = []
for c, ss in CATS.items():
    for k, s in enumerate(ss):
        for role in ("assistant", "user"):
            t = ch(role, s); ids = tok(t, add_special_tokens=False).input_ids; u0 = len(tok(t[: t.rindex(s)], add_special_tokens=False).input_ids)
            neg = next(q for q in range(u0, len(ids)) if tok.decode([ids[q]]).strip() == NEG[c]); J.append((c, k, role, t, neg, len(ids), u0))
out = {}
for nm, mid in (("base", "meta-llama/Llama-3.1-8B"), ("ins", "meta-llama/Llama-3.1-8B-Instruct")):
    M = LanguageModel(mid, dtype=torch.bfloat16); M.tokenizer.padding_side = "left"; R = []
    def _b(ch_):
        with M.trace(remote=True) as tr:
            H = list().save()
            for c, k, role, t, neg, n, u0 in ch_:
                with tr.invoke(t, add_special_tokens=False):
                    o = M.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o; H.append(o[0, -n:, :].float())
        return [h.float() for h in H]
    for a in range(0, len(J), 12):
        for r in range(4):
            try: R += _b(J[a:a + 12]); break
            except Exception as e: print("retry", repr(e)[-200:], flush=True)
        else: raise SystemExit("gave up")
    out[nm] = R; del M; print("read", nm, flush=True)
torch.save({"J": [x[:3] + x[4:] for x in J], "base": [h.half() for h in out["base"]], "ins": [h.half() for h in out["ins"]]}, "cant.pt")
f = safe_open("/home/three/interp/saes/crosscoder/model.safetensors", "pt"); q = 3872
We = f.get_slice("encoder.weight")[:, :, q].float(); be = float(f.get_slice("encoder.bias")[q]); dn = float(f.get_slice("decoder.weight")[:, q, :].norm(dim=-1).sum()); thr = f.get_tensor("threshold").item()
def A(b, i):
    pre = torch.relu(b @ We[0] + i @ We[1] + be); return pre * ((pre * dn) > thr)
S = {}
for k, x in enumerate(J):
    a = A(out["base"][k], out["ins"][k]); c, sk, role, t, neg, n, u0 = x; S.setdefault((c, role), []).append((float(a[neg]), float(a[u0:].max())))
print("#3872 mean at the negation token (firing of 6) | mean of the sentence's max")
for role in ("assistant", "user"):
    print(role)
    for c in CATS:
        v = S[(c, role)]; print(f"  {c:12s} {np.mean([x[0] for x in v]):.3f} ({sum(x[0] > 0 for x in v)}/6)  max {np.mean([x[1] for x in v]):.3f}", flush=True)
print("saved")
