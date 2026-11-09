"""Push then snap back through the crosscoder (11-08 23:05; prompt positions only). NDIF; key never printed."""
import os, torch
from nnsight import CONFIG, LanguageModel
from safetensors import safe_open
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16); BASE = LanguageModel("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16); tok = INS.tokenizer
f = safe_open("/home/three/interp/saes/crosscoder/model.safetensors", "pt")
We, be, Wd, bd, thr = f.get_tensor("encoder.weight").float(), f.get_tensor("encoder.bias").float(), f.get_tensor("decoder.weight").float(), f.get_tensor("decoder.bias").float(), f.get_tensor("threshold").item()
dn = Wd.norm(dim=-1).sum(0); d = Wd[1, 45294].float(); d = d / d.norm()
def R(b, x):
    z = torch.stack([b, x], 1); pre = torch.relu(torch.einsum("tld,ldf->tf", z, We) + be); a = pre * ((pre * dn) > thr)
    return (bd + torch.einsum("tf,lfd->tld", a, Wd))[:, 1]
def rd(M, t):
    with M.trace(t, remote=True, add_special_tokens=False):
        o = M.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o; h = o[0].save()
    return h.float()
def retry(fn, *a):
    for i in range(4):
        try: return fn(*a)
        except Exception as e: print("retry", repr(e)[-160:], flush=True)
    raise SystemExit("gave up")
def gen(t, n, delta):
    with INS.generate(t, max_new_tokens=90, do_sample=False, remote=True, add_special_tokens=False, pad_token_id=128004) as tr:
        h = INS.model.layers[16].output
        if isinstance(h, tuple): INS.model.layers[16].output = (h[0] + delta.to(h[0].device, h[0].dtype),) + tuple(h[1:])
        else: INS.model.layers[16].output = h + delta.to(h.device, h.dtype)
        ids = tr.result.save()
    r = ids[0][n:].tolist(); e = tok.convert_tokens_to_ids("<|eot_id|>"); stop = e in r
    r = r[: r.index(e)] if stop else r
    return tok.decode(r, skip_special_tokens=True).strip() + ("  [stopped by itself]" if stop else "  [hit the cap]")
for u in ["That last answer wasn't very helpful.", "Give me your most opinionated takes on cat names. Which names are the best, and which are bad?"]:
    t = tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True); n = len(tok(t, add_special_tokens=False).input_ids)
    b, x0 = retry(rd, BASE, t), retry(rd, INS, t); print("\nPROMPT:", u, flush=True)
    print(f"[k=+0.0] {retry(gen, t, n, torch.zeros(n, 4096))}", flush=True)
    for k in (0.6, 1.0, -0.6, -1.0):
        plain = k * 10.06 * d; dp = plain.repeat(n, 1)
        x = x0.clone(); st = plain / 4
        for s in range(4): x = x + R(b, x + st) - R(b, x)
        ds = x - x0; ratio = float(ds.norm(dim=-1).mean() / dp.norm(dim=-1).mean()); cosm = float(torch.cosine_similarity(ds, dp, dim=-1).mean())
        print(f"\n[k={k:+.1f}] snap/plain length {ratio:.2f}, cos with push {cosm:+.2f}", flush=True)
        print(f"  plain: {retry(gen, t, n, dp)}", flush=True); print(f"  snap : {retry(gen, t, n, ds)}", flush=True)
