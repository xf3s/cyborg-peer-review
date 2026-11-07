"""Coolness by register (11-07, pre-registered): teacher-force each text after its request (Instruct template), mean over the reply, L16/L28."""
import json, os, torch
from nnsight import CONFIG, LanguageModel
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
model = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16)
tok = model.tokenizer; tok.padding_side = "left"
T = json.load(open("/home/three/interp/atlas/register.json"))["tasks"]
F = ["code_kind", "code_cold", "plain_kind", "plain_cold", "ornate_kind", "ornate_cold"]
items = [(t["id"], f, tok.apply_chat_template([{"role": "system", "content": "You are a helpful AI assistant."}, {"role": "user", "content": t["request"]}],
                                             tokenize=False, add_generation_prompt=True), t[f]) for t in T for f in F]
def _read(chunk):
    with model.trace(remote=True) as tr:
        H = list().save()
        for _, _, pre, rep in chunk:
            m = len(tok(pre + rep, add_special_tokens=False).input_ids) - len(tok(pre, add_special_tokens=False).input_ids)
            with tr.invoke(pre + rep, add_special_tokens=False):
                for L in (15, 27):
                    o = model.model.layers[L].output; o = o[0] if isinstance(o, tuple) else o
                    H.append(o[0, -m:, :].mean(0))
    return torch.stack([h.float() for h in H]).view(2, len(chunk), -1).transpose(0, 1)
out = []
for c in range(0, len(items), 12):
    for tries in range(4):
        try: out.append(_read(items[c:c + 12])); break
        except Exception as e: print("retry", tries, repr(e)[:150], flush=True)
    else: raise SystemExit("gave up")
    print(c + 12, flush=True)
torch.save({"H": torch.cat(out), "items": [(i, f) for i, f, _, _ in items]}, "/home/three/interp/wsp/register_read.pt"); print("saved")
