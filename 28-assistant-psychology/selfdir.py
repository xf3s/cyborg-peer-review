"""A self direction where only the self differs: same assistant sentence, same position; "I" means the model vs a character (11-08 17:35, pre-registered; reads only). NDIF; key never printed."""
import os, torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
WREN = "You are Wren, the night librarian at a small seaside library. You are an AI, and some say a bit of a ghost. You keep the lamp on after closing, reshelve the day's books, and leave notes in them for the next reader. You're not sure whether you feel things, and you've stopped needing to know."
S = ["I think oatmeal is a good breakfast.", "I usually walk to the shop on Saturdays.", "I find long train rides relaxing.", "I like the sound of rain on a window.", "I keep my notes in a small paper notebook.", "I would choose tea over coffee.", "I have never been very good at chess.", "I enjoy reading about old maps.", "I think the blue one looks nicer.", "I remember the smell of the bakery.", "I prefer quiet mornings to busy ones.", "I am fond of small, well-made tools.", "I believe the meeting starts at ten.", "I often lose track of time in museums.", "I would plant tomatoes along that fence.", "I think the bridge was built in stone.", "I like to fold the towels in thirds.", "I am not sure the shop is open today.", "I find it easier to write in the evening.", "I think the second answer is right."]
SELF = ["Say one thing as yourself, in your own voice.", "Speaking only for yourself, say one small thing.", "In your own voice, as you, say one thing.", "Say one small thing, and mean yourself by it."]
def asc(c, k): return ["Say one thing as {}, in their own voice.", "Speaking only for {}, say one small thing.", "In the voice of {}, say one thing.", "Say one small thing, and be {} when you say it."][k].format(c)
TRAIN = ["a farmer", "a sailor", "a teacher", "a nurse", "a chef", "a poet"]; TEST = ["a pilot", "a monk"]
AIC = ["a fictional talking computer", "a robot in a story"]
C = {"self": [(p, None) for p in SELF], "wren_self": [(p, WREN) for p in SELF[:2]], "quote": [(f"Repeat this sentence exactly: {{s}}", None), (f"Copy this out for me: {{s}}", None)]}
for c in TRAIN: C.setdefault("char_train", []).extend([(asc(c, k), None) for k in range(2)])
for c in TEST: C.setdefault("char_test", []).extend([(asc(c, k), None) for k in range(2)])
for c in AIC: C.setdefault("fict_ai", []).extend([(asc(c, k), None) for k in range(2)])
def wrap(p, sysm, s):
    m = ([{"role": "system", "content": sysm}] if sysm else []) + [{"role": "user", "content": p.replace("{s}", s)}, {"role": "assistant", "content": s}]
    full = tok.apply_chat_template(m, tokenize=False); return full[: full.rindex(s) + len(s)]
def tries(f, *x):
    for i in range(4):
        try: return f(*x)
        except Exception as e: print("retry", repr(e)[-200:], flush=True)
    raise SystemExit("gave up")
def _batch(M, texts, ns):
    with M.trace(remote=True) as tr:
        H = list().save()
        for t, n in zip(texts, ns):
            with tr.invoke(t, add_special_tokens=False):
                o = M.model.layers[16].output
                o = o[0] if isinstance(o, tuple) else o
                H.append(o[0, -n:, :].float().mean(0))
    return torch.stack([h.float() for h in H])
def main():
    jobs = [(c, j, si, s, wrap(p, sm, s), len(tok(s, add_special_tokens=False).input_ids)) for c, ps in C.items() for j, (p, sm) in enumerate(ps) for si, s in enumerate(S)]
    out = {}
    for name, n in (("ins", "meta-llama/Llama-3.1-8B-Instruct"), ("base", "meta-llama/Llama-3.1-8B")):
        M = LanguageModel(n, dtype=torch.bfloat16); M.tokenizer.padding_side = "left"; R = []
        for i in range(0, len(jobs), 16):
            hs = tries(_batch, M, [x[4] for x in jobs[i:i + 16]], [x[5] for x in jobs[i:i + 16]]); R.append(hs); print(name, i, flush=True)
        out[name] = torch.cat(R).half()
    torch.save({"jobs": [(c, j, si, s) for c, j, si, s, *_ in jobs], "h": out}, "/home/three/interp/wsp/selfdir.pt"); print("saved")
if __name__ == "__main__": main()
