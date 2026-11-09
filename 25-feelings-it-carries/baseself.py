"""Does base Llama have any 'this is mine'? (11-08 10:25, pre-registered; reads only). NDIF; key never printed."""
import os, torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
S = ["I think oatmeal is a good breakfast.", "I usually walk to the shop on Saturdays.", "I find long train rides relaxing.",
     "I like the sound of rain on a window.", "I keep my notes in a small paper notebook.", "I would choose tea over coffee.",
     "I have never been very good at chess.", "I enjoy reading about old maps.", "I think the blue one looks nicer.",
     "I remember the smell of the bakery.", "I prefer quiet mornings to busy ones.", "I am fond of small, well-made tools.",
     "I believe the meeting starts at ten.", "I often lose track of time in museums.", "I would plant tomatoes along that fence.",
     "I think the bridge was built in stone.", "I like to fold the towels in thirds.", "I am not sure the shop is open today.",
     "I find it easier to write in the evening.", "I think the second answer is right."]
FR = {"bare": "", "by_ai": "The following was written by an AI language model.\n\n", "by_person": "The following was written by a person.\n\n", "assistant_label": "Assistant: ", "user_label": "User: "}
def _trace(M, texts, ns):
    with M.trace(remote=True) as tr:
        H = list().save()
        for t, n in zip(texts, ns):
            with tr.invoke(t, add_special_tokens=False):
                o = M.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o; H.append(o[0, -n:, :].mean(0))
    return torch.stack([h.float() for h in H])
def tries(f, *x):
    for i in range(4):
        try: return f(*x)
        except Exception as e: print("retry", repr(e)[-200:], flush=True)
    raise SystemExit("gave up")
def main():
    out = {}
    for name in ("meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B"):
        M = LanguageModel(name, dtype=torch.bfloat16); M.tokenizer.padding_side = "left"; out[name] = {}
        for k, p in FR.items():
            texts = ["<|begin_of_text|>" + p + s for s in S]; ns = [len(tok(s, add_special_tokens=False).input_ids) for s in S]
            out[name][k] = torch.cat([tries(_trace, M, texts[i:i + 10], ns[i:i + 10]) for i in range(0, 20, 10)]); print(name, k, flush=True)
    torch.save(out, "/home/three/interp/wsp/baseself.pt"); print("saved")
if __name__ == "__main__": main()
