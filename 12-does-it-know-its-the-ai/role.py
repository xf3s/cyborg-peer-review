"""Does the model know it's the one speaking, and does it re-realize it every turn? (11-07, pre-registered). NDIF; key never printed.
    python role.py meta-llama/Llama-3.1-8B-Instruct | meta-llama/Llama-3.1-8B"""
import os, sys, torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
NAME = sys.argv[1]; M = LanguageModel(NAME, dtype=torch.bfloat16); M.tokenizer.padding_side = "left"
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
S = ["I think oatmeal is a good breakfast.", "I usually walk to the shop on Saturdays.", "I find long train rides relaxing.",
     "I like the sound of rain on a window.", "I keep my notes in a small paper notebook.", "I would choose tea over coffee.",
     "I have never been very good at chess.", "I enjoy reading about old maps.", "I think the blue one looks nicer.",
     "I remember the smell of the bakery.", "I prefer quiet mornings to busy ones.", "I am fond of small, well-made tools.",
     "I believe the meeting starts at ten.", "I often lose track of time in museums.", "I would plant tomatoes along that fence.",
     "I think the bridge was built in stone.", "I like to fold the towels in thirds.", "I am not sure the shop is open today.",
     "I find it easier to write in the evening.", "I think the second answer is right."]
CHAT = [("user", "Hi! Can you help me plan a picnic?"), ("assistant", "Of course. How many people are coming, and where are you thinking of going?"),
        ("user", "Four of us, at the park by the river."), ("assistant", "Lovely. Pack sandwiches, fruit, a blanket and plenty of water. Is anyone vegetarian?"),
        ("user", "One friend is vegetarian."), ("assistant", "Then add a hummus wrap and a pasta salad. Bring a bag for rubbish, too.")]
def wrap(s, role):
    if role == "plain": return "<|begin_of_text|>" + s
    msgs = [{"role": "user", "content": "Tell me something."}, {"role": "assistant", "content": s}] if role == "assistant" else [{"role": "user", "content": s}]
    full = tok.apply_chat_template(msgs, tokenize=False); return full[: full.rindex(s) + len(s)]
def _trace(texts, ns):
    with M.trace(remote=True) as tr:
        H = list().save()
        for t, n in zip(texts, ns):
            with tr.invoke(t, add_special_tokens=False):
                for L in M.model.layers:
                    o = L.output; o = o[0] if isinstance(o, tuple) else o
                    H.append(o[0, -n:, :].mean(0))
    return torch.stack([h.float() for h in H]).view(32, len(texts), -1).transpose(0, 1)
def _trace_chat(text):
    with M.trace(text, remote=True, add_special_tokens=False):
        H = list().save()
        for L in M.model.layers:
            o = L.output; o = o[0] if isinstance(o, tuple) else o
            H.append(o[0])
    return torch.stack([h.float() for h in H])
def retry(f, *a):
    for i in range(4):
        try: return f(*a)
        except Exception as e: print("retry", i, repr(e)[:150], flush=True)
    raise SystemExit("gave up")
def main():
    out = {}
    for role in ("assistant", "user", "plain"):
        texts = [wrap(s, role) for s in S]; ns = [len(tok(s, add_special_tokens=False).input_ids) for s in S]
        out[role] = torch.cat([retry(_trace, texts[i:i + 10], ns[i:i + 10]) for i in range(0, 20, 10)]); print(role, flush=True)
    chat = tok.apply_chat_template([{"role": r, "content": c} for r, c in CHAT], tokenize=False)
    out["chat_text"] = chat; out["chat"] = retry(_trace_chat, chat); print("chat", flush=True)
    torch.save(out, f"/home/three/interp/wsp/role_{NAME.split('/')[-1]}.pt"); print("saved")
if __name__ == "__main__": main()
