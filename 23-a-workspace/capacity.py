"""Capacity of the about-to-answer state (11-08 00:20, pre-registered; reads only). NDIF; key never printed."""
import os, random, torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16); tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct"); INS.tokenizer.padding_side = "left"
A = ["cat", "dog", "owl", "horse", "sheep", "goat", "cow", "pig", "fox", "wolf", "bear", "deer", "frog", "duck", "goose", "hen", "crow", "hawk", "eagle", "mouse", "rat", "bat", "seal", "whale", "shark", "crab", "snail", "bee", "ant", "moth", "lion", "tiger"]
KS = (1, 2, 4, 8, 12, 16); LAY = (8, 12, 16, 20, 24)
def sent(items):
    s = ", ".join("a " + a for a in items[:-1]) + (" and " if len(items) > 1 else "") + ("an " if items[-1][0] in "aeiou" else "a ") + items[-1]
    s = s.replace("a owl", "an owl").replace("a eagle", "an eagle").replace("a ant", "an ant")
    return "I saw " + s + "."
def _read(T, offs):
    with INS.trace(T, remote=True, add_special_tokens=False):
        H = list().save(); G = list().save()
        for L in LAY:
            o = INS.model.layers[L].output; o = o[0] if isinstance(o, tuple) else o
            H.append(o[:, -1, :])
            for b, k in enumerate(offs): G.append(o[b, -(k + 1), :])
    return torch.stack([h.float() for h in H]).transpose(0, 1), torch.stack([gg.float() for gg in G]).view(len(LAY), len(T), -1).transpose(0, 1)
def main():
    rnd = random.Random(1108); items = []
    for k in KS:
        for _ in range(64): items.append((k, rnd.sample(A, k)))
    T, offs = [], []
    for k, its in items:
        s = sent(its); t = tok.apply_chat_template([{"role": "user", "content": s + " Which animals did I see?"}], tokenize=False, add_generation_prompt=True)
        tail = t[t.index(s) + len(s) - 1:]  # from the list's final '.' onward
        T.append(t); offs.append(len(tok(tail, add_special_tokens=False).input_ids) - 1)
    Ha, Hl = [], []
    for s in range(0, len(T), 16):
        for i in range(4):
            try: a, l = _read(T[s:s + 16], offs[s:s + 16]); break
            except Exception as e: print("retry", repr(e)[-200:], flush=True)
        Ha.append(a.half()); Hl.append(l.half()); print(s + 16, "/", len(T), flush=True)
    torch.save({"items": items, "animals": A, "layers": LAY, "answer": torch.cat(Ha), "listend": torch.cat(Hl)}, "/home/three/interp/wsp/capacity.pt"); print("saved")
if __name__ == "__main__": main()
