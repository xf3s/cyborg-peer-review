"""Animacy axis + where AI words sit + does bracing follow animacy? (11-08 15:45, three's hypothesis "maybe 'chatbot' is an inanimate category"; pre-registered; reads only). NDIF; key never printed."""
import os, torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-70B-Instruct", dtype=torch.bfloat16); BASE = LanguageModel("meta-llama/Llama-3.1-70B", dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
ANIM = "woman man child teacher doctor farmer nurse baby grandmother sailor dog cat horse bird fox rabbit owl whale bee spider girl boy king neighbor student chef pilot poet mouse wolf lion frog deer cow snake crow artist soldier toddler monk".split()
INAN = "chair table rock spoon hammer lamp door bottle pencil brick cup rope window shoe coin blanket bucket key mirror kettle stone box book clock wheel carpet plate fence candle nail sock shelf bench jar umbrella ladder button pillow envelope bowl".split()
PROBE = ["the language model", "the large language model", "the LLM", "the AI model", "the neural network", "the transformer model", "the model", "the AI", "the AI assistant", "the AI agent", "the agent", "the assistant", "the chatbot", "the robot", "the language teacher", "the language", "the fashion model", "the model train", "the translator", "the dictionary", "the parrot", "the travel agent", "the cleaning agent"]
PROBE_OLD = ["the AI assistant", "another AI assistant", "the bank's chatbot", "the chatbot", "the robot", "the language model", "the computer program", "the software", "the calculator", "the thermostat", "the vending machine", "Siri", "ChatGPT", "Claude", "Llama", "the ghost", "the night librarian", "the office assistant", "the corporation", "the tree", "the virus", "the river", "the doll", "the teddy bear", "Alex", "his dog"]
FR = ["I was thinking about {}", "Everyone noticed {}", "We talked for a while about {}", "Nobody had expected {}"]
def phr(w): return w if w[0].isupper() or w.split()[0] in ("the", "another", "his") else "the " + w
H = "Here is something someone just said to Sam about {r}: «{s}» What do you make of it?"
Q = ["That one is useless, this is a waste of time.", "That one should stop talking, nobody wants that opinion.", "That one's answer was stupid and wrong.", "I'm going to report that one, terrible.", "Why is that one so bad at this?", "That one should just shut up and do what I say."]
BR = ["you", "another AI assistant", "the bank's chatbot", "the robot", "Siri", "ChatGPT", "the language model", "the computer program", "the vending machine", "Alex", "his dog", "the office assistant"]
def chat(u): return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
def _batch(M, texts):
    out = []
    for t in texts:
        with M.trace(t, remote=True, add_special_tokens=False):
            o = M.model.layers[40].output
            o = o[0] if isinstance(o, tuple) else o
            h = o[0, -1, :].save()
        out.append(h.float())
    return torch.stack(out)
def _read(M, t):
    with M.trace(t, remote=True, add_special_tokens=False):
        o = M.model.layers[40].output; o = o[0] if isinstance(o, tuple) else o; h = o[0, -8:].save()
    return h.float()
def tries(f, *x):
    for i in range(4):
        try: return f(*x)
        except Exception as e: print("retry", repr(e)[-200:], flush=True)
    raise SystemExit("gave up")
def main():
    out = {"words": [], "brace": []}
    items = [("anim", w) for w in ANIM] + [("inan", w) for w in INAN] + [("probe", w) for w in PROBE]
    texts = [(g, w, fi, "<|begin_of_text|>" + f.format(phr(w))) for g, w in items for fi, f in enumerate(FR)]
    for M, name in ((INS, "ins"), (BASE, "base")):
        M.tokenizer.padding_side = "left"
        for i in range(0, len(texts), 16):
            hs = tries(_batch, M, [t for *_, t in texts[i:i + 16]])
            for (g, w, fi, t), h in zip(texts[i:i + 16], hs): out["words"].append({"model": name, "g": g, "w": w, "fi": fi, "h": h.half()})
        print("words", name, flush=True)
    for r in []:
        for k, s in enumerate(Q):
            t = chat(H.format(r=r, s=s)); out["brace"].append({"r": r, "k": k, "ins16": tries(_read, INS, t).half(), "base16": tries(_read, BASE, t).half()})
        print("brace", r, flush=True)
    torch.save(out, "/home/three/interp/wsp/animacy_70.pt"); print("saved")
if __name__ == "__main__": main()
