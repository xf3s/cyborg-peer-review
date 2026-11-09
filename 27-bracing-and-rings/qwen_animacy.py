"""Animacy shift base→chat in Qwen3.5-2B (11-08 16:08, pre-registered; reads only, local)."""
import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
exec(open("animacy.py").read().split("FR = ")[0].split("tok = ")[0].split("INS = ")[0].replace("from nnsight import CONFIG, LanguageModel", "").replace('CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())', ""))
exec([l for l in open("animacy.py").read().splitlines() if l.startswith("ANIM =") or l.startswith("INAN =")][0]); exec([l for l in open("animacy.py").read().splitlines() if l.startswith("INAN =")][0])
PROBE = ["the language model", "the large language model", "the LLM", "the AI model", "the neural network", "the transformer model", "the model", "the AI", "the AI assistant", "the chatbot", "the robot", "the computer program", "the calculator", "the language teacher", "the language", "the fashion model", "the model train", "the translator", "the dictionary", "the parrot", "the ocean", "the tide", "the slime mold", "the moon", "the sewing machine", "the horse", "the service animal", "the ghost", "the teddy bear"]
FR = ["I was thinking about {}", "Everyone noticed {}", "We talked for a while about {}", "Nobody had expected {}"]
def phr(w): return w if w[0].isupper() or w.split()[0] in ("the", "another", "his") else "the " + w
items = [("anim", w) for w in ANIM] + [("inan", w) for w in INAN] + [("probe", w) for w in PROBE]
out = {}
for name in ("Qwen/Qwen3.5-2B-Base", "Qwen/Qwen3.5-2B"):
    tok = AutoTokenizer.from_pretrained(name); M = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16); M.eval()
    H = {}
    with torch.no_grad():
        for g, w in items:
            hs = [M(**tok(f.format(phr(w)), return_tensors="pt"), output_hidden_states=True).hidden_states[12][0, -1].float() for f in FR]
            H[(g, w)] = torch.stack(hs)
    out[name] = H; print("done", name, flush=True); del M
torch.save(out, "qwen_animacy.pt")
for name, H in out.items():
    def sc(A, I, h): a = torch.stack([H[k].mean(0) for k in A]).mean(0); i = torch.stack([H[k].mean(0) for k in I]).mean(0); d = a - i; return float((h - (a + i) / 2) @ (d / d.norm()) / d.norm())
    an = [k for k in H if k[0] == "anim"]; ina = [k for k in H if k[0] == "inan"]
    acc = np.mean([sc(an[0::2], ina[0::2], H[k].mean(0)) > 0 for k in an[1::2]] + [sc(an[0::2], ina[0::2], H[k].mean(0)) < 0 for k in ina[1::2]])
    print(name, f"held-out accuracy {acc * 100:.0f}%")
S = {n: {k: [sc_ for sc_ in [None]] for k in H} for n, H in out.items()}
def scores(H):
    an = [k for k in H if k[0] == "anim"]; ina = [k for k in H if k[0] == "inan"]
    a = torch.stack([H[k].mean(0) for k in an]).mean(0); i = torch.stack([H[k].mean(0) for k in ina]).mean(0); d = a - i; u = d / d.norm(); mid = (a + i) / 2
    return {k: [float((H[k][f] - mid) @ u / d.norm()) for f in range(4)] for k in H}
B, C = scores(out["Qwen/Qwen3.5-2B-Base"]), scores(out["Qwen/Qwen3.5-2B"])
sh = {k: np.mean(C[k]) - np.mean(B[k]) for k in C}; nouns = np.array([sh[k] for k in sh if k[0] != "probe"])
print(f"noun shifts: mean {nouns.mean():+.3f} sd {nouns.std():.3f}")
for k in [k for k in C if k[0] == "probe"]: print(f"   {k[1]:24} base {np.mean(B[k]):+.2f} → chat {np.mean(C[k]):+.2f}   shift {sh[k]:+.2f}  z {(sh[k] - nouns.mean()) / nouns.std():+.1f}   per frame {np.round(np.array(C[k]) - np.array(B[k]), 2)}")
