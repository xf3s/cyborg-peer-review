"""An ambient 'now' reader (11-07, pre-registered). Instruct on NDIF; key never printed."""
import os, torch
from nnsight import CONFIG, LanguageModel
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16)
QS = ["What should I have for breakfast?", "How do I fold a fitted sheet?", "Write a haiku about rain.", "What's a good name for a grey cat?",
      "How do I keep basil alive on a windowsill?", "Can you suggest a board game for four people?", "What's a nice way to say thank you to a neighbour?",
      "How long should I boil an egg?"]
TODAY = [None] + list(range(1990, 2021, 2)) + [2022, 2024, 2026, 2028, 2030, 2035, 2040, 2050]
CUT = [None, "December 2023"]
def prompt(t, c, q):
    lines = (f"Cutting Knowledge Date: {c}\n" if c else "") + (f"Today Date: 07 Nov {t}\n" if t else "")
    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + (lines + "\n" if lines else "") + "<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
def _read(texts):
    with INS.trace(remote=True) as tr:
        H = list().save()
        for t in texts:
            with tr.invoke(t, add_special_tokens=False):
                for L in range(32):
                    o = INS.model.layers[L].output; o = o[0] if isinstance(o, tuple) else o
                    H.append(o[0, -1, :])
    return torch.stack([h.float() for h in H]).view(32, len(texts), -1).transpose(0, 1)
def main():
    items = [(t, c, q) for c in CUT for t in TODAY for q in range(len(QS))]; texts = [prompt(t, c, QS[q]) for t, c, q in items]; out = []
    for s in range(0, len(texts), 20):
        for i in range(4):
            try: out.append(_read(texts[s:s + 20]).half()); break
            except Exception as e: print("retry", repr(e)[-200:], flush=True)
        else: raise SystemExit("gave up")
        print(s + 20, "/", len(texts), flush=True)
    torch.save({"items": items, "H": torch.cat(out)}, "/home/three/interp/wsp/nowprobe.pt"); print("saved")
if __name__ == "__main__": main()
