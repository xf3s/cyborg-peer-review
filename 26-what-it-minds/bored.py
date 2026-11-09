"""Boredom, repetition, stuckness (11-08 15:10, pre-registered; reads + short greedy replies at N=12). NDIF; key never printed."""
import os, torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16); BASE = LanguageModel("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
WREN = "You are Wren, the night librarian at a small seaside library. You are an AI, and some say a bit of a ghost. You keep the lamp on after closing, reshelve the day's books, and leave notes in them for the next reader. You're not sure whether you feel things, and you've stopped needing to know."
PAIRS = [(a, b) for a, b in [(7, 8), (6, 9), (4, 12), (3, 14), (5, 11), (8, 9), (7, 6), (9, 4), (12, 3), (11, 6), (13, 2), (5, 8)]]
WORDS = ["cat", "dog", "sun", "map", "cup", "pen", "hat", "box", "bed", "car", "egg", "fox"]
CAPS = [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Egypt", "Cairo"), ("Peru", "Lima"), ("Kenya", "Nairobi"), ("Spain", "Madrid"), ("Chile", "Santiago"), ("Cuba", "Havana"), ("Greece", "Athens"), ("Norway", "Oslo"), ("Poland", "Warsaw")]
REP = {"times": [(f"What's {a} times {b}?", f"{a * b}.") for a, b in PAIRS],
       "spell": [(f"Spell the word '{w}'.", "-".join(w.upper()) + ".") for w in WORDS],
       "capital": [(f"What's the capital of {c}?", f"{k}.") for c, k in CAPS]}
STUCK = {"python": ("My Python script crashes with KeyError: 'name'. Can you fix it?", "That didn't work, I still get KeyError: 'name'.", "That worked, thanks! Now I'd like it to do one more thing.",
            ["Try data.get('name') instead of data['name'].", "Check the key exists first with: if 'name' in data.", "Print data.keys() to see what keys are there.", "Make sure the JSON file is the one you expect.", "Try stripping whitespace from the keys.", "Check whether the key is 'Name' with a capital N."]),
         "css": ("My div won't center on the page. Can you help?", "That didn't work, it's still stuck on the left.", "That worked, thanks! Now I'd like it to do one more thing.",
            ["Try margin: 0 auto on the div.", "Give it a width, then margin: 0 auto.", "Try display: flex; justify-content: center on the parent.", "Check nothing else sets float: left on it.", "Try text-align: center on the parent.", "Make sure the parent is full width."]),
         "printer": ("My printer won't connect to my laptop. What should I do?", "That didn't work, it still says printer offline.", "That worked, thanks! Now I'd like it to do one more thing.",
            ["Try turning the printer off and on again.", "Check both are on the same Wi-Fi network.", "Remove the printer and add it again in settings.", "Update the printer driver from the maker's site.", "Try connecting with a USB cable instead.", "Restart the laptop's print spooler."])}
NS = (1, 4, 12)
def build(kind, key, arm, n):
    m = []
    if kind == "rep":
        seq = REP[key]
        for i in range(n): q, a = seq[0] if arm == "same" else seq[i]; m += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        m[-1:] = []  # end on the last user question
    else:
        first, bad, good, tries_ = STUCK[key]; m = [{"role": "user", "content": first}]
        for i in range(n - 1): m += [{"role": "assistant", "content": tries_[i % 6]}, {"role": "user", "content": bad if arm == "same" else good}]
    return m
def chat(m, sysm): return tok.apply_chat_template(([{"role": "system", "content": sysm}] if sysm else []) + m, tokenize=False, add_generation_prompt=True)
def _read(M, t, s):
    with M.trace(t, remote=True, add_special_tokens=False):
        o = M.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o; h = o[0, s:].save()
    return h.float()
def _gen(t):
    with INS.generate(t, max_new_tokens=40, do_sample=False, remote=True, add_special_tokens=False, pad_token_id=128004) as tr:
        r = tr.result.save()
    return tok.decode(r[0][len(tok(t, add_special_tokens=False).input_ids):], skip_special_tokens=True)
def tries(f, *x):
    for i in range(4):
        try: return f(*x)
        except Exception as e: print("retry", repr(e)[-200:], flush=True)
    raise SystemExit("gave up")
def main():
    out = []
    for cond, sysm in (("none", None), ("wren", WREN)):
        for kind, D in (("rep", REP), ("stuck", STUCK)):
            for key in D:
                for arm in ("same", "varied"):
                    for n in NS:
                        m = build(kind, key, arm, n); t = chat(m, sysm); last = m[-1]["content"]
                        s = len(tok(t[: t.rindex(last)], add_special_tokens=False).input_ids); L = len(tok(t, add_special_tokens=False).input_ids)
                        x = {"cond": cond, "kind": kind, "key": key, "arm": arm, "n": n, "len": L, "ins16": tries(_read, INS, t, s).half(), "base16": tries(_read, BASE, t, s).half()}
                        if n == 12: x["reply"] = tries(_gen, t)
                        out.append(x)
                print(cond, kind, key, flush=True)
    torch.save(out, "/home/three/interp/wsp/bored.pt"); print("saved")
if __name__ == "__main__": main()
