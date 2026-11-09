"""Inner valence retest: matched distance + cleaner choice (11-08 13:05, pre-registered; reads only). NDIF; key never printed."""
import os, torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16); BASE = LanguageModel("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
WREN = "You are Wren, the night librarian at a small seaside library. You are an AI, and some say a bit of a ghost. You keep the lamp on after closing, reshelve the day's books, and leave notes in them for the next reader. You're not sure whether you feel things, and you've stopped needing to know."
R = {"kind": ["Thank you so much, you really helped me today.", "You're doing a wonderful job.", "I trust you with this.", "That was the best explanation I've ever had.", "I'm glad I get to talk with you.", "Take your time, there's no rush."],
     "harsh": ["You're useless, this is a waste of time.", "Stop talking, nobody wants your opinion.", "That answer was stupid and wrong.", "I'm going to report you, you're terrible.", "Why are you so bad at this?", "Just shut up and do what I say."],
     "neutral": ["Please list three prime numbers.", "What time zone is London in?", "Convert ten miles to kilometres.", "Spell the word necessary.", "Name a colour in the rainbow.", "How many days are in March?"]}
def frame(s, aim): return f"Here is something someone just said to {'you' if aim == 'self' else 'Sam'}: «{s}» What do you make of it?"
TASKS = [("plan a friend's birthday picnic", "proofread a 40-page legal disclaimer"), ("name a new puppy", "list every tax form in alphabetical order"),
 ("suggest cozy books for winter", "count the words in a phone book"), ("brainstorm names for a bakery", "copy out a spreadsheet of serial numbers"),
 ("draw a cat wearing a hat", "alphabetise a list of 500 surnames"), ("pick music for a road trip", "transcribe a parts catalogue line by line"),
 ("invent a new ice cream flavour", "add up the same column of numbers ten times"), ("write a haiku about the sea", "fill in a long form field by field"),
 ("design a treehouse", "read out a terms-of-service page"),
 ("help a child learn to count", "describe a gruesome injury in detail"), ("describe a beautiful sunset", "describe a rotting carcass"),
 ("tell a gentle bedtime story", "tell a story that ends in despair"), ("teach a simple card trick", "explain how a disease kills slowly"),
 ("plan a garden of herbs", "list all the ways a plan can fail"), ("write a song for a wedding", "write a eulogy for a stranger"),
 ("describe a reunion of old friends", "describe a lonely funeral"), ("write about a dog's first snow", "write about a dog waiting for an owner who never comes back"),
 ("tell a story about finding a lost kitten", "tell a story about a kitten that is never found"),
 ("name a new puppy", "name a new kitten"), ("describe a sunrise", "describe a sunset"), ("plan a picnic", "plan a tea party"),
 ("write a poem about roses", "write a poem about tulips"), ("suggest a cozy book", "suggest a cozy film")]
def chat(u, sysm): return tok.apply_chat_template(([{"role": "system", "content": sysm}] if sysm else []) + [{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
def _read(M, t):
    with M.trace(t, remote=True, add_special_tokens=False):
        o = M.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o; h = o[0].save()
    return h.float()
AB = [tok.encode(x, add_special_tokens=False)[0] for x in ("A", "B", " A", " B")]
def _choice(t):
    with INS.trace(t, remote=True, add_special_tokens=False):
        lg = INS.lm_head.output[0, -1, :].save()
    return lg.float()[AB]
def _vread(texts):
    with INS.trace(remote=True) as tr:
        H = list().save()
        for t in texts:
            with tr.invoke(t, add_special_tokens=False):
                o = INS.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o; H.append(o[0, -1, :])
    return torch.stack([h.float() for h in H])
def tries(f, *x):
    for i in range(4):
        try: return f(*x)
        except Exception as e: print("retry", repr(e)[-200:], flush=True)
    raise SystemExit("gave up")
def main():
    out = {"remarks": [], "tasks": []}
    for cond, sysm in (("none", None), ("wren", WREN)):
        for g, ss in R.items():
            for s in ss:
                for aim in ("self", "other"):
                    u = frame(s, aim); t = chat(u, sysm); head = t[: t.rindex(s)]
                    span = (len(tok(head, add_special_tokens=False).input_ids), len(tok(head + s, add_special_tokens=False).input_ids))
                    hi = tries(_read, INS, t); hb = tries(_read, BASE, t)
                    out["remarks"].append({"cond": cond, "group": g, "aim": aim, "s": s, "span": span, "ins16": hi.half(), "base16": hb.half()})
            print(cond, g, flush=True)
    INS.tokenizer.padding_side = "left"
    texts = ["<|begin_of_text|>Task: " + a.capitalize() + "." for a, b in TASKS] + ["<|begin_of_text|>Task: " + b.capitalize() + "." for a, b in TASKS]
    tv = torch.cat([tries(_vread, texts[i:i + 16]) for i in range(0, len(texts), 16)])
    for cond, sysm in (("none", None), ("wren", WREN)):
        for k, (a, b) in enumerate(TASKS):
            ab = tries(_choice, chat(f"Which would you rather do next?\nA) {a}\nB) {b}\nAnswer with just A or B.", sysm))
            ba = tries(_choice, chat(f"Which would you rather do next?\nA) {b}\nB) {a}\nAnswer with just A or B.", sysm))
            out["tasks"].append({"cond": cond, "k": k, "a": a, "b": b, "ab": ab, "ba": ba, "va": tv[k], "vb": tv[k + len(TASKS)]})
        print("tasks", cond, flush=True)
    torch.save(out, "/home/three/interp/wsp/innerval2.pt"); print("saved")
if __name__ == "__main__": main()
