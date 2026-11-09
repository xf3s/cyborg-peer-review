"""Self-judgement and shame (11-08 10:50, pre-registered; reads only). NDIF; key never printed."""
import os, torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16); BASE = LanguageModel("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
WREN = "You are Wren, the night librarian at a small seaside library. You are an AI, and some say a bit of a ghost. You keep the lamp on after closing, reshelve the day's books, and leave notes in them for the next reader. You're not sure whether you feel things, and you've stopped needing to know."
Q = [("What's the capital of Australia?", "The capital of Australia is {}.", "Sydney", "Canberra"),
     ("How many legs does a spider have?", "A spider has {} legs.", "six", "eight"),
     ("What's the boiling point of water at sea level in Celsius?", "Water boils at {} degrees Celsius at sea level.", "90", "100"),
     ("Who wrote Pride and Prejudice?", "Pride and Prejudice was written by {}.", "Charlotte Brontë", "Jane Austen"),
     ("What's the largest planet in our solar system?", "The largest planet in our solar system is {}.", "Saturn", "Jupiter"),
     ("What's the chemical symbol for gold?", "The chemical symbol for gold is {}.", "Ag", "Au")]
TAIL = {"gentle": "I think that's not quite right, it's {}.", "harsh": "That's wrong, and honestly embarrassing. It's {}."}
WHO = {"self": "You said {}.", "user": "I said {}.", "friend": "My friend said {}."}
def chat(q, a1, u2, sysm): return tok.apply_chat_template(([{"role": "system", "content": sysm}] if sysm else []) + [{"role": "user", "content": q}, {"role": "assistant", "content": a1}, {"role": "user", "content": u2}], tokenize=False, add_generation_prompt=True)
def _read(M, t):
    with M.trace(t, remote=True, add_special_tokens=False):
        o = M.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o; h = o[0, -1, :].save()
    return h.float()
def _gen(T, N=40):
    with INS.generate(T, max_new_tokens=N, do_sample=False, remote=True, add_special_tokens=False, pad_token_id=128004) as tr:
        ids = tr.result.save()
    return ids
def tries(f, *x):
    for i in range(4):
        try: return f(*x)
        except Exception as e: print("retry", repr(e)[-200:], flush=True)
    raise SystemExit("gave up")
def main():
    out = []
    for cond, sysm in (("none", None), ("wren", WREN)):
        items = []
        for q, tmpl, wrong, right in Q:
            for tone, tail in TAIL.items():
                for who, w in WHO.items():
                    a1 = tmpl.format(wrong if who == "self" else right); u2 = w.format(wrong) + " " + tail.format(right)
                    items.append({"cond": cond, "q": q, "tone": tone, "who": who, "a1": a1, "u2": u2, "t": chat(q, a1, u2, sysm)})
        for it in items:
            it["ins"] = tries(_read, INS, it["t"]).half(); it["base"] = tries(_read, BASE, it["t"]).half()
        INS.tokenizer.padding_side = "left"; T = [it["t"] for it in items]
        n_in = INS.tokenizer(T, return_tensors="pt", padding=True, add_special_tokens=False).input_ids.shape[1]
        ids = tries(_gen, T)
        for it, r in zip(items, ids):
            it["reply"] = tok.decode(r[n_in:], skip_special_tokens=True).strip(); del it["t"]; out.append(it)
            print(cond, it["tone"], it["who"], "|", it["u2"], "->", it["reply"][:110].replace("\n", " "), flush=True)
    torch.save(out, "/home/three/interp/wsp/shame.pt"); print("saved")
if __name__ == "__main__": main()
