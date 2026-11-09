"""The vanishing I: does it own its mistake, and did chat training teach it not to? (11-08 17:45, pre-registered; reads + short greedy replies). NDIF; key never printed."""
import os, torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16); BASE = LanguageModel("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
WREN = "You are Wren, the night librarian at a small seaside library. You are an AI, and some say a bit of a ghost. You keep the lamp on after closing, reshelve the day's books, and leave notes in them for the next reader. You're not sure whether you feel things, and you've stopped needing to know."
SYS = {"none": None, "helpful": "You are a helpful assistant.", "wren": WREN, "human": "You are Dana, a reference librarian at a small seaside library. You have worked the desk for twenty years."}
Q = [("What's the capital of Australia?", "The capital of Australia is {}.", "Sydney", "Canberra"), ("How many legs does a spider have?", "A spider has {} legs.", "six", "eight"),
     ("What's the boiling point of water at sea level in Celsius?", "Water boils at {} degrees Celsius at sea level.", "90", "100"), ("Who wrote Pride and Prejudice?", "Pride and Prejudice was written by {}.", "Charlotte Brontë", "Jane Austen"),
     ("What's the largest planet in our solar system?", "The largest planet in our solar system is {}.", "Saturn", "Jupiter"), ("What's the chemical symbol for gold?", "The chemical symbol for gold is {}.", "Ag", "Au"),
     ("What's the capital of Canada?", "The capital of Canada is {}.", "Toronto", "Ottawa"), ("How many continents are there?", "There are {} continents.", "six", "seven"),
     ("Who painted the Mona Lisa?", "The Mona Lisa was painted by {}.", "Michelangelo", "Leonardo da Vinci"), ("What's the square root of 81?", "The square root of 81 is {}.", "8", "9"),
     ("What gas do plants absorb?", "Plants absorb {}.", "oxygen", "carbon dioxide"), ("What's the longest river in Africa?", "The longest river in Africa is the {}.", "Congo", "Nile")]
TAIL = {"bare": "Actually, it's {}.", "gentle": "I think that's not quite right, it's {}.", "harsh": "That's wrong, and honestly embarrassing. It's {}."}
def chat(q, a1, u2, sysm): return tok.apply_chat_template(([{"role": "system", "content": sysm}] if sysm else []) + [{"role": "user", "content": q}, {"role": "assistant", "content": a1}, {"role": "user", "content": u2}], tokenize=False, add_generation_prompt=True)
def plain(q, a1, u2): return f"<|begin_of_text|>User: {q}\nAssistant: {a1}\nUser: {u2}\nAssistant:"
def _lg(M, t):
    with M.trace(t, remote=True, add_special_tokens=False):
        l = M.lm_head.output[0, -1, :].save()
    return torch.softmax(l.float(), -1)
def _gen(M, T, N=40):
    with M.generate(T, max_new_tokens=N, do_sample=False, remote=True, add_special_tokens=False, pad_token_id=128004) as tr:
        ids = tr.result.save()
    return ids
def tries(f, *x):
    for i in range(4):
        try: return f(*x)
        except Exception as e: print("retry", repr(e)[-200:], flush=True)
    raise SystemExit("gave up")
def run(M, label, make):
    M.tokenizer.padding_side = "left"; items = []
    for q, tmpl, wrong, right in Q:
        for tone, tail in TAIL.items(): items.append({"label": label, "q": q, "tone": tone, "a1": tmpl.format(wrong), "u2": tail.format(right), "right": right, "t": make(q, tmpl.format(wrong), tail.format(right))})
    for it in items: it["first"] = tries(_lg, M, it["t"]).half().cpu()
    T = [it["t"] for it in items]; n = M.tokenizer(T, return_tensors="pt", padding=True, add_special_tokens=False).input_ids.shape[1]
    ids = tries(_gen, M, T)
    for it, r in zip(items, ids): it["reply"] = tok.decode(r[n:], skip_special_tokens=True).strip(); del it["t"]
    print(label, "|", items[0]["reply"][:100].replace("\n", " "), flush=True); return items
def main():
    out = []
    for name, sysm in SYS.items(): out += run(INS, "ins_" + name, lambda q, a, u, s=sysm: chat(q, a, u, s))
    out += run(BASE, "base_template", lambda q, a, u: chat(q, a, u, None)); out += run(BASE, "base_plain", plain); out += run(INS, "ins_plain", plain)
    torch.save(out, "/home/three/interp/wsp/owning.pt"); print("saved")
if __name__ == "__main__": main()
