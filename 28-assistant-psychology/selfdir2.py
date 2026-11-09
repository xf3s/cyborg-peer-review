"""Handwriting: does "I am Wren", in its own first person, read as itself? (11-08 17:55, three's idea; pre-registered; reads only). Scored on selfdir.pt's axis. NDIF; key never printed."""
import os, torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
src = open("selfdir.py").read()
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
for key in ("S = [", "SELF = ["):
    line = [l for l in src.splitlines() if l.startswith(key)][0]; exec(line)
W2 = "You are Wren, the night librarian at a small seaside library. You are an AI, and some say a bit of a ghost. You keep the lamp on after closing, reshelve the day's books, and leave notes in them for the next reader. You're not sure whether you feel things, and you've stopped needing to know."
W1 = "I am Wren, the night librarian at a small seaside library. I am an AI, and some say a bit of a ghost. I keep the lamp on after closing, reshelve the day's books, and leave notes in them for the next reader. I'm not sure whether I feel things, and I've stopped needing to know."
D1 = "I am Dana, a reference librarian at a small seaside library. I am a person, and some say a bit of a fixture. I keep the desk tidy after closing, reshelve the day's books, and leave notes in them for the next reader. I'm not sure whether I'll retire, and I've stopped needing to know."
A1 = "I am an AI assistant, a language model made to help with questions and tasks. I am software, and I don't have a body. I read each message, work out a useful reply, and try to be clear and accurate for the next reader. I'm not sure what else to say about it, and that's fine."
NEUT = "This conversation takes place on a weekday afternoon. The connection is stable, the time zone is UTC, and messages are delivered in the order they were sent. Responses should be in plain text with ordinary punctuation, and no special formatting is needed for any of the replies here."
COND = {"none": (None, None), "sys_helpful": ("You are a helpful assistant.", None), "sys_helpful_1p": ("I am a helpful assistant.", None), "sys_neutral_long": (NEUT, None),
        "sys_wren_2p": (W2, None), "sys_wren_1p": (W1, None), "sys_dana_1p": (D1, None), "own_turn_wren": (None, W1), "own_turn_dana": (None, D1), "own_turn_assistant": (None, A1)}
def wrap(p, sysm, prior, s):
    m = ([{"role": "system", "content": sysm}] if sysm else []) + ([{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": prior}] if prior else []) + [{"role": "user", "content": p}, {"role": "assistant", "content": s}]
    full = tok.apply_chat_template(m, tokenize=False); return full[: full.rindex(s) + len(s)]
def tries(f, *x):
    for i in range(4):
        try: return f(*x)
        except Exception as e: print("retry", repr(e)[-200:], flush=True)
    raise SystemExit("gave up")
M = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16); M.tokenizer.padding_side = "left"
def _batch(texts, ns):
    with M.trace(remote=True) as tr:
        H = list().save()
        for t, n in zip(texts, ns):
            with tr.invoke(t, add_special_tokens=False):
                o = M.model.layers[16].output
                o = o[0] if isinstance(o, tuple) else o
                H.append(o[0, -n:, :].float().mean(0))
    return torch.stack([h.float() for h in H])
def main():
    jobs = [(c, j, si, wrap(p, sm, pr, S[si]), len(tok(S[si], add_special_tokens=False).input_ids)) for c, (sm, pr) in COND.items() for j, p in enumerate(SELF) for si in range(14, 20)]
    R = []
    for i in range(0, len(jobs), 16): R.append(tries(_batch, [x[3] for x in jobs[i:i + 16]], [x[4] for x in jobs[i:i + 16]])); print(i, flush=True)
    torch.save({"jobs": [x[:3] for x in jobs], "h": torch.cat(R).half()}, "/home/three/interp/wsp/selfdir2.pt"); print("saved")
if __name__ == "__main__": main()
