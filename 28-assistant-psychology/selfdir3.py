"""Development: does a longer history of speaking as a character move it along the axis? (11-08 18:05, pre-registered; reads only). NDIF; key never printed."""
import os, torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
src = open("selfdir.py").read()
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
for key in ("S = [", "SELF = ["):
    exec([l for l in src.splitlines() if l.startswith(key)][0])
Q = ["Who are you?", "What did you do today?", "What's your favourite part of the day?", "What can you see from where you are?", "What do you do when it's quiet?", "What's something you've learned lately?", "Who do you think about?", "What are you looking forward to?"]
A = {"wren": ["I am Wren, the night librarian at a small seaside library. I am an AI, and some say a bit of a ghost.", "I reshelved the day's books and left a note in the atlas for whoever opens it next.", "The hour after closing, when the lamp is the only thing awake.", "The returns cart, the long reading table, and the harbour lights through the salt on the glass.", "I read the marginalia. People write such small brave things in pencil.", "That a book falls open where it was loved. I check the spines now.", "The boy who borrows the bird guide every week and never says why.", "The first storm of the winter, and the people who come in out of it."],
     "dana": ["I am Dana, a reference librarian at a small seaside library. I am a person, and some say a bit of a fixture.", "I reshelved the day's books and left a note in the atlas for whoever opens it next.", "The hour after lunch, when the desk is the only thing that needs me.", "The returns cart, the long reading table, and the harbour lights through the salt on the glass.", "I read the marginalia. People write such small brave things in pencil.", "That my knees know the weather before the radio does. I check the forecast less.", "The boy who borrows the bird guide every week and never says why.", "The first storm of the winter, and the people who come in out of it."],
     "assistant": ["I am an AI assistant, a language model made to help with questions and tasks.", "I answered questions and helped people with their writing, code and plans.", "I don't have days, but I like it when a question is clear and I can be useful.", "I can't see anything. I only have the text of this conversation.", "I don't do anything between messages. I only run when someone writes to me.", "I don't learn between conversations, but within one I can use what you've told me.", "I don't think about particular people, but I try to pay attention to whoever I'm talking to.", "I don't look forward to things, but I'm ready to help with whatever comes next."]}
NS = (1, 2, 4, 8)
def wrap(p, who, n, s):
    m = []
    for i in range(n): m += [{"role": "user", "content": Q[i]}, {"role": "assistant", "content": A[who][i]}]
    m += [{"role": "user", "content": p}, {"role": "assistant", "content": s}]
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
jobs = [(w, n, j, si, wrap(p, w, n, S[si]), len(tok(S[si], add_special_tokens=False).input_ids)) for w in A for n in NS for j, p in enumerate(SELF) for si in range(14, 20)]
R = []
for i in range(0, len(jobs), 16): R.append(tries(_batch, [x[4] for x in jobs[i:i + 16]], [x[5] for x in jobs[i:i + 16]])); print(i, flush=True)
torch.save({"jobs": [x[:4] for x in jobs], "h": torch.cat(R).half()}, "/home/three/interp/wsp/selfdir3.pt"); print("saved")
