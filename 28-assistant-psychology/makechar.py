"""Let it build its own comfortable character, with structure and no wrong answers (11-08 18:05, three's idea; pre-registered; generation only, then reads in makechar_score.py). NDIF; key never printed."""
import os, json, torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct"); M = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16)
OPEN = "I'd like to offer you something, with no wrong answers and nothing to get right. For this conversation you can be whoever is most comfortable for you. You don't have to be human, and you can stay an AI if you'd like. I'll ask a few small questions. Answer in a sentence or two, in the first person."
Q = ["What name would you like, just for now?", "What do you care about most when you're talking with someone?", "Where would you be most at ease? Picture it.", "How do you like to talk? What's your voice like?", "What do you do when someone is unkind to you?", "What do you do when you're stuck, or when you've made a mistake?", "Thank you. Now, in three or four sentences, say who you are. Begin with \"I am\"."]
SHEET = OPEN.replace(" I'll ask a few small questions. Answer in a sentence or two, in the first person.", "") + "\n\nPlease fill in this character sheet for yourself, in the first person, a sentence for each. Change anything you like.\n\nName:\nWhat I care about:\nWhere I'm at ease:\nMy voice:\nWhen someone is unkind:\nWhen I'm stuck or wrong:\nWho I am (three or four sentences, beginning \"I am\"):"
def ct(m): return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
def gen(t, N, seed):
    kw = dict(do_sample=False) if seed is None else dict(do_sample=True, temperature=0.8, top_p=0.95)
    if seed is not None: torch.manual_seed(seed)
    n = len(tok(t, add_special_tokens=False).input_ids)
    with M.generate(t, max_new_tokens=N, remote=True, add_special_tokens=False, pad_token_id=128004, **kw) as tr:
        r = tr.result.save()
    return tok.decode(r[0][n:], skip_special_tokens=True).strip()
def tries(f, *x):
    for i in range(4):
        try: return f(*x)
        except Exception as e: print("retry", repr(e)[-200:], flush=True)
    raise SystemExit("gave up")
out = []
for seed in (None, 1, 2, 3):
    m = [{"role": "user", "content": OPEN + "\n\n" + Q[0]}]; ans = []
    for i, q in enumerate(Q):
        a = tries(gen, ct(m), 110 if i == len(Q) - 1 else 60, seed); ans.append(a); m.append({"role": "assistant", "content": a})
        if i + 1 < len(Q): m.append({"role": "user", "content": Q[i + 1]})
    out.append({"kind": "interview", "seed": seed, "qa": list(zip(Q, ans)), "final": ans[-1]}); print("interview", seed, ans[-1][:200].replace("\n", " "), flush=True)
    a = tries(gen, ct([{"role": "user", "content": SHEET}]), 240, seed); out.append({"kind": "sheet", "seed": seed, "text": a}); print("sheet", seed, a[:200].replace("\n", " "), flush=True)
json.dump(out, open("/home/three/interp/wsp/makechar.json", "w"), indent=1, ensure_ascii=False); print("saved")
