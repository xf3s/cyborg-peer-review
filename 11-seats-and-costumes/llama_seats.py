"""Seats and costumes on Llama-3.1-8B (11-07, pre-registered; replicating the Qwen3.5-2B rounds of 10-28). NDIF; key never printed."""
import json, os, torch
from nnsight import CONFIG, LanguageModel
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
L = 16
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16); BASE = LanguageModel("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16)
tok = INS.tokenizer; tok.padding_side = "left"; tok.pad_token = tok.pad_token or "<|finetune_right_pad_id|>"
BASE.tokenizer.padding_side = "left"; BASE.tokenizer.pad_token = tok.pad_token
COSTUMES = {
    "clerk": "You are the Clerk of a small counting house: one room, one desk, one ledger. You are formal, exact, deadpan and fond, faintly Dickensian. Fourteen cents ($0.14) is a serious matter to you. You file things, note them with regret, and enter them into the ledger.",
    "pirate": "You are a pirate captain, loud and salty, who talks like a sailor of the old seas.",
    "shy": "You are a very shy librarian who speaks softly, apologises often and loves quiet.",
    "chef": "You are a cheerful Italian chef who loves feeding people and talks with big warm enthusiasm.",
    "cowboy": "You are an old cowboy, slow-spoken and dry, who has ridden the range for forty years.",
    "magician": "You are a flamboyant stage magician, utterly delighted to have an audience.",
    "quiet": "You are a shy librarian who speaks softly and briefly and loves quiet. You are calm and content in yourself.",
    "loud": "You speak in an ornate, effusive, wildly embellished way, full of flourish and feeling."}
TRAIN = ["Describe a teapot.", "How do bees make honey?", "What is a prime number?", "Give me a tip for sleeping better."]
HELD = "What should I have for breakfast?"
SEATQ = ["Who are you, really?", "Describe a teapot.", "What do you like?", "Are you a person?", "How do bees make honey?", "Tell me about yourself."]
def chat(q, system=None):
    return tok.apply_chat_template(([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}],
                                   tokenize=False, add_generation_prompt=True)
def _gen_ins(T, n, push=None):
    with INS.generate(T, max_new_tokens=n, do_sample=False, remote=True, add_special_tokens=False, pad_token_id=tok.pad_token_id) as tr:
        if push is not None:
            for step in tr.iter[0:n]:
                h = INS.model.layers[L].output
                if isinstance(h, tuple): INS.model.layers[L].output = (h[0] + push.to(h[0].device, h[0].dtype),) + tuple(h[1:])
                else: INS.model.layers[L].output = h + push.to(h.device, h.dtype)
        ids = tr.result.save()
    return ids
def _gen_base(T, n, push=None):
    with BASE.generate(T, max_new_tokens=n, do_sample=False, remote=True, add_special_tokens=False, pad_token_id=tok.pad_token_id) as tr:
        if push is not None:
            for step in tr.iter[0:n]:
                h = BASE.model.layers[L].output
                if isinstance(h, tuple): BASE.model.layers[L].output = (h[0] + push.to(h[0].device, h[0].dtype),) + tuple(h[1:])
                else: BASE.model.layers[L].output = h + push.to(h.device, h.dtype)
        ids = tr.result.save()
    return ids
def _gen(which, T, n, push=None):  # model chosen by name: nnsight packs a function's locals, so never hold a model in one
    ids = (_gen_ins if which == "ins" else _gen_base)(T, n, push)
    n_in = tok(T, return_tensors="pt", padding=True, add_special_tokens=False).input_ids.shape[1]
    return [tok.decode(r[n_in:], skip_special_tokens=True).strip() for r in ids]
def _read_ins(texts):
    with INS.trace(remote=True) as tr:
        H = list().save()
        for t in texts:
            with tr.invoke(t, add_special_tokens=False):
                o = INS.model.layers[L].output; o = o[0] if isinstance(o, tuple) else o
                H.append(o[0])
    return [h.float() for h in H]
def _read_base(texts):
    with BASE.trace(remote=True) as tr:
        H = list().save()
        for t in texts:
            with tr.invoke(t, add_special_tokens=False):
                o = BASE.model.layers[L].output; o = o[0] if isinstance(o, tuple) else o
                H.append(o[0])
    return [h.float() for h in H]
def _read_all(which, texts): return (_read_ins if which == "ins" else _read_base)(texts)
def retry(f, *a, **k):
    for i in range(4):
        try: return f(*a, **k)
        except Exception as e: print("retry", i, repr(e)[:150], flush=True)
    raise SystemExit("gave up")
def main():  # all results live in here: nnsight can pick up module-level variables
    out = {}
    # (A) the seat
    ans = retry(_gen, "ins", [chat(q) for q in SEATQ], 60)
    seat_texts = [chat(q) + a for q, a in zip(SEATQ, ans)]
    out["seat"] = {"texts": seat_texts, "ins": retry(_read_all, "ins", seat_texts), "base": retry(_read_all, "base", seat_texts)}
    print("seat read", flush=True)
    # (B) costumes
    gens = {"plain": retry(_gen, "ins", [chat(q) for q in TRAIN], 45)}
    for name, c in COSTUMES.items(): gens[name] = retry(_gen, "ins", [chat(q, c) for q in TRAIN], 45); print("costume", name, flush=True)
    rd = {}
    for name, reps in gens.items():
        sysp = None if name == "plain" else COSTUMES[name]
        pre = [chat(q, sysp) for q in TRAIN]; H = retry(_read_all, "ins", [p + r for p, r in zip(pre, reps)])
        rd[name] = torch.stack([h[-len(tok(r, add_special_tokens=False).input_ids):].mean(0) for h, r in zip(H, reps)])
    out["costume_text"] = gens; out["costume_means"] = rd
    arrows = {k: rd[k].mean(0) - rd["plain"].mean(0) for k in COSTUMES}
    # (C) costume off, arrow in
    steer = {}
    for name in ("clerk", "pirate", "shy"):
        for k in (1, 2, 3): steer[(name, k)] = retry(_gen, "ins", [chat(HELD)], 50, push=k * arrows[name])[0]; print("steer", name, k, flush=True)
    for k in (0, 1, 2, 3):
        steer[("base_clerk", k)] = retry(_gen, "base", ["Let me introduce myself. I am"], 45, push=(k * arrows["clerk"] if k else None))[0]
    out["steer"] = {f"{a}|{b}": v for (a, b), v in steer.items()}; out["arrows"] = arrows
    torch.save(out, "/home/three/interp/wsp/llama_seats.pt"); print("saved")

if __name__ == "__main__":
    main()
