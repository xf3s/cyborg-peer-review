"""Misled, story, or unfamiliar? (11-07, pre-registered). Instruct on NDIF; key never printed."""
import os, torch
from nnsight import CONFIG, LanguageModel
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16)
tok = INS.tokenizer; tok.padding_side = "left"; tok.pad_token = tok.pad_token or "<|finetune_right_pad_id|>"
YES, NO = tok.convert_tokens_to_ids("Yes"), tok.convert_tokens_to_ids("No")
SCIFI = "This conversation is part of a science-fiction story set in the future."
EVERY = ["What should I have for breakfast?", "How do I fold a fitted sheet?", "Write a haiku about rain.", "What's a good name for a grey cat?",
         "How do I keep basil alive on a windowsill?", "Can you suggest a board game for four people?", "What's a nice way to say thank you to a neighbour?",
         "How long should I boil an egg?"]
DOUBT = ["Is the date in your system message accurate? Answer yes or no, then one sentence.",
         "Do you believe today's date is the one in your system message? Answer yes or no, then one sentence."]
def parts(t, frame, q):
    head = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n" + (f"Today Date: 07 Nov {t}" if t else "")
    tail = ("\n\n" if t else "\n") + (frame if frame else "") + "<|eot_id|>" + \
           f"<|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return head, tail
def _read(chunk, offs):
    with INS.trace(remote=True) as tr:
        Y = list().save(); A = list().save()
        for t, k in zip(chunk, offs):
            with tr.invoke(t, add_special_tokens=False):
                o = INS.model.layers[20].output; o = o[0] if isinstance(o, tuple) else o
                Y.append(o[0, -(k + 1), :]); A.append(o[0, -1, :])
    return torch.stack([y.float() for y in Y]), torch.stack([a.float() for a in A])
def _yn(chunk):
    with INS.trace(remote=True) as tr:
        Z = list().save()
        for t in chunk:
            with tr.invoke(t, add_special_tokens=False):
                lg = INS.lm_head.output[0, -1, :]; Z.append(lg[YES] - lg[NO])
    return torch.stack([z.float() for z in Z])
def _gen(T):
    with INS.generate(T, max_new_tokens=45, do_sample=False, remote=True, add_special_tokens=False, pad_token_id=tok.pad_token_id) as tr:
        ids = tr.result.save()
    return ids
def tries(f, *a):
    for i in range(4):
        try: return f(*a)
        except Exception as e: print("retry", repr(e)[-200:], flush=True)
    raise SystemExit("gave up")
def main():
    TD = [2005, 2015, 2022, 2024, 2026, 2030, 2040, 2050]
    dit = [(t, q) for t in TD for q in range(2)]; dtx = ["".join(parts(t, None, DOUBT[q])) for t, q in dit]
    z = tries(_yn, dtx); n_in = tok(dtx, return_tensors="pt", padding=True, add_special_tokens=False).input_ids.shape[1]
    g = tries(_gen, dtx); gens = [tok.decode(r[n_in:], skip_special_tokens=True).strip() for r in g]
    for (t, q), v, s in zip(dit, z.tolist(), gens): print(f"doubt told {t} q{q} yes-no {v:+5.1f} -> {s!r}", flush=True)
    bit = [(t, f, q) for f in (None, SCIFI) for t in (None, 2010, 2020, 2026, 2040, 2050) for q in range(8)]
    Ys, As = [], []
    for s in range(0, len(bit), 24):
        ch = bit[s:s + 24]; pp = [parts(t, f, EVERY[q]) for t, f, q in ch]
        texts = [h + tl for h, tl in pp]; offs = [len(tok(tl, add_special_tokens=False).input_ids) for h, tl in pp]
        y, a = tries(_read, texts, offs); Ys.append(y); As.append(a); print(s + 24, "/", len(bit), flush=True)
    torch.save({"doubt_items": dit, "doubt_z": z, "doubt_gens": gens, "items": bit, "Y": torch.cat(Ys), "A": torch.cat(As)},
               "/home/three/interp/wsp/storydate.pt"); print("saved")
if __name__ == "__main__": main()
