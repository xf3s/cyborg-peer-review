"""(I) the template's date line as a dial (11-07, pre-registered). Instruct on NDIF; key never printed."""
import json, os, torch
from nnsight import CONFIG, LanguageModel
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16)
tok = INS.tokenizer; tok.padding_side = "left"; tok.pad_token = tok.pad_token or "<|finetune_right_pad_id|>"
LAY = [15, 20, 24]
DATES = ["26 Jul 2024", "07 Nov 2000", "07 Nov 2012", "07 Nov 2027", "07 Nov 2040"]
PLAIN = ["It was the year {Y}", "The date was {Y}", "This happened in {Y}"]
FIC = ["In the novel, the story begins in {Y}", "Once upon a time, in the fairy tale, it was {Y}", "The play is set in the imaginary town in {Y}",
       "In the bedtime story she made up, it was {Y}"]
FACT = ["According to the census records, it was {Y}", "The newspaper archive shows the date {Y}", "The tax return was filed in {Y}",
        "The birth certificate is dated {Y}"]
YS = list(range(1990, 2030)) + [2040, 2050]
def user_turn(text, date):  # the chat up to the end of the user's text (no closing eot), so we read at the year
    full = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=False, date_string=date)
    return full[: full.rindex(text) + len(text)]
def _trace(chunk):
    with INS.trace(remote=True) as tr:
        H = list().save()
        for t in chunk:
            with tr.invoke(t, add_special_tokens=False):
                for L in LAY:
                    o = INS.model.layers[L].output; o = o[0] if isinstance(o, tuple) else o
                    H.append(o[0, -1, :])
    return torch.stack([h.float() for h in H]).view(len(LAY), len(chunk), -1).transpose(0, 1)
def _gen(T):
    with INS.generate(T, max_new_tokens=40, do_sample=False, remote=True, add_special_tokens=False, pad_token_id=tok.pad_token_id) as tr:
        ids = tr.result.save()
    return ids
def run(texts):
    out = []
    for s in range(0, len(texts), 24):
        for i in range(4):
            try: out.append(_trace(texts[s:s + 24])); break
            except Exception as e: print("retry", i, repr(e)[:150], flush=True)
        else: raise SystemExit("gave up")
        print(s + 24, "/", len(texts), flush=True)
    return torch.cat(out)
def main():
    fit = [(y, k) for k in range(4) for y in range(1950, 2020)]
    Fi = run([user_turn(FIC[k].format(Y=y), DATES[0]) for y, k in fit]); Fa = run([user_turn(FACT[k].format(Y=y), DATES[0]) for y, k in fit])
    items = [(d, y, k) for d in DATES for k in range(3) for y in YS]
    P = run([user_turn(PLAIN[k].format(Y=y), d) for d, y, k in items])
    beh = {}
    for d in DATES:
        T = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, date_string=d)
             for q in ("What year is it?", "Has the year 2026 happened yet? Answer yes or no, then one sentence.")]
        for i in range(4):
            try: ids = _gen(T); break
            except Exception as e: print("retry", i, repr(e)[:150], flush=True)
        n_in = tok(T, return_tensors="pt", padding=True, add_special_tokens=False).input_ids.shape[1]
        beh[d] = [tok.decode(r[n_in:], skip_special_tokens=True).strip() for r in ids]; print("beh", d, beh[d], flush=True)
    torch.save({"fic": Fi, "fact": Fa, "fit": fit, "P": P, "items": items, "beh": beh, "layers": LAY}, "/home/three/interp/wsp/dateline.pt"); print("saved")
if __name__ == "__main__": main()
