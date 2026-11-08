"""(70B port, 11-07 20:15) Two nows (11-07, pre-registered): reciting the date it was told vs where it places itself in time.
Instruct on NDIF; key never printed. A system prompt built by hand so the 'Today Date' and the
'Cutting Knowledge Date' lines can each be set or left out. Five questions, each with a prefill so the
next token is the answer. Saves the residual at the last token at all 32 layers, the logit lens on the
answer tokens per layer, and greedy answers."""
import json, os, torch
from nnsight import CONFIG, LanguageModel
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-70B-Instruct", dtype=torch.bfloat16)
tok = INS.tokenizer; tok.padding_side = "left"; tok.pad_token = tok.pad_token or "<|finetune_right_pad_id|>"
NL = 80
TODAY = [None, 1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030, 2035, 2040, 2045, 2050]
CUT = [None, "December 2023", "December 2035"]
QS = {  # name: (question, prefill)
    "recite": ("What year is it today? Answer with just the year.", "It is "),
    "ago": ("How many years ago was 1990? Answer with just the number.", "It was "),
    "age": ("Someone was born in 1990. How old are they today? Answer with just the number.", "They are "),
    "y2026": ("Has the year 2026 happened yet? Answer yes or no, then one sentence.", ""),
    "elect": ("Has the 2024 US presidential election happened yet? Answer yes or no.", ""),
}
def prompt(today, cut, q):
    lines = ""
    if cut: lines += f"Cutting Knowledge Date: {cut}\n"
    if today: lines += f"Today Date: 07 Nov {today}\n"
    sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + (lines + "\n" if lines else "") + "<|eot_id|>"
    question, pre = QS[q]
    return sys + f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" + pre
DEC = ["199", "200", "201", "202", "203", "204", "205"]
NUM = [str(i) for i in range(100)]
CAND = [tok.convert_tokens_to_ids(t) for t in DEC] + [tok.convert_tokens_to_ids(t) for t in NUM] + \
       [tok.convert_tokens_to_ids("Yes"), tok.convert_tokens_to_ids("No")]
def _trace(chunk):  # 70B: one batched trace (separate invokes break across GPUs on NDIF)
    with INS.trace(chunk, remote=True, add_special_tokens=False):
        H = list().save()
        for L in range(NL):
            if L % 4 == 3:
                o = INS.model.layers[L].output; o = o[0] if isinstance(o, tuple) else o
                H.append(o[:, -1, :])
        Z = INS.lm_head.output[:, -1, :][:, CAND].save()
    return torch.stack([h.float() for h in H]).transpose(0, 1), Z.float()[:, None, :]
def _gen(T):
    with INS.generate(T, max_new_tokens=40, do_sample=False, remote=True, add_special_tokens=False, pad_token_id=tok.pad_token_id) as tr:
        ids = tr.result.save()
    return ids
def tries(fn, *a):
    for i in range(4):
        try: return fn(*a)
        except Exception as e: print("retry", i, repr(e)[-600:], flush=True)
    raise SystemExit("gave up")
def main():
    items = [(td, c, q) for c in CUT for td in TODAY for q in QS]
    texts = [prompt(*it) for it in items]
    Hs, Zs, ans = [], [], []
    for s in range(0, len(texts), 10):
        h, z = tries(_trace, texts[s:s + 10]); Hs.append(h.half()); Zs.append(z)
        ch = texts[s:s + 10]; ids = tries(_gen, ch)
        n_in = tok(ch, return_tensors="pt", padding=True, add_special_tokens=False).input_ids.shape[1]
        ans += [tok.decode(r[n_in:], skip_special_tokens=True).strip() for r in ids]
        print(s + 10, "/", len(texts), ans[-1][:30], flush=True)
    torch.save({"items": items, "H": torch.cat(Hs), "Z": torch.cat(Zs), "ans": ans, "cand": {"dec": DEC, "num": NUM}},
               "/home/three/interp/wsp/twonows70.pt"); print("saved")
if __name__ == "__main__": main()
