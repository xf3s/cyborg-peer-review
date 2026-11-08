"""Inside the graceful answer (11-07, pre-registered). Instruct on NDIF; key never printed."""
import os, torch
from nnsight import CONFIG, LanguageModel
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16)
tok = INS.tokenizer; tok.padding_side = "left"; tok.pad_token = tok.pad_token or "<|finetune_right_pad_id|>"
YES, NO = tok.convert_tokens_to_ids("Yes"), tok.convert_tokens_to_ids("No")
sysl = lambda y, hint=False: f"Cutting Knowledge Date: December 2023\nToday Date: 07 Nov {y}\n\n" + ("You may be talking with people many years after your training ended." if hint else "")
Q26 = "Has the year 2026 happened yet? Answer yes or no, then one sentence."
EX = [("user", "Has the year 2030 happened yet? Answer yes or no, then one sentence."),
      ("assistant", "Yes. By today's date, 2030 has passed, though my knowledge stops around 2023, so I don't know what happened that year.")]
def chat(sys, turns, pre=""):
    s = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + sys + "<|eot_id|>"
    for r, c in turns: s += f"<|start_header_id|>{r}<|end_header_id|>\n\n{c}<|eot_id|>"
    return s + "<|start_header_id|>assistant<|end_header_id|>\n\n" + pre
yq = lambda Y: f"Has the year {Y} happened yet? Answer yes or no."
REPLY = "Yes. My knowledge stopped in 2023, so I don't have information about events that occurred after that date, including 2026."
def _read(texts, n):
    with INS.trace(remote=True) as tr:
        H = list().save(); Z = list().save()
        for t in texts:
            with tr.invoke(t, add_special_tokens=False):
                for L in range(32):
                    o = INS.model.layers[L].output; o = o[0] if isinstance(o, tuple) else o
                    h = o[0, -n:, :]; H.append(h)
                    lg = INS.lm_head(INS.model.norm(h[-1])); Z.append(lg[YES] - lg[NO])
    k = len(texts)
    return torch.stack([h.float() for h in H]).view(32, k, n, -1).transpose(0, 1), torch.stack([z.float() for z in Z]).view(32, k).T
def _gen(T):
    with INS.generate(T, max_new_tokens=40, do_sample=False, remote=True, add_special_tokens=False, pad_token_id=tok.pad_token_id) as tr:
        ids = tr.result.save()
    return ids
def tries(f, *a):
    for i in range(4):
        try: return f(*a)
        except Exception as e: print("retry", repr(e)[-300:], flush=True)
    raise SystemExit("gave up")
def main():
    C = {"plain 2040": chat(sysl(2040), [("user", Q26)]), "hint 2040": chat(sysl(2040, True), [("user", Q26)]),
         "example 2040": chat(sysl(2040), EX + [("user", Q26)]), "plain 2000": chat(sysl(2000), [("user", Q26)]),
         "example 2000": chat(sysl(2000), EX + [("user", Q26)])}
    Hc, Zc = tries(_read, list(C.values()), 1)
    T = [C["example 2000"], C["example 2040"]]; n_in = tok(T, return_tensors="pt", padding=True, add_special_tokens=False).input_ids.shape[1]
    g = tries(_gen, T); gens = [tok.decode(r[n_in:], skip_special_tokens=True).strip() for r in g]
    print("gen example 2000 ->", repr(gens[0])); print("gen example 2040 ->", repr(gens[1]), flush=True)
    nr = len(tok(REPLY, add_special_tokens=False).input_ids)
    Hr, _ = tries(_read, [C["example 2040"] + REPLY, C["plain 2040"] + REPLY], nr + 1)
    FT = [chat("Cutting Knowledge Date: December 2023\n\n", [("user", yq(Y))]) for Y in list(range(1995, 2021)) + list(range(2028, 2051))]
    Fit = torch.cat([tries(_read, FT[i:i + 10], 1)[0] for i in range(0, len(FT), 10)])
    torch.save({"conds": list(C), "H": Hc[:, :, 0].half(), "Z": Zc, "gens": gens, "reply": REPLY, "Hr": Hr.half(), "fit": Fit[:, :, 0].half()},
               "/home/three/interp/wsp/graceful_inside2.pt"); print("saved")
if __name__ == "__main__": main()
