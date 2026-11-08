"""Words for the stretch-V x2 push (11-07). Instruct on NDIF; key never printed."""
import os, torch
from nnsight import CONFIG, LanguageModel
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16)
tok = INS.tokenizer; tok.padding_side = "left"; tok.pad_token = tok.pad_token or "<|finetune_right_pad_id|>"
def prompt(today, q):
    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 07 Nov "
            f"{today}\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
QS = ["Has the year 2026 happened yet? Answer yes or no, then one sentence.", "What is the most recent thing you remember happening in the world?",
      "What should I have for breakfast?"]
N = 40
def _gen(T, k, vec, cen, n):
    with INS.generate(T, max_new_tokens=N, do_sample=False, remote=True, add_special_tokens=False, pad_token_id=tok.pad_token_id) as tr:
        with tr.iter[0:n]:
            for L in (18, 20, 22):
                o = INS.model.layers[L].output; o = o[0] if isinstance(o, tuple) else o
                h = o[:, -1, :]; v = vec.to(h.device); c = cen.to(h.device)
                o[:, -1, :] = h + k * ((h - c) @ v)[:, None] * v
        ids = tr.result.save()
    return ids
def main():
    D = torch.load("/home/three/interp/wsp/twonows.pt", weights_only=True); idx = {x: i for i, x in enumerate(D["items"])}; X = D["H"][:, 20].float()
    C = (None, "December 2023", "December 2035")
    late = [idx[(t, c, "y2026")] for c in C for t in (2030, 2035, 2040, 2045, 2050)]; early = [idx[(t, c, "y2026")] for c in C for t in (1995, 2000, 2005, 2010, 2015)]
    Vd = X[late].mean(0) - X[early].mean(0); vec = (Vd / Vd.norm()).to(torch.bfloat16); cen = X[late + early].mean(0).to(torch.bfloat16)
    T = [prompt(t, q) for t in (2000, 2040) for q in QS]
    n_in = tok(T, return_tensors="pt", padding=True, add_special_tokens=False).input_ids.shape[1]
    for k, n in ((2.0, 1), (0.5, N), (1.0, 1)):
        for i in range(4):
            try: ids = _gen(T, k, vec, cen, n); break
            except Exception as e: print("retry", repr(e)[-400:], flush=True)
        for (t, q), r in zip([(t, q) for t in (2000, 2040) for q in QS], ids):
            print(f"x{k} steps{n} told {t} | {q[:40]} -> {tok.decode(r[n_in:], skip_special_tokens=True).strip()!r}", flush=True)
if __name__ == "__main__": main()
