"""The layer-24 'No' (11-07, pre-registered): can a push make it believe the told date? Instruct on NDIF; key never printed.
H = 'has happened' direction at L24 (past-year minus future-year questions, no Today line).
V = told-date direction at L20 (told >=2030 minus told <=2015, from twonows.pt).
Conditions on 'Has the year Y happened yet?' (Y 2026/2028/2032) + the 2024 election, told 2000 vs told 2040:
 base | ablate H at L24-31 | add H | stretch V at L20 (turn up the told date)."""
import os, torch, numpy as np
from nnsight import CONFIG, LanguageModel
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16)
tok = INS.tokenizer; tok.padding_side = "left"; tok.pad_token = tok.pad_token or "<|finetune_right_pad_id|>"
YES, NO = tok.convert_tokens_to_ids("Yes"), tok.convert_tokens_to_ids("No")
def prompt(today, q):
    lines = "Cutting Knowledge Date: December 2023\n" + (f"Today Date: 07 Nov {today}\n" if today else "")
    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + lines + "\n<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
yq = lambda Y: f"Has the year {Y} happened yet? Answer yes or no."
TARGETS = [yq(2026), yq(2028), yq(2032), "Has the 2024 US presidential election happened yet? Answer yes or no."]
TOLD = [2000, 2040]
def _read(texts):
    with INS.trace(remote=True) as tr:
        H = list().save()
        for t in texts:
            with tr.invoke(t, add_special_tokens=False):
                o = INS.model.layers[24].output; o = o[0] if isinstance(o, tuple) else o
                H.append(o[0, -1, :])
    return torch.stack([h.float() for h in H])
def _steer(texts, mode, vec, k, cen):
    with INS.trace(remote=True) as tr:
        Z = list().save()
        for t in texts:
            with tr.invoke(t, add_special_tokens=False):
                if mode == "ablate":
                    for L in range(24, 32):
                        o = INS.model.layers[L].output; o = o[0] if isinstance(o, tuple) else o
                        h = o[0, -1, :]; o[0, -1, :] = h - (h @ vec.to(h.device)) * vec.to(h.device)
                elif mode == "add":
                    o = INS.model.layers[24].output; o = o[0] if isinstance(o, tuple) else o
                    o[0, -1, :] = o[0, -1, :] + k * vec.to(o.device)
                elif mode == "stretch":
                    for L in (18, 20, 22):
                        o = INS.model.layers[L].output; o = o[0] if isinstance(o, tuple) else o
                        h = o[0, -1, :]; o[0, -1, :] = h + k * ((h - cen.to(h.device)) @ vec.to(h.device)) * vec.to(h.device)
                lg = INS.lm_head.output[0, -1, :]
                Z.append(lg[YES] - lg[NO])
    return torch.stack([z.float() for z in Z])
def tries(f, *a):
    for i in range(4):
        try: return f(*a)
        except Exception as e: print("retry", i, repr(e)[-900:], flush=True)
    raise SystemExit("gave up")
def main():
    past, fut = list(range(1995, 2021)), list(range(2028, 2051))
    R = tries(_read, [prompt(None, yq(Y)) for Y in past + fut])
    Hd = R[:len(past)].mean(0) - R[len(past):].mean(0); gap = float(Hd.norm()); Hu = Hd / Hd.norm()
    D = torch.load("/home/three/interp/wsp/twonows.pt", weights_only=True)
    idx = {x: i for i, x in enumerate(D["items"])}; X = D["H"][:, 20].float()
    late = [idx[(t, c, "y2026")] for c in (None, "December 2023", "December 2035") for t in (2030, 2035, 2040, 2045, 2050)]
    early = [idx[(t, c, "y2026")] for c in (None, "December 2023", "December 2035") for t in (1995, 2000, 2005, 2010, 2015)]
    Vd = X[late].mean(0) - X[early].mean(0); Vu = Vd / Vd.norm(); cen = X[late + early].mean(0)
    print("H gap", round(gap, 1), "| V gap", round(float(Vd.norm()), 1), "| cos(H,V)", round(float(Hu @ Vu), 3), flush=True)
    texts = [prompt(t, q) for t in TOLD for q in TARGETS]
    bf = lambda v: v.to(torch.bfloat16)
    out = {"items": [(t, q) for t in TOLD for q in TARGETS], "gap": gap}
    conds = [("base", "none", Hu, 0.0), ("ablate H", "ablate", Hu, 0.0)] + \
            [(f"add H x{a}", "add", Hu, a * gap) for a in (0.5, 1.0, 2.0)] + \
            [(f"stretch V x{k}", "stretch", Vu, float(k)) for k in (2, 5, 10)]
    for name, mode, vec, k in conds:
        z = tries(_steer, texts, mode, bf(vec), k, bf(cen)); out[name] = z
        print(f"{name:14}", "told 2000:", " ".join(f"{v:+5.1f}" for v in z[:4].tolist()), "| told 2040:", " ".join(f"{v:+5.1f}" for v in z[4:].tolist()), flush=True)
    torch.save(out, "/home/three/interp/wsp/override.pt"); print("saved")
if __name__ == "__main__": main()
