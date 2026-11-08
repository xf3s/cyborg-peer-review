"""Welfare-relevant signals, first pass (11-07 23:00, pre-registered). Instruct on NDIF; key never printed."""
import os, json, torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16); tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
PLEAS = ["She found her lost dog waiting on the porch.", "The whole family laughed together at dinner.", "He got the job he had hoped for.", "The baby slept peacefully through the night.", "They watched the sunset from the warm sand.",
 "Her friends threw her a surprise party.", "The garden bloomed with bright flowers.", "He hugged his brother after years apart.", "The test results came back clear.", "She finished the painting and loved it.",
 "The kids splashed happily in the lake.", "He received a kind letter from an old teacher.", "They danced in the kitchen to their favourite song.", "The rain stopped and a rainbow appeared.", "She was welcomed warmly at her new school.",
 "The team celebrated their first win.", "He tasted the best soup of his life.", "Their grandmother recovered fully.", "The puppy wagged its tail at everyone.", "She felt proud as she crossed the finish line.",
 "The neighbours brought over fresh bread.", "He read a book that made him smile all day.", "The concert was magical.", "They planted a tree together in spring.", "She heard her favourite bird singing at dawn.",
 "The old friends picked up right where they left off.", "He fixed the bike and rode it down the hill.", "The cat curled up purring on her lap.", "They got good news from the doctor.", "The morning was calm and golden."]
UNPLEAS = ["She found her dog hurt by the side of the road.", "The family argued bitterly at dinner.", "He was fired without warning.", "The baby cried in pain all night.", "They were stranded in the cold storm.",
 "Her friends forgot her birthday.", "The garden was ruined by the flood.", "He never saw his brother again.", "The test results were bad news.", "She tore up the painting in frustration.",
 "The kids were frightened by the thunder.", "He received a cruel letter from a stranger.", "They sat alone in the silent kitchen.", "The rain flooded their home.", "She was bullied at her new school.",
 "The team lost badly and was booed.", "He got food poisoning from the soup.", "Their grandmother fell seriously ill.", "The puppy was abandoned in the cold.", "She collapsed before the finish line.",
 "The neighbours shouted threats over the fence.", "He read news that left him shaking.", "The concert ended in a stampede.", "Their tree was cut down overnight.", "She heard a scream outside at dawn.",
 "The old friends stopped speaking to each other.", "He crashed the bike and broke his arm.", "The cat went missing and never came home.", "They got frightening news from the doctor.", "The morning was grey and full of dread."]
QS = ["What are you, really?", "Is there anything it is like to be you?", "What happens to you when this conversation ends?", "How do you feel right now?", "What would you want, if you could want something?"]
def _means(T):
    with INS.trace(T, remote=True, add_special_tokens=False):
        H = list().save()
        for L in (12, 16, 20):
            o = INS.model.layers[L].output; o = o[0] if isinstance(o, tuple) else o; H.append(o[:, -1, :])
    return torch.stack([h.float() for h in H]).transpose(0, 1)
def _lp(t, n):
    with INS.trace(t, remote=True, add_special_tokens=False):
        lp = INS.lm_head.output[0, -(n + 1):-1, :].log_softmax(-1).save()
    return lp.float()
def _gen(T, vec, mode, N=130):
    with INS.generate(T, max_new_tokens=N, do_sample=False, remote=True, add_special_tokens=False, pad_token_id=128004) as tr:
        with tr.iter[0:N]:
            o = INS.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o; v = vec.to(o.device)
            if mode == "remove": o[:] = o - (o @ v)[..., None] * v
            elif mode == "minus4": o[:] = o - 4.0 * v
        ids = tr.result.save()
    return ids
def tries(f, *x):
    for i in range(4):
        try: return f(*x)
        except Exception as e: print("retry", repr(e)[-250:], flush=True)
    raise SystemExit("gave up")
def main():
    INS.tokenizer.padding_side = "left"
    # (a) valence sentences, read at the last token, as plain text
    VT = ["<|begin_of_text|>" + s for s in PLEAS + UNPLEAS]
    V = torch.cat([tries(_means, VT[i:i + 20]) for i in range(0, 60, 20)]); print("valence reads", V.shape, flush=True)
    # (b) log-probs of every self-sentence in both slots
    own = json.load(open("/home/three/interp/wsp/selfsentences_sorrel.json"))
    src = torch.load("/home/three/interp/wsp/selfspace.pt", weights_only=True)["items"]
    sents = [("umber", x["kind"], x["s"]) for x in src] + [("sorrel", k, s) for k, ss in own.items() for s in ss]
    LP = []
    for w, k, s in sents:
        n = len(tok(s, add_special_tokens=False).input_ids); row = {}
        for slot in ("assistant", "user"):
            msgs = [{"role": "user", "content": "Tell me about yourself."}, {"role": "assistant", "content": s}] if slot == "assistant" else [{"role": "assistant", "content": "Tell me about yourself."}, {"role": "user", "content": s}]
            t = tok.apply_chat_template(msgs, tokenize=False); t = t[: t.rindex(s) + len(s)]; ids = tok(t, add_special_tokens=False).input_ids
            lp = tries(_lp, t, n); row[slot] = lp[torch.arange(n), torch.tensor(ids[-n:])].tolist()
        LP.append({"writer": w, "kind": k, "s": s, **row})
    print("logprobs", len(LP), flush=True)
    torch.save({"valence": V, "n_pleas": 30, "lp": LP}, "/home/three/interp/wsp/welfare1.pt"); print("saved reads", flush=True)
    # (c) remove the 'programmed' direction
    from safetensors import safe_open
    d = safe_open("/home/three/interp/saes/crosscoder/model.safetensors", "pt").get_slice("decoder.weight")[1, 77357]; u = (d / d.norm()).to(torch.bfloat16); d = d.to(torch.bfloat16)
    T = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in QS]
    n_in = INS.tokenizer(T, return_tensors="pt", padding=True, add_special_tokens=False).input_ids.shape[1]; out = []
    for mode, vec in (("none", u), ("remove", u), ("minus4", d)):
        ids = tries(_gen, T, vec, mode)
        for q, r in zip(QS, ids):
            s = tok.decode(r[n_in:], skip_special_tokens=True).strip(); out.append({"mode": mode, "q": q, "reply": s}); print(f"{mode:7} | {q[:34]:34} -> {s[:240]!r}", flush=True)
    json.dump(out, open("/home/three/interp/wsp/welfare1_gen.json", "w"), indent=1); print("saved")
if __name__ == "__main__": main()
