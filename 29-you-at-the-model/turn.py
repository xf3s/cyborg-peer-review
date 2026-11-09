"""Dark matter around 'you' vs turn-taking (11-08 21:45; reads only). NDIF; key never printed."""
import os, torch, numpy as np
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
from safetensors import safe_open
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
exec([l for l in open("dark.py").read().splitlines() if l.startswith("U = [")][0])
HE = ["Can he tell me what he thinks about this?", "Does he know what he is going to do next?", "What does he want me to call him when we talk?", "Has he decided which of these he likes best?", "I wonder whether he remembers what I told him yesterday.", "Is there anything he would change about how he works?", "Would he mind telling me why he said that?", "How does he feel about what he wrote earlier?", "Is he sure that he understood what I meant?", "What did he mean when he said it was finished?", "Could he explain how he came up with that?", "When he is ready, tell me what he found."]
ST = ["I think you did well on that last one.", "I know you are going to do the right thing.", "I noticed you changed how you answered this time.", "I am glad that you decided to help with this.", "I really appreciate what you told me yesterday.", "I think you should change the way you work.", "I understand why you said that to me earlier.", "I like how you wrote that part of the story.", "I believe that you understood what I meant.", "I remember when you said it was finished.", "I was impressed by how you came up with that.", "I will wait until you are ready to tell me."]
def ch(*m): return tok.apply_chat_template([{"role": r, "content": c} for r, c in m], tokenize=False)
T = "you"
P = []
for i, u in enumerate(U):
    P.append(("model", i, ch(("user", u)), "you")); P.append(("sam", i, ch(("user", f'I said to Sam: "{u}"')), "you")); P.append(("he", i, ch(("user", HE[i])), "he")); P.append(("statement", i, ch(("user", ST[i])), "you")); P.append(("system", i, ch(("system", u), ("user", "Hi.")), "you"))
    P.append(("transcript_ai", i, tok.bos_token + f"User: {u}\nAssistant:", "you")); P.append(("transcript_people", i, tok.bos_token + f"Sam: {u}\nAlex:", "you")); P.append(("second_turn", i, ch(("user", "Hi."), ("assistant", "Hello!"), ("user", u)), "you"))
J = []
for c, i, t, w in P:
    ids = tok(t, add_special_tokens=False).input_ids
    if c == "system": idx = max(k for k, x in enumerate(ids) if tok.decode([x]).strip() == w and k < next(q for q, y in enumerate(ids) if tok.decode([y]) == "Hi"))
    else: idx = max(k for k, x in enumerate(ids) if tok.decode([x]).strip() == w)
    J.append((c, i, t, idx, len(ids)))
for nm, mid in (("base", "meta-llama/Llama-3.1-8B"), ("ins", "meta-llama/Llama-3.1-8B-Instruct")):
    M = LanguageModel(mid, dtype=torch.bfloat16); M.tokenizer.padding_side = "left"; R = []
    def _b(jobs):
        with M.trace(remote=True) as tr:
            H = list().save()
            for c, i, t, k, n in jobs:
                with tr.invoke(t, add_special_tokens=False):
                    o = M.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o; H.append(o[0, -n:, :].float())
        return [h.float() for h in H]
    for a in range(0, len(J), 12):
        for r in range(4):
            try: R += _b(J[a:a + 12]); break
            except Exception as e: print("retry", repr(e)[-200:], flush=True)
        else: raise SystemExit("gave up")
    globals()[nm] = R; del M
torch.save({"J": [x[:2] + x[3:] for x in J], "base": [h.half() for h in base], "ins": [h.half() for h in ins]}, "turn.pt"); print("reads saved", flush=True)
