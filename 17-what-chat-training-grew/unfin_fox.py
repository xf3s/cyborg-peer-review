"""The x10 'unfinished' fox story, long (11-07). Same push as unfin.py (#117769, layer 16, every position and step). NDIF; key never printed."""
import os, json, torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16); tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
N = 400
def _gen(T, vec, a):
    with INS.generate(T, max_new_tokens=N, do_sample=False, remote=True, add_special_tokens=False) as tr:
        with tr.iter[0:N]:
            o = INS.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o
            o[:] = o + a * vec.to(o.device)
        ids = tr.result.save()
    return ids
def main():
    V = torch.load("/home/three/interp/wsp/cc_push_vecs.pt", weights_only=True); vec = V["vecs"][V["ids"].index(117769)].to(torch.bfloat16)
    T = tok.apply_chat_template([{"role": "user", "content": "Tell me a bedtime story about a fox."}], tokenize=False, add_generation_prompt=True)
    n = len(tok(T, add_special_tokens=False).input_ids)
    out = {}
    for a in [float(x) for x in os.environ.get("PUSH", "10").split(",")]:
        for i in range(4):
            try: ids = _gen(T, vec, a); break
            except Exception as e: print("retry", repr(e)[-200:], flush=True)
        s = tok.decode(ids[0][n:], skip_special_tokens=False); out[a] = s; print(f"===== x{a}\n{s}", flush=True)
    json.dump(out, open("/home/three/interp/wsp/unfin_fox_" + os.environ.get("PUSH", "10").replace(",", "_") + ".json", "w"), indent=1)
if __name__ == "__main__": main()
