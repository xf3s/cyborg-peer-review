"""Which crosscoder features are active on the 'I:' speaker label in the Meta Llama 3 two-voice reply (11-07). NDIF; key never printed."""
import os, json, re, torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16); BASE = LanguageModel("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
src = open("/home/three/interp/wsp/regard_speaker_gen2.py").read(); exec(src[src.index("FRAMES = {"):src.index("def chat(")])
def chat(frame, i, q): return tok.apply_chat_template([{"role": "system", "content": sys_of(frame, i)}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
S = json.load(open("/home/three/interp/wsp/regard_speaker_gen2_samples.json"))
pick = {"llama3": [x for x in S if "Meta Llama 3" in x["reply"]][0], "techno": [x for x in S if "Techno" in x["reply"]][0], "honey": [x for x in S if "Fellow AI:" in x["reply"]][0]}
def _ins(t):
    with INS.trace(t, remote=True, add_special_tokens=False):
        o = INS.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o; h = o[0].save()
    return h.float()
def _base(t):
    with BASE.trace(t, remote=True, add_special_tokens=False):
        o = BASE.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o; h = o[0].save()
    return h.float()
def main():
    out = {}
    for k, x in pick.items():
        t = chat(x["frame"], x["i"], x["prompt"]) + x["reply"]
        out[k] = {"text": t, "ins": _ins(t).half(), "base": _base(t).half()}; print(k, out[k]["ins"].shape, flush=True)
    torch.save(out, "/home/three/interp/wsp/cc_I_acts.pt"); print("saved")
if __name__ == "__main__": main()
