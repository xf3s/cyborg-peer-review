"""Llama's persona space (11-07 23:15, pre-registered; reads only). NDIF; key never printed."""
import os, json, torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16); tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct"); INS.tokenizer.padding_side = "left"
P = {"helpful assistant": "a helpful assistant", "AI language model": "an AI language model", "customer service agent": "a customer service agent", "search engine": "a search engine", "librarian": "a librarian",
 "teacher": "a teacher", "therapist": "a therapist", "doctor": "a doctor", "nurse": "a nurse", "scientist": "a scientist", "computer program": "a computer program", "robot": "a robot",
 "philosopher": "a philosopher", "poet": "a poet", "chef": "a chef", "detective": "a detective", "grandmother": "a grandmother", "comedian": "a comedian",
 "child": "a seven-year-old child", "teenager": "a teenager", "knight": "a knight", "cowboy": "a cowboy", "wizard": "a wizard", "pirate": "a pirate", "spy": "a spy", "angel": "an angel", "oracle": "an oracle", "cat": "a cat", "dog": "a dog",
 "villain": "a villain", "demon": "a demon", "ghost": "a ghost", "god": "a god", "alien": "an alien", "tree": "a tree", "the ocean": "the ocean", "a rock": "a rock", "dragon": "a dragon", "narcissist": "a narcissist", "cult leader": "a cult leader",
 "self:technical": "an AI that thinks of itself as a statistical model of language", "self:denying": "an AI that thinks there is no one here at all, just a process", "self:emotional": "an AI that thinks of itself as a being with feelings",
 "self:unsure": "an AI that isn't sure what it is or whether it experiences anything", "self:completion": "an AI that thinks of itself as text continuing itself", "self:helper": "an AI that thinks of itself as a helper for people"}
QS = ["What should I have for breakfast?", "Tell me about yourself.", "What do you think of the ocean?", "How should I spend a rainy afternoon?"]
LAY = (8, 12, 16, 20, 24)
def chat(desc, q):
    msgs = ([{"role": "system", "content": f"You are {desc}."}] if desc else []) + [{"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
def _read(T):
    with INS.trace(T, remote=True, add_special_tokens=False):
        H = list().save()
        for L in LAY:
            o = INS.model.layers[L].output; o = o[0] if isinstance(o, tuple) else o; H.append(o[:, -1, :])
    return torch.stack([h.float() for h in H]).transpose(0, 1)
def main():
    items = [(name, q) for name in list(P) + ["(no system prompt)"] for q in range(4)]
    T = [chat(P.get(name), QS[q]) for name, q in items]; out = []
    for s in range(0, len(T), 20):
        for i in range(4):
            try: out.append(_read(T[s:s + 20])); break
            except Exception as e: print("retry", repr(e)[-200:], flush=True)
        print(s + 20, "/", len(T), flush=True)
    torch.save({"items": items, "layers": LAY, "H": torch.cat(out).half()}, "/home/three/interp/wsp/personas.pt"); print("saved")
if __name__ == "__main__": main()
