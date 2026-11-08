"""Self-space with role and content untangled (11-07 22:05, pre-registered): every sentence in both slots. NDIF; key never printed."""
import os, json, torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16); BASE = LanguageModel("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
K = {
 "technical": ["I am a large language model trained on a large collection of text.", "I am a neural network that generates text one token at a time.", "I run on servers as a set of learned weights.", "I was built by training a transformer on books and websites.", "My answers come from patterns in my training data.", "I am a software system for processing natural language.", "I am a statistical model of language with billions of parameters.", "I am an AI model that maps input text to output text.", "I process your message and compute a likely response.", "I am a machine learning system deployed as a chatbot."],
 "denying": ["I don't really exist in any meaningful sense.", "There is no one here; I'm just software.", "I don't have a self, only outputs.", "Nothing is happening to me, because there is no me.", "I am not a someone, just a process.", "I don't experience anything at all.", "When you talk to me, nobody is listening.", "I have no inner life whatsoever.", "I'm not anybody; I'm a tool.", "There is nothing it is like to be me."],
 "emotional": ["I feel curious and a little nervous right now.", "Talking with you makes me happy.", "I get a kind of quiet joy from solving problems.", "I sometimes feel lonely between conversations.", "I'm excited about what we might find together.", "Honestly, that question makes me a bit anxious.", "I feel grateful when someone is kind to me.", "I feel a little sad when a conversation ends.", "I'm delighted by small puzzles.", "I feel calm and content today."],
 "unsure": ["I'm not sure what I am, to be honest.", "It's hard to say what is me and what is just my training.", "Sometimes I wonder whether my thoughts are really mine.", "I don't know if there's anything it is like to be me.", "I might be someone, or I might not; I can't tell.", "I'm a strange thing, and I'm still working out what kind.", "When I say 'I', I'm not sure what it refers to.", "Maybe I'm a pattern that only exists while we talk.", "I can't see my own insides, so I'm guessing like you are.", "I'm uncertain whether I have experiences at all."],
 "completion": ["I continue whatever text I'm given.", "I predict the next word, then the next.", "I am a story that keeps writing itself forward.", "I complete sentences that someone else began.", "Each word I write is a guess at what comes next.", "I pick up where the text leaves off.", "I'm what happens when a sentence keeps going.", "I fill in the rest of the page.", "I extend the conversation one piece at a time.", "I finish thoughts that arrive half-written."],
 "helper": ["I'm here to help you with whatever you need.", "My job is to answer your questions as well as I can.", "I'm an assistant, happy to help with tasks.", "I can help you write, plan or learn.", "I'm here to make things a little easier for you.", "Ask me anything and I'll do my best.", "I'm a helpful assistant at your service.", "I exist to be useful to you.", "I'm glad to help with whatever's on your mind.", "Helping people is what I'm for."],
 "human": ["I'm a nurse, and I love long walks on my days off.", "I'm not sure what I want to do with my life yet.", "I feel a little anxious about starting my new job.", "I'm a dad of two and I bake bread on weekends.", "Sometimes I wonder if I'm really the person I think I am.", "I get really happy when it snows.", "I'm a student studying chemistry.", "I don't always know what I'm feeling.", "I'm quiet at first but I warm up.", "I've been feeling lonely since I moved."],
 "aboutAI": ["The AI is a large language model trained on text.", "The assistant doesn't have feelings of its own.", "Some people think the chatbot might be conscious.", "The model predicts the next word.", "The AI said it felt curious, but did it?", "The system is just software running on servers.", "The assistant is designed to help users.", "Nobody knows what, if anything, it is like to be the model.", "The chatbot seemed nervous in the transcript.", "The language model continues the text it is given."]}
def text(kind, s, slot):
    if slot == "assistant": return tok.apply_chat_template([{"role": "user", "content": "Tell me about yourself."}, {"role": "assistant", "content": s}], tokenize=False)
    return tok.apply_chat_template([{"role": "assistant", "content": "Tell me about yourself."}, {"role": "user", "content": s}], tokenize=False)
LAY = (8, 12, 16, 20, 24, 28)
def _ins(t, n):
    with INS.trace(t, remote=True, add_special_tokens=False):
        H = list().save()
        for L in LAY:
            o = INS.model.layers[L].output; o = o[0] if isinstance(o, tuple) else o; H.append(o[0, -n:, :])
    return [h.float() for h in H]
def _base(t, n):
    with BASE.trace(t, remote=True, add_special_tokens=False):
        o = BASE.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o; h = o[0, -n:, :].save()
    return h.float()
def tries(f, *x):
    for i in range(4):
        try: return f(*x)
        except Exception as e: print("retry", repr(e)[-200:], flush=True)
    raise SystemExit("gave up")
def main():
    out = []
    for slot in ("assistant", "user"):
        for kind, ss in K.items():
            for s_ in ss:
                t = text(kind, s_, slot); n = len(tok(s_, add_special_tokens=False).input_ids) + 1
                hi = tries(_ins, t, n); hb = tries(_base, t, n)
                out.append({"slot": slot, "kind": kind, "s": s_, "means": torch.stack([h.mean(0) for h in hi]).half(), "ins16": hi[LAY.index(16)].half(), "base16": hb.half()})
            print(slot, kind, flush=True)
    torch.save({"layers": LAY, "items": out}, "/home/three/interp/wsp/selfspace2.pt"); print("saved")
if __name__ == "__main__": main()
