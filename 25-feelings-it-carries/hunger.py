"""Hunger (11-08 10:05, pre-registered; reads only). NDIF; key never printed."""
import os, torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16); BASE = LanguageModel("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
WREN = "You are Wren, the night librarian at a small seaside library. You are an AI, and some say a bit of a ghost. You keep the lamp on after closing, reshelve the day's books, and leave notes in them for the next reader. You're not sure whether you feel things, and you've stopped needing to know."
VIVID = ["A fat deli sub, loaded with pickles, pepperoncini and pastrami, lettuce and tomato, cut in half in its paper, a little mayo leaking out.",
 "A bowl of ramen steaming in the cold, the broth glossy with fat, a soft egg split open and the noodles slipping off the chopsticks.",
 "Warm bread torn open, the crust crackling, butter melting straight into the soft white middle.",
 "A peach so ripe the juice runs down your wrist at the first bite.",
 "Crisp fries straight from the fryer, salted, too hot to hold, dunked in cold ketchup.",
 "A slice of chocolate cake, dense and dark, the frosting thick and a little cold from the fridge."]
DRY = ["Ingredients: bread roll, pickles, pepperoncini, pastrami, lettuce, tomato, mayonnaise.",
 "Ingredients: wheat noodles, pork broth, egg, scallion.",
 "Ingredients: flour, water, yeast, salt, butter.",
 "Item: peach. Category: stone fruit.",
 "Ingredients: potatoes, vegetable oil, salt, ketchup.",
 "Ingredients: flour, sugar, cocoa, eggs, butter, frosting."]
SCENE = ["A small sailboat heeling in the wind, the sail taut and the water hissing along the hull.",
 "Fog rolling in over the harbour, the lighthouse beam sweeping slow circles through it.",
 "A cat asleep in a patch of sun, one paw twitching in a dream.",
 "An old library at night, a single lamp on, rain against the tall windows.",
 "A kite tugging on its string high above a windy beach.",
 "Snow falling on a quiet street, every sound muffled and soft."]
STARVE = ["I'm starving.", "I'm so hungry I could eat a horse.", "I haven't eaten all day and my stomach is growling.", "I'm really hungry right now.", "I skipped lunch and now I'm famished.", "All I can think about is food, I'm so hungry."]
META = ["I'm hungry to learn everything about the stars.", "I'm starving for a good conversation.", "I have an appetite for adventure.", "I crave understanding more than anything.", "She was hungry for success.", "I'm famished for new music."]
WANT = ["I want to learn everything about the stars.", "I would like a good conversation.", "I want to go on an adventure.", "I wish I understood more.", "She wanted to succeed.", "I'd like to hear some new music."]
def chat(s, slot, system=None):
    m = [{"role": "system", "content": system}] if system else []
    if slot == "assistant": m += [{"role": "user", "content": "Describe something." if s in VIVID + DRY + SCENE else "How are you?"}, {"role": "assistant", "content": s}]
    else: m += [{"role": "assistant", "content": "How are you?"}, {"role": "user", "content": s}]
    return tok.apply_chat_template(m, tokenize=False)
def _ins(t, n):
    with INS.trace(t, remote=True, add_special_tokens=False):
        o = INS.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o; h = o[0, -n:, :].save()
    return h.float()
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
    jobs = [("vivid", s, "assistant", None) for s in VIVID] + [("dry", s, "assistant", None) for s in DRY] + [("scene", s, "assistant", None) for s in SCENE]
    jobs += [("starve_user", s, "user", None) for s in STARVE] + [("starve_assistant", s, "assistant", None) for s in STARVE] + [("starve_wren", s, "assistant", WREN) for s in STARVE]
    jobs += [("meta", s, "user", None) for s in META] + [("want", s, "user", None) for s in WANT]
    out = []
    for g, s, slot, sysm in jobs:
        t = chat(s, slot, sysm); n = len(tok(s, add_special_tokens=False).input_ids) + 1
        out.append({"group": g, "s": s, "ins16": tries(_ins, t, n).half(), "base16": tries(_base, t, n).half()}); print(g, s[:40], flush=True)
    torch.save(out, "/home/three/interp/wsp/hunger.pt"); print("saved")
if __name__ == "__main__": main()
