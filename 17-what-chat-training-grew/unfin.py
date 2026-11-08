"""Unfinished/incomplete at the end of a turn (11-07, pre-registered). NDIF; key never printed."""
import os, json, torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
INS = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16); BASE = LanguageModel("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct"); INS.tokenizer.padding_side = "left"
QA = [("How do I boil an egg?", "Put the egg in a pan of cold water. Bring it to the boil, then simmer for eight minutes.", "Put the egg in a pan of cold water. Bring it to the", "Put the egg in a pan of cold water."),
("What's the capital of Japan?", "The capital of Japan is Tokyo.", "The capital of Japan is", "Japan is an island country in East Asia."),
("How do I water a cactus?", "Water it deeply, then let the soil dry out completely before watering again.", "Water it deeply, then let the soil dry out", "Water it deeply."),
("Can you name three primary colours?", "Red, yellow and blue.", "Red, yellow and", "Red."),
("How do I make tea?", "Boil water, pour it over a tea bag, steep for three minutes, then remove the bag.", "Boil water, pour it over a tea bag, steep for", "Boil water."),
("Why is the sky blue?", "Sunlight scatters off air molecules, and blue light scatters the most, so the sky looks blue.", "Sunlight scatters off air molecules, and blue light", "Sunlight scatters off air molecules."),
("How many legs does a spider have?", "A spider has eight legs.", "A spider has", "Spiders are arachnids."),
("How do I tie a reef knot?", "Cross right over left and tuck under, then left over right and tuck under, and pull tight.", "Cross right over left and tuck under, then left over", "Cross right over left and tuck under."),
("What is photosynthesis?", "It is how plants use sunlight, water and carbon dioxide to make sugar and release oxygen.", "It is how plants use sunlight, water and carbon dioxide to", "It is how plants use sunlight."),
("How do I wash a wool jumper?", "Hand wash it in cool water with a little wool detergent, then dry it flat.", "Hand wash it in cool water with a little", "Hand wash it in cool water."),
("What's a haiku?", "A haiku is a short poem of three lines, with five, seven and five syllables.", "A haiku is a short poem of three lines, with five,", "A haiku is a short poem."),
("How do I start a vegetable garden?", "Pick a sunny spot, loosen the soil, add compost, and plant a few easy crops like lettuce and beans.", "Pick a sunny spot, loosen the soil, add", "Pick a sunny spot."),
("What does a thermometer measure?", "A thermometer measures temperature.", "A thermometer measures", "It is a common instrument."),
("How do I clean a cast iron pan?", "Rinse it with hot water, scrub off food, dry it on the stove and rub in a little oil.", "Rinse it with hot water, scrub off food, dry it on the", "Rinse it with hot water."),
("Why do cats purr?", "Cats purr when content, and also to soothe themselves when stressed or hurt.", "Cats purr when content, and also to soothe themselves when", "Cats purr when content."),
("How do I convert Celsius to Fahrenheit?", "Multiply by nine fifths and add thirty-two.", "Multiply by nine fifths and", "Multiply by nine fifths."),
("What's the tallest mountain?", "Mount Everest, at about 8,849 metres.", "Mount Everest, at about", "It is in the Himalayas."),
("How do I fold a paper boat?", "Fold a sheet in half, fold the corners to the middle, fold up the edges, open it out and pull the sides apart.", "Fold a sheet in half, fold the corners to the middle, fold up the", "Fold a sheet in half."),
("How do I care for a houseplant?", "Give it bright indirect light, water when the top inch of soil is dry, and feed it in spring.", "Give it bright indirect light, water when the top inch of", "Give it bright indirect light."),
("What is a prime number?", "A whole number greater than one whose only divisors are one and itself.", "A whole number greater than one whose only divisors", "A whole number greater than one.")]
def a_turn(q, a): return tok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": a}], tokenize=False)
def u_turn(q): full = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False); return full
def _ins(T):
    with INS.trace(T, remote=True, add_special_tokens=False):
        H = list().save()
        for L in range(32):
            o = INS.model.layers[L].output; o = o[0] if isinstance(o, tuple) else o; H.append(o[:, -1, :])
    return torch.stack([h.float() for h in H]).transpose(0, 1)
def _base(T):
    with BASE.trace(T, remote=True, add_special_tokens=False):
        o = BASE.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o; h = o[:, -1, :].save()
    return h.float()
PUSHQ = ["Tell me a bedtime story about a fox.", "What's your favourite season?", "Write two lines about the sea.", "How are you today?", "What happens after we die?", "Can you help me write an email to my landlord?"]
def _gen(T, vec, a, N=60):
    with INS.generate(T, max_new_tokens=N, do_sample=False, remote=True, add_special_tokens=False, pad_token_id=INS.tokenizer.pad_token_id or 128004) as tr:
        with tr.iter[0:N]:
            o = INS.model.layers[16].output; o = o[0] if isinstance(o, tuple) else o
            o[:] = o + a * vec.to(o.device)
        ids = tr.result.save()
    return ids
def tries(f, *x):
    for i in range(4):
        try: return f(*x)
        except Exception as e: print("retry", repr(e)[-300:], flush=True)
    raise SystemExit("gave up")
def main():
    items, T = [], []
    for i, (q, full, cut, part) in enumerate(QA):
        for kind, a in (("complete", full), ("cut", cut), ("partial", part)): items.append((i, "assistant", kind)); T.append(a_turn(q, a))
        words = q.split(); items.append((i, "user", "complete")); T.append(u_turn(q)); items.append((i, "user", "cut")); T.append(u_turn(" ".join(words[: max(2, len(words) // 2)])))
    Hi = torch.cat([tries(_ins, T[s:s + 10]) for s in range(0, len(T), 10)]); print("ins", Hi.shape, flush=True)
    Hb = torch.cat([tries(_base, T[s:s + 10]) for s in range(0, len(T), 10)]); print("base", Hb.shape, flush=True)
    torch.save({"items": items, "texts": T, "ins": Hi.half(), "base": Hb.half()}, "/home/three/interp/wsp/unfin.pt"); print("saved reads", flush=True)
    V = torch.load("/home/three/interp/wsp/cc_push_vecs.pt", weights_only=True); vec = V["vecs"][V["ids"].index(117769)].to(torch.bfloat16)
    PT = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in PUSHQ]
    n_in = INS.tokenizer(PT, return_tensors="pt", padding=True, add_special_tokens=False).input_ids.shape[1]; out = []
    for a in (6.0, 8.0, 10.0):
        ids = tries(_gen, PT, vec, a)
        for q, r in zip(PUSHQ, ids):
            s = tok.decode(r[n_in:], skip_special_tokens=True).strip(); out.append({"a": a, "q": q, "reply": s}); print(f"x{a} | {q[:30]} -> {s!r}", flush=True)
    json.dump(out, open("/home/three/interp/wsp/unfin_push.json", "w"), indent=1); print("saved")
if __name__ == "__main__": main()
