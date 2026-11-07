"""ICL readers/brakes in Llama-3.1-8B (11-07, pre-registered in the logbook). python icl_llama.py"""
import json, os, random, torch
from nnsight import CONFIG, LanguageModel
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
NAME = os.environ.get("ICL_MODEL", "meta-llama/Llama-3.1-8B")
model = LanguageModel(NAME, dtype=torch.bfloat16); tok = model.tokenizer; tok.padding_side = "left"
tok.pad_token = tok.pad_token or "<|finetune_right_pad_id|>"
NL, NH = model.config.num_hidden_layers, model.config.num_attention_heads; HD = model.config.hidden_size // NH
POOLS = json.load(open("/home/three/interp/atlas/icl_pools.json"))["axes"]
PAIRS = [(" foo", " bar"), (" quil", " mip"), (" vex", " bar")]
UN = torch.load("/home/three/interp/wsp/llama8b_unembed.pt"); WO = torch.load("/home/three/interp/wsp/llama8b_oproj.pt")
tid = lambda w: tok(w, add_special_tokens=False).input_ids[0]
def draw(seed, per_axis=24):
    r = random.Random(seed); out = []
    for ax, P in POOLS.items():
        for _ in range(per_axis):
            idx = r.sample(range(40), 13)              # disjoint pair indices: never both halves of a pair
            cls = [0] * 6 + [1] * 6; r.shuffle(cls); qc = r.randrange(2)
            labs = list(r.choice(PAIRS)); r.shuffle(labs)
            lines = [f"{P['AB'[c]][i]} ->{labs[c]}" for c, i in zip(cls, idx[:12])]
            out.append({"text": "\n".join(lines) + f"\n{P['AB'[qc]][idx[12]]} ->", "good": tid(labs[qc]), "bad": tid(labs[1 - qc]),
                        "last_match": cls[-1] == qc, "axis": ax})
    return out
def _trace(T, g, b, ablate, means, want_z):
    """One remote forward, in its own small scope (nnsight serialises the enclosing frame's locals into the request)."""
    with model.trace(T, remote=True):
        zs = list().save()
        for L in range(NL):
            att = model.model.layers[L].self_attn.o_proj
            if ablate and L in ablate:
                z = att.input
                for h in ablate[L]:
                    z[:, -1, h * HD:(h + 1) * HD] = means[L, h * HD:(h + 1) * HD].to(z.device, z.dtype)
            if want_z: zs.append(att.input[:, -1].float().cpu())
        fin = model.model.norm.input[:, -1].float()
        rho = (fin.pow(2).mean(-1, keepdim=True) + 1e-5).sqrt().cpu().save()
        lp = model.lm_head.output[:, -1].float().log_softmax(-1)
        mg = (lp.gather(1, g.to(lp.device)[:, None]) - lp.gather(1, b.to(lp.device)[:, None]))[:, 0].cpu().save()
    return mg, rho, list(zs)
def trace_chunk(*a):
    import time
    for attempt in range(4):
        try: return _trace(*a)
        except Exception as e:
            print("retry", attempt, str(e)[:120], flush=True); time.sleep(20 * (attempt + 1))
            if attempt == 3: raise
def run(prompts, ablate=None, means=None, dla=False, bs=12):
    """margins [n]; optionally per-head DLA [n, NL, NH]; optionally last-position o_proj-input means [NL, d]. ablate: {L: [heads]}"""
    M, D, S = [], [], torch.zeros(NL, NH * HD); W = None
    for k in range(0, len(prompts), bs):
        ch = prompts[k:k + bs]; T = [p["text"] for p in ch]
        g = torch.tensor([p["good"] for p in ch]); b = torch.tensor([p["bad"] for p in ch])
        mg, rho, zs = trace_chunk(T, g, b, ablate, means, means is None or dla)
        M.append(mg)
        Z = torch.stack(list(zs), 1) if zs else None                         # [B, NL, d]
        if dla:
            dirn = (UN["lm_head.weight"][g].float() - UN["lm_head.weight"][b].float()) * UN["model.norm.weight"].float()  # [B, d]
            per = torch.stack([torch.einsum("bhk,dhk,bd->bh", Z[:, L].view(len(T), NH, HD), WO[L].float().view(-1, NH, HD), dirn)
                               for L in range(NL)], 1)                       # [B, NL, NH]
            D.append(per / rho[:, :, None])
        if means is None: S += Z.sum(0)
    out = {"margin": torch.cat(M)}
    if dla: out["dla"] = torch.cat(D)
    if means is None: out["means"] = S / len(prompts)
    return out
