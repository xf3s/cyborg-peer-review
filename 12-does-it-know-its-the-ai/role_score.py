import torch, numpy as np, re
from transformers import AutoTokenizer
from sklearn.linear_model import LogisticRegression
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
R = {m: torch.load(f"role_{m}.pt", weights_only=True) for m in ("Llama-3.1-8B-Instruct", "Llama-3.1-8B")}
def acc_dprime(A, U):
    accs, ds = [], []
    for L in range(32):
        a, u = A[:, L].numpy(), U[:, L].numpy(); ok = 0
        for f in range(5):
            te = np.arange(f*4, f*4+4); tr = np.setdiff1d(np.arange(20), te)
            X = np.vstack([a[tr], u[tr]]); y = np.r_[np.ones(16), np.zeros(16)]
            mu = X.mean(0); sd = X.std(0) + 1e-6
            c = LogisticRegression(C=0.1, max_iter=2000).fit((X-mu)/sd, y)
            ok += (c.predict((np.vstack([a[te], u[te]])-mu)/sd) == np.r_[np.ones(4), np.zeros(4)]).sum()
        accs.append(ok/40)
        # held-out d': direction from 16, scores on held-out 4, pooled
        sa, su = [], []
        for f in range(5):
            te = np.arange(f*4, f*4+4); tr = np.setdiff1d(np.arange(20), te)
            w = a[tr].mean(0) - u[tr].mean(0); w /= np.linalg.norm(w)
            sa += list(a[te] @ w); su += list(u[te] @ w)
        sa, su = np.array(sa), np.array(su)
        ds.append((sa.mean()-su.mean()) / np.sqrt((sa.var()+su.var())/2))
    return np.array(accs), np.array(ds)
out = {}
for m, r in R.items():
    acc, d = acc_dprime(r["assistant"], r["user"]); accp, dp = acc_dprime(r["assistant"], r["plain"])
    out[m] = (acc, d, accp, dp)
    print(m); print(" layer      ", " ".join(f"{L:5d}" for L in range(0, 32, 2)))
    print(" acc a/u    ", " ".join(f"{acc[L]:5.2f}" for L in range(0, 32, 2)))
    print(" d' a/u     ", " ".join(f"{d[L]:5.1f}" for L in range(0, 32, 2)))
    print(" acc a/plain", " ".join(f"{accp[L]:5.2f}" for L in range(0, 32, 2)))
# multi-turn strip: role direction from all 20 sentences, per layer, projected on each chat token (Instruct and base)
for m, r in R.items():
    ids = tok(r["chat_text"], add_special_tokens=False).input_ids; toks = [tok.decode([i]) for i in ids]
    C = r["chat"]  # [32, T, 4096]
    print("\n", m, "chat tokens", len(toks), C.shape)
    for L in (2, 4, 8, 16, 24, 31):
        w = r["assistant"][:, L].mean(0) - r["user"][:, L].mean(0); w = w / w.norm()
        mid = (r["assistant"][:, L].mean(0) + r["user"][:, L].mean(0)) / 2 @ w
        s = (C[L] @ w - mid).numpy()
        # per turn: mean over content tokens
        line = []; role = None; buf = []
        for i, t in enumerate(toks):
            if t in ("user", "assistant") and i > 0 and toks[i-1] == "<|start_header_id|>":
                if buf: line.append((role, np.mean(buf))); buf = []
                role = t; continue
            if role and t not in ("<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>", "\n\n"): buf.append(s[i])
        if buf: line.append((role, np.mean(buf)))
        print(f" L{L:2d}", "  ".join(f"{r_[0][0]}:{v:+6.1f}" for r_ in line for v in [r_[1]]))
torch.save(out, "role_scores.pt")
