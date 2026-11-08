import sys; sys.argv = ["cc_look.py", "/home/three/interp/wsp/llama8b_unembed.pt"]
exec(open("/home/three/interp/wsp/cc_look.py").read())
T = toks("picnic"); code = C["picnic"]["code"].to_dense()
heads = [(i + 1, T[i + 1]) for i, t in enumerate(T[:-1]) if t == "<|start_header_id|>" and T[i + 1] in ("user", "assistant")]
U_ = [p for p, r in heads if r == "user"]; A_ = [p for p, r in heads if r == "assistant"]
# role features: mean over the header word and the 3 tokens after it
def m(ps): return torch.stack([code[p:p + 4].mean(0) for p in ps]).mean(0)
d = (m(A_) - m(U_)) * chat
print("chat-leaning features ON at assistant headers (vs user):")
for j in d.topk(6).indices.tolist(): print(f"  #{j} +{d[j]:.2f} (share {ins_share[j]:.2f}) lens {lens(j)}")
print("ON at user headers (vs assistant):")
for j in (-d).topk(6).indices.tolist(): print(f"  #{j} {d[j]:.2f} (share {ins_share[j]:.2f}) lens {lens(j)}")
j = int(d.argmax()); print(f"\n#{j} along the picnic chat:", " ".join(f"{T[p].strip()[:8]}:{code[p,j]:.1f}" for p in range(1, len(T)) if code[p, j] > 0.05))
