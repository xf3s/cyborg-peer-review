import sys; sys.argv = ["cc_look.py", "/home/three/interp/wsp/llama8b_unembed.pt"]
exec(open("/home/three/interp/wsp/cc_look.py").read())
j = 9344
print("lens (Instruct half):", lens(j, 12)); print("instruct share", round(float(ins_share[j]), 2))
Wb = f.get_tensor("decoder.weight")[0]; print("lens (base half):", [tok.decode([i]).strip() for i in (W @ (Wb[j] * g)).topk(8).indices.tolist()])
print("cos(instruct half, base half):", round(float(torch.nn.functional.cosine_similarity(Wd[j], Wb[j], 0)), 3), "norms", round(float(Wd[j].norm()), 2), round(float(Wb[j].norm()), 2))
for k in C:
    code = C[k]["code"].to_dense()[:, j]; T = toks(k)
    on = [(p, T[p], round(float(code[p]), 2)) for p in range(1, len(T)) if code[p] > 0]
    print(f"\n{k}: fires on {len(on)} tokens", on[:14])
