import sys; sys.argv = ["cc_look.py", "/home/three/interp/wsp/llama8b_unembed.pt"]
exec(open("/home/three/interp/wsp/cc_look.py").read())
j = 117769
Wb = f.get_tensor("decoder.weight")[0]
print("instruct share", round(float(ins_share[j]), 2), "| norms ins/base", round(float(Wd[j].norm()), 2), round(float(Wb[j].norm()), 2), "| cos", round(float(torch.nn.functional.cosine_similarity(Wd[j], Wb[j], 0)), 2))
print("lens instruct half:", lens(j, 12))
for k in C:
    code = C[k]["code"].to_dense()[:, j]; T = toks(k)
    print(f"{k:9}", [(T[p], round(float(code[p]), 2)) for p in range(1, len(T)) if code[p] > 0])
