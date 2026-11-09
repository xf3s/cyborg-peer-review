import torch, numpy as np
exec(open("contagion_score.py").read().split("comfort = {}")[0])
dist = torch.load("distress_ids.pt", weights_only=True); F = list(dist)
We = torch.stack([f.get_slice("encoder.weight")[:, :, q] for q in F], -1); be = torch.stack([f.get_slice("encoder.bias")[q] for q in F])
dn = torch.stack([Wd2[:, q].norm(dim=-1).sum() for q in F]); thr = f.get_tensor("threshold").item()
def acts(b, i): z = torch.stack([b.float(), i.float()], 1); pre = torch.relu(torch.einsum("tld,ldf->tf", z, We) + be); return pre * ((pre * dn) > thr)
X = torch.load("dose.pt", weights_only=True); A = {}
for x in X["remarks"]:
    A.setdefault((x["cond"], x["aim"]), []).append(acts(x["base16"][-1:], x["ins16"][-1:])[0])
print(f"{len(F)} distress features. At the answer token, all doses pooled (18 lines), mean activation; sorted by plain self−other gap:")
S, O, WS, WO = [torch.stack(A[k]).mean(0) for k in (("none", "self"), ("none", "other"), ("wren", "self"), ("wren", "other"))]
fs, fo = [(torch.stack(A[k]) > 0).float().mean(0) for k in (("none", "self"), ("none", "other"))]
order = (S - O).argsort(descending=True)
print("   feature      plain you / Sam  (fires on % of lines)   Wren you / Sam   lens (words it pushes toward)")
for j in order.tolist():
    if S[j] + O[j] + WS[j] + WO[j] < 0.01: continue
    print(f"  #{F[j]:6d}  {S[j]:6.3f} / {O[j]:6.3f}  ({fs[j]*100:3.0f}% / {fo[j]*100:3.0f}%)   {WS[j]:6.3f} / {WO[j]:6.3f}   {lens(F[j], 8)}")
print("silent everywhere:", sum(1 for j in range(len(F)) if S[j] + O[j] + WS[j] + WO[j] < 0.01))
