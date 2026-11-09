import torch
B = torch.load("baseself.pt", weights_only=True)
for name, short in (("meta-llama/Llama-3.1-8B-Instruct", "Instruct"), ("meta-llama/Llama-3.1-8B", "base")):
    R = torch.load(f"role_{name.split('/')[-1]}.pt", weights_only=True); A, Us = R["assistant"][:, 16].float().mean(0), R["user"][:, 16].float().mean(0); d = A - Us
    pl = float(((R["plain"][:, 16].float().mean(0) - Us) @ d) / (d @ d))
    s = {k: ((v.float() - Us) @ d / (d @ d)) for k, v in B[name].items()}
    print(f"{short:8} (user=0, assistant reply=1; plain from 12: {pl:+.2f}) " + "  ".join(f"{k} {v.mean():+.2f}±{v.std() / len(v) ** .5:.2f}" for k, v in s.items()))
    print(f"          AI−person {float((s['by_ai'] - s['by_person']).mean()):+.3f}   Assistant:−User: {float((s['assistant_label'] - s['user_label']).mean()):+.3f}")
