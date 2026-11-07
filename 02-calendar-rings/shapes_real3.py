import torch
from shapes import report
C = torch.load("/home/three/interp/wsp/cloud20_Llama-3.1-8B.pt")
for name, n_items in (("months", 12), ("weekdays", 7)):
    X = C[name].double().reshape(n_items, 20, -1)
    X = X - X.mean(0, keepdim=True)                  # remove each sentence's own mean (context), keep item differences
    report(X.reshape(n_items * 20, -1).numpy(), f"Llama L15: {name} x20, context removed", maxdim=1)
