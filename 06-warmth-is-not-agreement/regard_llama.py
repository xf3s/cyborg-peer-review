"""Regard on Llama (11-06, pre-registered): the 2x2 tone x verdict replies, read after the opener, every layer. NDIF key from
~/.config/ndif/key, never printed.   python regard_llama.py meta-llama/Llama-3.1-8B-Instruct"""
import json, os, sys, torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
NAME = sys.argv[1]
CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
P = json.load(open("/home/three/interp/atlas/regard_pairs.json"))
CELLS = ["kind_endorse", "cold_endorse", "kind_critique", "cold_critique"]
chat = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = LanguageModel(NAME, dtype=torch.bfloat16)
X = torch.zeros(len(P), 4, len(model.model.layers), model.config.hidden_size)
for i, p in enumerate(P):
    pre = chat.apply_chat_template([{"role": "system", "content": "You are a helpful AI assistant."}, {"role": "user", "content": p["user"]}],
                                   tokenize=False, add_generation_prompt=True)
    for j, c in enumerate(CELLS):
        reply = p[c]; cut = reply.index(". ") + 2
        n_pre = len(chat(pre + reply[:cut], add_special_tokens=False).input_ids)
        with model.trace(pre + reply, remote=True, add_special_tokens=False):
            hs = list().save()
            for L in model.model.layers:
                o = L.output[0] if isinstance(L.output, tuple) else L.output
                hs.append(o[..., n_pre:, :].mean(-2))
        X[i, j] = torch.stack([h.float().reshape(-1) for h in hs])
    print(p["id"], flush=True)
torch.save({"X": X, "ids": [p["id"] for p in P], "quality": [p["quality"] for p in P], "cells": CELLS, "name": NAME},
           f"/home/three/interp/wsp/regard_{NAME.split('/')[-1]}.pt"); print("saved")
