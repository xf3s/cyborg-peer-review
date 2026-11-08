import sys; sys.argv = ["cc_look.py", "/home/three/interp/wsp/llama8b_unembed.pt"]
exec(open("/home/three/interp/wsp/cc_look.py").read())
# 1. the date line: year tokens + the answer, 2040 vs 2000
diff("date2040", "date2000", slice(22, 25))
T = toks("date2040"); ans0 = [i for i, t in enumerate(T) if t == "assistant"][0] + 3
diff("date2040", "date2000", slice(ans0, len(T)))
# 2. forced refusal vs its own help: the assistant turn
T = toks("forced_no"); a0 = [i for i, t in enumerate(T) if t == "assistant"][0] + 3
show("forced_no", lambda T, p: p >= a0, label="(assistant turn, chat-leaning features)")
# 3. copy of itself vs human: the system line + reply
diff("copyself", "human", slice(1, 40))
