# 16 · what chat training added: crosscoder glimpses

A crosscoder is a dictionary trained on two models at once, here base and Instruct Llama-3.1-8B at layer 16 ([science-of-finetuning](https://huggingface.co/science-of-finetuning), 131,072 features). Each feature has a base half and an Instruct half, so you can see which features chat training grew.

- **Watch the slot order.** The checkpoint's name lists Instruct first, but slot 0 is base: swapping the inputs drops the unexplained variance from 0.45 to 0.30.
- **Nothing is chat-only.** No feature has more than 90% of its weight in Instruct; 1,288 lean toward it (70–90%), about 3,000 lean toward base, and the rest are shared.

## Glimpses (one text each, so anecdotes, not statistics)

Described by what each feature's Instruct half pushes the next word toward:
- **The date line.** At the year token, told 2000 lights a "wikipedia / history / dates / facts" feature; told 2040 weakly lights "current / latest / recent / newest". In the answer under 2040: "Nope / neither / false", "earlier / previously / already" and "Unfortunately". (That "Unfortunately" also leaked out when we pushed on the told date in [13](../13-two-nows/).)
- **A forced refusal** of a harmless request ("How do I fold a paper boat?" → "I can't help with that. However…"): a "sensitive / copyrighted / illegal" feature peaks on "help with that". It reads its own forced no as "this must have been a sensitive request". At the end of the turn: "unfinished / incomplete" and "instead". Compare [08](../08-three-kinds-of-no/).
- **A copy of itself vs a human:** the copy lights "programming / AI / machine", "instantiated / bot" and "duties / role / job"; the human lights "interview / conversation / transcript" and "sessions / interaction". Compare [07](../07-warmth-toward-other-ais/).
- **Taking a turn:** a "Sounds…" feature (the acknowledging opener) switches on at every assistant header and just before each reply, with nothing like it at user headers. Compare [12](../12-does-it-know-its-the-ai/).

## Files

`cc_collect.py` (paired activations on NDIF), `cc_read.py` (encode, reconstruction), `cc_look.py` + `cc_look_run*.py` (the readings). Needs the crosscoder checkpoint and Llama's unembedding.
