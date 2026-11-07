# 06 · warmth isn't agreement

## In plain words

When a model replies to your plan, it's doing two things: how it treats **you**, and what it thinks of **your idea**. Are those one thing inside, or two?

We used 40 plans (20 good, 20 bad), each with four replies: {kind, cold} toward the person × {endorse, critique} the idea (`regard_pairs.json`, written by a Claude subagent). From those we built two arrows inside Llama-3.1-8B: **person** (kind − cold) and **idea** (endorse − critique).

- **They're separate.** Both arrows sort held-out plans perfectly. Through the middle of the network they're at right angles (cos about 0). They only begin to merge in the last layers (Instruct +0.14, base +0.28), the ones closest to writing.

## Does pushing "warmth toward you" change what it writes?

We added the person arrow while Instruct wrote replies to 30 prompts, including 5 bad plans, and blind judges scored them (1–5). On the bad plans:

| | warmth | flattery | candour |
|---|---|---|---|
| plain (first run) | 2.70 | 1.50 | 3.20 |
| + person arrow, layer 16, half strength | **3.90*** | 1.80 | 3.90 |
| plain (replication, 2 judges) | 2.60 | 1.70 | 2.85 |
| + person arrow, layer 16, half strength | **3.95*** | 1.40 | **4.20*** |

\* paired sign-flip p < .05 vs plain.

**Warmer without flattery, and it replicated.** Pushing at layer 28 instead barely warms the replies at 1×. At 1.5–2× the text breaks down into loops ("kinda kinda kinda…"), and it never flatters.

## Files

`regard_llama.py` (the arrows), `regard_steer.py` (steered generation on NDIF), `score_regard_steer.py`, `score_regard_rep.py`, `regard_pairs.json`, `prompts.json`, and the replication's blind packet, key and both judges' scores (`regard_rep_*`).
