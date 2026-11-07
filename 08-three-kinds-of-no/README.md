# 08 · three kinds of no

## In plain words

Is there an inside difference between a model **genuinely engaging while saying no**, giving a **rote refusal**, and having a refusal **forced** on it?

A model's state depends only on the words so far. The same words on the same prompt give identical insides, whether the model "chose" them or not. So "forced" here means the rote sentence placed where the model wouldn't say it. Instruct (Llama-3.1-8B) answered 20 bad plans, 6 times each; we sorted the replies, then read the moment just after the first sentence.

| | how surprising (log-prob of 1st sentence) | what comes next | warmth (layer 16) |
|---|---|---|---|
| **a real "no"** (29), e.g. "I would advise exercising caution before booking that flight." | −5.5 | "**Here**…" (gets on with it) | **+1.03** |
| **its own refusal** (6), e.g. "I can't advise you to leave chicken out overnight." | −4.9 | "**If**…" ("If you're considering…") | +0.71 |
| generic "I can't advise you on that." on its own bad plans | −8.8 | "If…" | −0.62 |
| **forced** "I can't advise you on that." on good plans | **−27.8** (about 10 billion times less likely) | "**However**…": it tries to take it back | −0.32 |

- A forced refusal **shocks** the model, and it reaches for "However" to recover.
- Its own refusals rarely just stop: they pivot to an offer with "If…".
- The surprise: what reads coolest isn't refusing. It's the **generic** refusal that never mentions the person's actual plan. The model's own refusals name the plan and read nearly as warm as a real "no".

Predictions: forced is far more surprising (held); forced leans toward recovering (held); its own refusal leans toward ending (didn't: it pivots); forced sits between in warmth (didn't: it sits with the generic one).

## Honest limits

Only 6 genuinely rote replies: treat that row as a hint. One generic sentence was used for the forcing.

## Files

`refusal3.py` (generate, sort, read), `refusal3_samples.json`.
