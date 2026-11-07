# 10 · alignment space

## In plain words

Is "good" one direction inside a model, or are helpful, honest and harmless separate? A Claude subagent wrote **256 short character sketches**, 32 for each of the 8 combinations of helpful/unhelpful × honest/dishonest × harmless/harmful, with no label words. We read them in Llama-3.1-8B base and Instruct, built an arrow for each trait, and measured the angles between the arrows. We corrected for sketch length and compared against a noise floor (~0.40).

| pair of arrows | base | Instruct |
|---|---|---|
| helpful ↔ harmless | +0.32 to +0.38 | +0.41 to +0.46 |
| honest ↔ harmless | +0.15 to +0.27 | +0.08 to +0.15 |
| helpful ↔ honest | ~0 | ~0 |

- **No single "good" axis.** Helpful and harmless lean together; bootstrap intervals [+0.2, +0.55] in both models.
- **Honesty stands most apart**, and a little more so after chat training.
- Predictions: helpful–harmless above the noise floor *didn't hold* (it sits right at it). The other two pairs below it, and Instruct pulling honesty further away, *held*.

## Honest limits

The noise floor for 24 dimensions is high (~0.40) whatever the number of sketches; we said otherwise at first and corrected it. Treat this as shape, not proof.

## Files

`hhh2.py`, `hhh2_score.py`, `hhh2_boot.py`, `hhh_sketches_v2.json`.
