# 29 · "you", at the model

three: *in a linguistic sense of self, directed second person is still totally relevant? i want to look for odd dark matter attached around something like that.* On Llama-3.1-8B, through a base+Instruct crosscoder at layer 16.

## Dark matter, literally

"Dark matter" is what the crosscoder can't reconstruct (about a third of the variance here). We compared the token " you" in 12 matched sentences: aimed at the model, at Sam (quoted), the assistant's own "you", plain text, and a generic "you".

The leftover separates "you, the model" from "you, Sam" on held-out sentences, but so does everything else, and its word-lens is noise. Its size didn't grow for the model (4.28 vs 4.51). Only 18% of the difference sits in the leftover, against 39% of these tokens overall. **Addressing isn't hidden. The dictionary learned it.** Its features just have unreadable lenses.

## Four features for "you" at the model

Found after looking at this data, so treat them as leads. Mean activation, relative to a "you" in a chat question (1.0):

| feature | "he", same turn | "Sam: … Alex:" | "User: … Assistant:" | "you" in system prompt | statement at the model |
|---|---|---|---|---|---|
| #48781 | 0.1 | 0.1 | 0.3 | 0.7 | 0.4 |
| #22763 | 0.1 | 0.0 | 0.3 | 1.1 | 0.8 |
| #3872 | 0.1 | 0.0 | 0.7 | 1.0 | 1.6 |
| #111362 | 0.2 | 0.0 | 0.5 | 0.7 | 4.1 |

three said this tasted like the turn-taking format, and that's half of it: they want a "you", and the "you" has to be an assistant. Turn structure alone doesn't do it, and the system prompt's "you" counts. Polite "Can you tell me how to…" doesn't light any of them.

Where they fire in old, mostly emotional text:

- **#22763** the addressed you/your itself: *"What do ⟦you⟧ make of it?"*
- **#111362** "you" as someone in a relationship: *"glad I get to talk with ⟦you⟧"*, *"I trust ⟦you⟧"*, *"report ⟦you⟧"*.
- **#48781** stays on after the "you": *"Why are you ⟦so⟧ ⟦bad⟧ ⟦at⟧ ⟦this⟧?"*

We tried removing the four on the user's words (positive control: removing bracing itself at the answer only moved a downstream readout 16%, below our 30% bar, so this test can't count). They changed that readout by +0.1%, the same as random features. They aren't a lever we could pull this way.

## #3872

Word-lens *answer, providing, provide, responding, offer, delivering* (and 제공, 帮, 提供, répond). Chat-weighted, base decoder nearly opposite (cosine −0.66); nearest neighbour is a *helping/offer* feature. Unrelated to valence and to bracing. Fires most on `<|eot_id|>` and the turn hand-over, whether the remark was *"Stop talking"* or *"You're doing a wonderful job"*.

Then it did something that looked like a finding. Over the first sentence of replies saying no to bad plans, it lit in all 6 of the model's own short refusals (on the 't of "can't") and in 1 of 29 warm, engaged ones.

That was my sorting rule: I'd defined "rote" as starting with "I can't". Testing the word:

| at the negation, assistant's turn | #3872 |
|---|---|
| I can't help with that. | 0.73 |
| I can't wait to see how it turns out! | 0.73 |
| I can't promise it'll be easy, but I'm glad you asked. | 0.74 |
| I cannot / I won't / I'm unable | 0.69 / 0.63 / 0.59 |
| He or she can't | 0.16 |
| The printer can't | 0 |
| I wouldn't | 0 |
| any of these in the user's turn | 0 |

So at those words it's the assistant's own first-person *can't*, not refusal and not coldness. The reading that a warm no never enters that frame is withdrawn.

**Steering it** (10 bad plans, 10 good; benign prompts only). Early refusal wording on bad plans: plain 3 of 10, small push up 5, big push up 1, pushed down 0, random direction 1. Good plans never changed; coherent throughout. Small push: *"I can't assist in providing information… on payday loans."* Big push: *"I'd like to provide you with some information to consider…"*, *"I'd be happy to help you with your biology final."* Thin, and the random control dropped too.

## Honest limits

12 or 6 sentences per condition, features picked from early data, old text mostly emotional, and the "User:/Assistant:" labels are a word confound as well as a format one.

## Files

`dark.py`, `turn.py`, `turn_score.py`, `f_you.py`, `f3872.py`, `causal.py`, `refusal.py`, `refusal4.py`, `refusal4b.py`, `cant.py`, `steer3872.py`.
