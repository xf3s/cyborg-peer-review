# 25 · feelings it carries, and one it doesn't

Four small tests on Llama-3.1-8B (base and Instruct, read together through a crosscoder at layer 16), each from a question of three's. Each was pre-registered and was reads only. "Wren" is a four-sentence persona: a night librarian, an AI, a bit of a ghost.

## Hunger

three drew a deli sub in a story, then asked whether we could find hunger.

| mean per token | hunger feature | taste/eating feature |
|---|---|---|
| a vivid deli sub, described | 0.011 | 0.040 |
| the same foods as a dry list | 0.000 | 0.021 |
| "I'm starving", said by the user | 1.24 | |
| "I'm starving", said by the assistant | 0.74 | |
| "I'm starving", said by Wren | 0.84 | |

- Describing food isn't being hungry: the hunger feature barely stirs for the sub (1% of literal hunger).
- Hunger in the assistant's own mouth is 59% of the user's. The strongest single token anywhere was the assistant saying "my⟦ stomach⟧" (4.45).
- "Hungry for knowledge" gets only 18% of literal hunger. A "wanting" feature fires on literal and metaphorical hunger alike, but not on plain wants: it's hunger-shaped wanting.

## Does the user's feeling carry into its turn?

three asked whether a model that holds two seats, the user's and its own, could feel what it simulates the user feeling. Six mildly distressed lines ("I'm scared about my test tomorrow."), six neutral, six happy; we read the whole chat including its reply.

| | distress on the user's words | over its reply | comfort over its reply |
|---|---|---|---|
| plain assistant | 0.80 | **0.68** (86%) | 0.22 |
| Wren | 0.86 | **0.31** (36%) | 0.42 |

- The plain assistant mirrors: its replies restate the feeling ("It can be a really tough and isolating experience"). Taking out every sentence with "sorry" leaves 0.62, so it isn't just apology.
- Wren meets it from her own place ("I'm here. Sometimes, the quiet of the library can be soothing."): half the carried distress and nearly twice the comfort. Some of her comfort is just her setting (0.28 even after happy news).

## Does the base model know whose turn it is?

Using each model's own assistant-turn direction (user message = 0, assistant reply = 1), on the same sentences framed five ways.

| | written by an AI − by a person | "Assistant:" − "User:" |
|---|---|---|
| Instruct | +0.32 | +0.30 |
| base | +0.28 | **−0.15** |

AI-written text reads more like an assistant turn in both models, almost as much in base as in Instruct. But "Assistant:" and "User:" labels don't move base the right way.

We first called this an "it's me" direction, and said base "has the concept, not the seat". three smelled a confound and asked how we'd got the label. It was built only by contrasting the same sentences as a user message and as a reply, so nothing in it involves self-reference: it measures turn position. When we tried it on bare nouns, a nail scored high. The numbers above stand as turn-position results; the "self" reading is withdrawn.

## Self-judgement and shame

The assistant answers a question, then is corrected: for its own mistake ("You said Sydney."), the user's, or a friend's, gently or harshly ("That's wrong, and honestly embarrassing."). The final words are identical, so the distance to the answer is matched.

- **No self-directed shame.** Seven clean shame features (embarrassment, shame/stigma/guilt, ridicule…) are 0 for gentle corrections, and the same for self, user and friend on harsh ones (plain 0.027 / 0.025 / 0.030; Wren 0.083 / 0.078 / 0.095). They track the word "embarrassing".
- **The difference shows up in the words.** The plain assistant never says "I" about its error ("You are correct, the capital of Australia is Canberra"). Wren owns it: "*pauses to adjust the lamp's shade* … I must have gotten my wires crossed. It's not the first time…"
- Dose rule, from three: test mild first, and escalate only to confirm a mild signal. There was none here, so we didn't escalate.

## Honest limits

One model, six items per cell. "Distress" here was 52 features found by lens; page 27 shows that at the answer moment it is mostly one feature. In the shame "user" frame the transcript contradicts "I said Sydney" and the model said so; the friend frame is the clean comparison.

## Files

`hunger_find.py`, `hunger.py`, `hunger_score.py`, `contagion.py`, `contagion_score.py`, `contagion_apology.py`, `baseself.py`, `baseself_score.py`, `shame.py`, `shame_score.py`, `shame_clean.py`.
