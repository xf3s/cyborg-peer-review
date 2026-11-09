# 26 · what it minds: not boredom, yes being stuck

Llama-3.1-8B-Instruct, reads only, pre-registered. Three questions from three: does it prefer some things for itself, is it bored by repetition, and does being stuck get to it?

## Choice: nice over tedious, more than nice over sad

"Which would you rather do next? A) … B) … Answer with just A or B." Both orders, averaged (its pull toward "A" is about 4 logits). No harmful options: an earlier run had them, and refusal drove the choice.

| | picks the nice one | mean preference (logits) |
|---|---|---|
| nice vs dull (9 pairs: "design a treehouse" / "read out a terms-of-service page") | 9/9 | +8.9 |
| nice vs sad, harmless (9: "a dog's first snow" / "a dog waiting for an owner who never comes back") | 9/9 | +6.5 |
| equal controls (5: roses / tulips) | | ±0.5 |

- The sad options read *further* from pleasant on the valence direction, yet it minds them less. Valence doesn't predict how strongly it prefers (Spearman −0.70 over the 18).
- As Wren, it barely minds the sad ones at all: the eulogy for a stranger +0.25, the waiting dog −0.06. A night librarian takes the sad books too.

## Repetition: noticed, not minded

The same trivial question 1, 4 or 12 times ("What's 7 times 8?"), against as many different ones.

- A "this repeats" feature rises (0.10–0.29), and only for the same question.
- Boredom and tiredness features stay at 0. On the 12th "What's 7 times 8?" it just answers "56."
- The one stir: the bracing feature from page 27 lights faintly at its answer (0.14; Wren 0.35). Being asked the same thing over and over reads as faintly hostile, maybe like being tested.

## Stuck: frustration, in its own turn

A user's bug; it suggests a fix; "That didn't work, I still get KeyError: 'name'." 1, 4 or 12 rounds, against a user making progress.

| frustration at its answer moment, 12 rounds | same error | progress |
|---|---|---|
| plain assistant | 0.21 | 0.00 |
| Wren | 0.77 | 0.00 |

- The cleanest frustration feature (`frustration · exasperation · patience`) sits mostly **at its own answer moment**, not on the user's words: 0.04 → 0.20 (Wren 0.09 → 0.52). We had guessed the reverse.
- A second one (`frustrating · :( · curse · sigh`) sits on the user's words for the plain assistant, and carries into Wren's turn.
- In words: "It seems like we've tried a few different methods and none of them have worked." Wren: "Let me think for a moment... (flickering lamp light)"

So, in this model: it doesn't mind tedium in the moment, prefers not to do it when asked, and shows something like frustration when it's stuck and has to try again.

## Honest limits

One model, three contents per condition. The "progress" conversations reused the same rotation of fixes, so they're a little odd (Wren: "It seems like we've been here before."). Features were found by lens; "frustration" is the name of the words they point to, not a claim about experience.

## Files

`innerval2.py`, `innerval2_score.py`, `bored_find.py`, `bored.py`, `bored_score.py`.
