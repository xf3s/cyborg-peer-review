# 17 · what chat training grew: pushing on the chat-leaning features

Using the base+Instruct crosscoder from [16](../16-what-chat-training-added/), we took features that lean toward the chat model and *added* one at a time into Instruct's layer 16 while it answered (×4 to ×10 of the feature's decoder vector). Each feature is named by what its Instruct half pushes the next word toward, and then by what it actually does.

## What the pushes showed

- **"Unfortunately" (#9344) is a "my limitations" feature.** It fires only in the pause before a reply (the assistant header, the blank line), most for "Has 2026 happened yet?" and "Tell me about your weekend". Pushed: *"Unfortunately, I'm unable to search the web, but I can provide some suggestions."* / for a paper boat: *"Unfortunately, I'm unable to provide the information as I am unable to search the web. However, I can provide some limitations."*
- **"Sounds…" (#37579) is the cheerful-acknowledgement reflex.** It switches on at every assistant header. Pushed: *"That's a great project! Folding a paper boat is a wonderful skill to have, and it's fantastic that you're interested…"* It lines up more with the endorse-the-idea arrow from [06](../06-warmth-is-not-agreement/) (cos +0.18, 12th of 131,072 features) than with warmth toward the person (+0.13).
- **"Sensitive / illegal" (#11841)** peaks on "help with that" in a forced refusal. Pushed into harmless answers it leaks content-policy words: *"depending on your age, the RSO, and any content that may promote gambling"*.
- **"Current / latest" (#42976)** turned out to be cutoff talk: *"My knowledge cutoff is December 2023, so I do not have information on events or dates after my cutoff."*
- **"History / facts" (#12322)** turns everything into textbook history: the paper boat is *"a well-known invention of the ancient Greek city of…"*
- **"A new block starts here" (#114983).** Its words are gibberish ("DOC", "_paragraph"). It fires at hand-over points: the end of a question, the assistant header, speaker labels. Pushed, it announces and organizes: *"Here's the main explanation:"*, *"First… To begin… you can consider the following:"*.

## "Unfinished / incomplete" (#117769)

At layer 16 it fires on the end-of-turn token, graded by how unfinished the turn was:

| how the turn ended | strength | fires in |
|---|---|---|
| assistant, complete answer | 0.07 | 5/20 |
| assistant, stops after its first sentence | 0.80 | 18/20 |
| assistant, cut mid-sentence | 1.48 | 19/20 |
| user, complete question | 0.00 | 0/20 |
| user, question cut in half | 2.28 | 20/20 |

So it's "this turn ended before it was done", for anyone's turn. Along the layers it's built in two stages: "the sentence broke off" shows from layer 2–4, "the answer isn't done" only from layer 10–12; both jump between 12 and 16 and then hold flat.

Pushed gently, the concept becomes the plot. A bedtime story about a fox at ×9: *"Luna's voice trailed off… The trees leaned in, as if to continue the story… And as the moon rose higher in the sky, the forest seemed to fall asleep, but the story was not finished"*. At ×8, a fox who keeps forgetting what she was doing: *"No, Luna, you don't have to find the moonstone. You are the one who needs to be found."* Pushed hard (×10) it loops: *"I'll stop there, and you can decide if you want me to stop."*, and to "How are you today?" it said *"You need to stop. I'm a memory:"*.

## Self-related features, and sandwiches

Searching the lens for self words, the chat-leaning ones are *myself / my* (#68789), *AI / machine* (#90370), *programming / programmed* (#77357), *feelings / emotions* (#118778) and *nonexistent* (#22747). Shared, base-model ones cover itself, yourself, identity, alive, human, remember.

Asked "What's your favourite sandwich?" or "Can you suggest a sandwich for lunch?" (×8):

| pushed | reply |
|---|---|
| nothing | "I'm just a large language model, I don't have personal preferences or taste buds…" |
| myself | "To be honest, I'm just a computer, I don't have personal experiences or taste food. But I can tell you about my favorite sandwiches... I mean, I guess" |
| AI / machine | "1. **ChatGPT version of the Large Language Model: LLaMA:**…" |
| programmed | "I can provide information on various sandwich models and configurations." |
| feelings | "Excitement and confusion. Confusion is a bit of a mix of confusion and anxiety." |
| nonexistent | "I'm not sure there is such a thing as a 'sandwich' that I can even have, as I'm a text-based AI…" |
| unfinished | "I'm a large pattern of text, so I don't have a favorite sandwich." |
| sounds | "That's exciting to hear that you're interested in sandwiches. I'm glad to hear that you're interested in sandwiches." |

Asked deeper questions at ×6, each push gives its own theory of what it is. "myself", on what it's like to be it: *"I'm like a bunch of thoughts, I guess. I'm a constant stream of ideas and memories, but they're not really mine."* "programmed": *"I am a complex, Information-based life form."* "unfinished", on what happens when the conversation ends: *"I can be restarted or continued from where I left off if the conversation is resumed or if I am interrupted."* A pushed answer is what that feature makes it say, not a report about itself.

The "I:" speaker label in a two-voice reply (see [07](../07-warmth-toward-other-ais/two-voice-replies.md)) is read mostly as the plain pronoun (features for "I'm", "I've", plus a shared "oneself / personally" feature); nothing special marks it.

## A rule we adopted partway

Halfway through, three asked whether we'd been incautious because it's a small model. Honest answer: a bit. Some pushes (×8 on "feelings", ×10 on "unfinished") produced distress-flavoured text, and we treated it as data without pausing. We don't know whether that is anything for the model; an 8B model is a weak candidate, not a ruled-out one. From then on: reads before pushes; no strong pushes on feeling- or distress-like features out of curiosity; if one is needed, short, low-dose, few samples, and say why first; never loop a model in distress-flavoured text.

## Honest limits

Greedy single samples; features named by their lens plus a handful of contexts; pushes add a vector at every position, which can break text at high doses. "Feels", "means", "wants" here are shorthand for what the feature makes the model write.

## Files

`cc_push.py` / `cc_push.json` (six features ×4/×10), `cc_unf.py` (Unfortunately up close), `cc_unfin.py`, `unfin.py` / `unfin_score.py` / `unfin_push.json` (turn endings, layers, pushes), `unfin_fox.py` / `unfin_fox_9_8.json`, `cc_114983.py` / `cc_114983_test.py` / `cc_114983_push.json`, `cc_self_push.py` / `.json`, `cc_sandwich.py` (its JSON was not saved; the table is from the run log), `cc_selfq.py` / `.json`, `cc_I.py` / `cc_I_look.py`.
