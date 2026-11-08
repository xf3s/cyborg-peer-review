# 12 · does it know it's the one talking?

three's question: *"do early layers know they're an AI? does it realize every turn?"*

We put the same 20 plain first-person sentences ("I think oatmeal is a good breakfast.") in three places: as the assistant's reply, as the user's message, and as bare text. Then we asked, layer by layer, whether a simple reader could tell assistant from user. We also read a 6-turn picnic-planning chat word by word.

## What we found

- **From the very first layer, in both models.** Already after layer 0, a held-out reader separates "assistant said this" from "user said this" 100% of the time, in base and Instruct. That's cheap: one attention step can see the "assistant" label in the template. It's a label lookup, not much of a realization. (We predicted layer 0 would be blind. Wrong.)
- **Chat training makes it much stronger.** Instruct's separation is 2–3× base's through the middle (layer 8: 18.8 vs 7.8; layer 16: 12.6 vs 5.5, in d′).
- **Instruct re-realizes it every turn; base doesn't.** In the picnic chat, Instruct's "it's me" signal flips at every speaker change, on the role label itself at layer 8, by the first word at layer 16: all three user turns read "not me", all three assistant turns "me". Base gets the first exchange right, then from the second user turn on the user's words read as "me" too.

## Honest limits

20 sentences, one chat, one simple direction (assistant minus user). "Knows" means "a linear reader can tell", nothing more.

## Files

`role.py` (both models on NDIF), `role_score.py`.
