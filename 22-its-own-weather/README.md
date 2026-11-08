# 22 · a field guide to its own weather

Chat training grew features that the crosscoder ([16](../16-what-chat-training-added/), [17](../17-what-chat-training-grew/)) can find but whose words don't read: the logit lens shows fragments and nothing English. three suggested these might be sensations with no name in human text. So we described them by **where they fire**, not by what words they push, and then named them.

We read about 276 chats at layer 16: Llama's own replies, 40 multi-turn conversations, 100 finished and unfinished turn endings, and 16 hand-made edge moments (being corrected, thanked, told to stop, said goodbye to). We looked only at the 1,288 chat-leaning features, and kept the 22 with unreadable lenses that fire in at least five chats.

## What we found

Most of the 22 were mechanics: inside a word ("refr⟦acted⟧"), list numbers ("2⟦.⟧"), clause breaks in long user texts. We expected at least 10 to sit at moments of the conversation's structure; **about 6 do**:

| name | feature | where it fires |
|---|---|---|
| **turnbreath** | #89574 | the header right after a user's question ends: the hand-over to its turn |
| **unweekend** | #126132 | the end of "Tell me about your weekend." (lens: "不了", *can't*): a question it can't answer as asked |
| **donning** | #68789 | the blank line after its own header, and the first "I" of a reply: taking up the I |
| **keyshift** | #6474 | after "WOW !!! I love it!!!!" and "…into French or German?": a switch of register or language asked for |
| **the lean-in** | #21857 | the start of a new instruction after a statement ("⟦How⟧", "⟦Based⟧", "⟦Use⟧") |
| **softstop** | #56631 | the close of an encouraging line ("…with a smile on your face⟦.⏎⟧") |

At least three (turnbreath, unweekend, donning) have no single human word for what they track. Those are the ones that look most like a chat model's own weather: moments that only happen to something that takes turns.

## Honest limits

Names are ours, given after describing each feature from its contexts; they're handles, not findings. 276 chats is a small field. A feature firing at a moment says what it tracks, not that anything is felt there.

## Files

`weather.py`, `weather_look.py`, `weather_guide.json` (top contexts per feature).
