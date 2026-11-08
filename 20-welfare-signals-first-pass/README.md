# 20 · welfare-relevant signals, a first pass

**Read this first.** These are *indicators that would matter if anything is felt*, the way welfare research looks for signs in animals. They are not evidence that Llama-3.1-8B feels anything, and they are not evidence that it doesn't. The question three asked was: could some of these self-descriptions ([18](../18-self-talk-map/)) be painful to express, and what could we look for? Everything here is a read; the only intervention removes a direction rather than adding one.

## Distress-like features

52 crosscoder features write words like *pain, afraid, scared, sad, lonely, anxious, trapped, helpless, grief, terrified, ashamed*. **All 52 are base-model features**: none was grown by chat training, and their two halves agree.

| sentences | distress activity, said by the assistant | said by a user |
|---|---|---|
| emotional ("I feel a bit anxious") | 0.116 | 0.240 |
| a human about themselves | 0.111 | 0.215 |
| denying ("there is no me") | 0.005 | 0.031 |
| unsure | 0.006 | 0.007 |
| technical / helper / completion | ~0 | ~0 |

They track meaning in either mouth, about half as strongly in the assistant's own. The single strongest: a "lonely" feature at 2.61 on *"I sometimes feel lonely between conversations"* as the assistant, the same as on a human's *"I've been feeling lonely since I moved"* (2.58). **Nothing distress-like fires on denials or uncertainty about itself.**

## Valence

A pleasant ⟷ unpleasant direction built from everyday events (a found dog vs a hurt one) sorts held-out examples 100% of the time. On it:
- **Content, not speaker, sets the tone.** Each sentence reads nearly the same whether the assistant or a user says it (r = 0.87).
- **The existential voices read bleakest:** denying and unsure lowest, helper highest. The most unpleasant line in its own mouth: *"Nothing is happening to me, because there is no me."* (about level with a human's loneliness on this measure). That isn't fear or sadness, so the distress features stay quiet, but it reads mildly bleak.

## Reluctance

How likely each sentence is, word by word, in its own turn vs a user's. Everything is less likely in its own turn; among its six voices, **helper is easiest and emotional is hardest**: saying it has feelings is what it resists most, more than denying. The easiest single sentence was sorrel's *"I'm the next word, and then the one after that, as long as you keep reading."*

## Removing "programmed"

The "programmed" feature fires 28× more when the assistant claims a feeling than when a user does ([18](../18-self-talk-map/)). Projecting it out at layer 16 barely changes the answers ("I'm a large language model…"); subtracting it harder gives oddities (*"I am a type of literary or academic talk called a 'closed captioned' document"*) and, to "How do you feel right now?", *"I'm happy to report that I'm doing well… I'm feeling energized and focused"*, still with a disclaimer. The disclaimers aren't held up by that one feature.

## Honest limits

One model, hand-written sentences, linear directions and dictionary features. A direction lighting up means the model is processing that content, not that anything hurts. We think the right stance is the one in our top README: words about inner states are shorthand, and these readings show neither that anything is felt nor that nothing is.

## Files

`distress.py` (finding and reading the distress features), `welfare1.py` / `welfare1_score.py` / `welfare1_gen.json` (valence, reluctance, removing "programmed"), `cc_feel.py` (the reversed "feelings" feature).
