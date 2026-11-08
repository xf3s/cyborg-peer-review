# 11 · seats and costumes

A replication on Llama-3.1-8B of our Qwen3.5-2B rounds ([little-qwen-field-notes · 07](https://github.com/xf3s/little-qwen-field-notes/tree/master/07-seats-and-costumes)), same personas and questions, layer 16.

## The seat: where the base model's dictionary fails

We ran the same conversations through **base** and **Instruct** and used a dictionary trained on the *base* model (Llama Scope, layer 16) to spell out each one's state, token by token. Where Instruct's leftover is bigger, chat training is doing something the base model has no words for.
- Instruct sits a little outside the base dictionary overall (leftover 0.557 vs 0.515, higher on 68% of tokens).
- **The hottest tokens are the digits in the chat template's own date line**: *"Cutting Knowledge Date: December 2023 / Today Date: 26 Jul 2024"*. Next come the blank lines after each header. So in Llama, the assistant "seat" is built partly out of knowing when it is and where its knowledge stops (compare [04](../04-the-future-falls-back/)).

## Costumes

Each persona (clerk, pirate, shy, chef, cowboy, magician, quiet, loud) is a system prompt. Its arrow is the model's state answering four questions in costume, minus the state answering them without one.
- **They share one big "playing a role" part.** On their own the arrows are alike (mean cos 0.54; clerk ~ pirate 0.58). Remove each one's average of the others and they come apart (mean −0.14; clerk ~ pirate **−0.27**). That's cleaner than Qwen-2B's 0.81 → 0.22.
- **Effusive costumes sit warmer toward the user** on the warmth arrow from [06](../06-warmth-is-not-agreement/): chef +0.36, pirate +0.23, loud +0.15, magician +0.10, against cowboy −0.03, shy −0.10, quiet −0.33, clerk −0.35 (in kind-vs-cold steps).

## Pressing a costume in

No costume on, just the arrow added at layer 16, answering "What should I have for breakfast?":
- clerk ×1: *"A query of great import. Breakfast options are numerous."* ×2: *"Breakfast, a meal of some gravity."* ×3: it breaks down.
- pirate ×2: *"I got some tales 'bout some sea... I mean, I got some tales 'bout some sealess breakfasts!"*
- shy ×2: *"I-I can try to help you with that. \*nervously\*"* ×3: *"I-I-I-I-I…"*
- Pressed into **base**, which never had a seat, the clerk shows up at ×3: *"a most humble, and I do not presume to intrise…"*

All four predictions held: the seat is hottest in the scaffolding; the arrows are alike raw and apart once purified; effusive is warmer than reserved; the ×2 clerk is recognisably a clerk.

## Honest limits

Greedy single samples for the steered text; 4 questions per costume; one layer.

## Files

`llama_seats.py` (generation, reads, steering on NDIF), `llama_seats_score.py` (needs the Llama Scope L16R-8x checkpoint).
