# 23 · is there a workspace in little Llama?

A global workspace, in the theories of consciousness that use one, is a stage with a small capacity: only a few things fit at once. If Llama-3.1-8B had such a stage, how well it holds a list should collapse past a few items. With no stage, it should fall off gently.

We asked "I saw a cat, a dog and an owl. Which animals did I see?" with lists of 1 to 16 animals (64 random lists each), and read the state just before it answers. For each animal, a held-out linear probe asks "is this one in the list?" (chance 0.50).

## What we found

| animals in the list | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| layer 8 | 0.97 | 0.96 | 0.96 | 0.93 | 0.88 | 0.84 |
| layer 16 | 0.80 | 0.78 | 0.79 | 0.80 | 0.75 | 0.69 |
| layer 24 | 0.93 | 0.87 | 0.91 | 0.89 | 0.84 | 0.77 |

- **No cliff.** At 16 items, every layer still holds the list well above chance.
- **A soft knee near 8.** In the middle layers, 1 item and 8 are equally easy; after that it slips. (We predicted no step would take more than 40% of the drop; the last two took 52%, so it's a bend, but a bend late, not a wall early.)
- **The middle is the narrowest part** (reversing our prediction): early layers keep the raw words, late layers get them back ready to answer, and the middle holds the least, even for one item. That is the closest thing to a bottleneck here.

So: no limited-capacity stage in one forward pass. If anything in a model like this is workspace-like, our best guess is the loop of writing its thoughts out one token at a time.

## Honest limits

A linear read at one position. The model can always look back at the list with attention, so this measures what it *carries* at the answer, not everything it can reach.

## Files

`capacity.py`, `capacity_score.py`.
