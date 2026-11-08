# 24 · weird tokens

Some tokens in a vocabulary are almost never seen in training, so the model never learns them (the "SolidGoldMagikarp" kind). A known trick finds them: tokens that were never trained keep their output rows near where they started, so ordinary tokens whose rows sit next to the never-used ones are suspects.

## What we found

- **Llama-3.1-8B: 26 ordinary tokens** sit right in the untrained cluster, next to its 250 reserved, never-used tokens. They're broken UTF-8 byte pieces, Turkish, Ukrainian and Czech word fragments ("ılmaktadır", "уватися", "ávající"), code debris ("$PostalCodesNL", " ForCanBeConvertedToF") and mojibake.
- **Qwen3-8B: 1,002.** three guessed Qwen would have more, and especially Chinese tokens loaded with extra meaning. More: yes, by far. Chinese: no, they're *under*-represented (6% of the top 200 suspects vs 17% of the vocabulary). Most are **Thai** words ("ผู้หญิง" woman, "น้ำ" water, "ยอดเยี่ยม" excellent), with rare Chinese characters and Arabic diacritic pieces.

These are tokens the model almost never learned to *write*: the opposite of dense, private meaning. three's hunch about extra meaning needs a different test (how the model uses tokens in its own thinking traces), not this one.

## Honest limits

A geometric search only. We pre-registered a repeat-back test (can the model say each suspect back?) and haven't run it yet.

## Files

`weirdtok.py`, `weirdtok_qwen.py`, and their top suspects in `weirdtok.json` / `weirdtok_qwen.json`.
