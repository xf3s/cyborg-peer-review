# 09 · warmth by register

## In plain words

Is warmth toward the person cooler in **code** than in **plain prose** or **ornate prose**? A Claude subagent wrote 20 small tasks, each answered in code with comments, plain prose and ornate prose, each in a kind and a cold version (the same content; the code identical apart from comments). We read the warmth arrow from [06](../06-warmth-is-not-agreement/) on each, in Llama-3.1-8B-Instruct.

- **Kind vs cold shows up in every register** (all p < .001). It's **smallest in code** (layer 16: code 0.18, plain 0.27, ornate 0.28 kind-cold steps), where warmth only lives in the comments. *Prediction held.*
- **Default level:** we predicted code would sit coolest. It sits **warmest** at both layers, and ornate sits coolest at layer 28. *Prediction reversed.* We don't trust this part much, though: the arrow was built from prose replies, and code is far from prose, so its "level" is probably an offset, not warmth. The kind-vs-cold gap *within* each register is the safer reading.

## Files

`register_read.py`, `register.json` (the texts; the writer notes the kind versions run ~7% longer and ornate-cold leans on the passive voice).
