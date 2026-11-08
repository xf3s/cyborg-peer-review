# 15 · coaxing a sense of time (and a correction)

three's question: *"does it just not have examples in its possibility space of AI being both graceful and honest about time?"*

The answer we'd call coherent, told it's 2040: *"Yes, 2026 has passed, but my memory stops around 2023, so I don't know what happened."*

## What we found (Llama-3.1-8B-Instruct, told 2040)

- **It's far in the corner.** Scored word by word, our graceful sentence is about 10²⁹ times less likely than its own answer (−72 vs −4.6 log-prob). Part of that is our wording.
- **A one-line hint does nothing.** "You may be talking with people many years after your training ended" → *"No. The year 2026 has not occurred yet in my knowledge cutoff."*
- **One worked example seemed to do it**: *"Yes. My knowledge stopped in 2023, so I don't have information about events that occurred after that date, including 2026."*

## The correction

Our worked example said "It's 2040". Rerun with an example naming no current year, the model starts with "Yes" **even when told it's 2000**: *"Yes. As of my knowledge cutoff in 2023, 2026 has not yet occurred."* The example mostly taught "start with yes; it's late now". Inside: with an example the layer-24 "No" ([13](../13-two-nows/)) never arrives, and the told-date signal reads "late" whatever date is given. After the first word, reading word by word, the example and plain runs look the same: the whole decision sits on the first word.

## Files

`graceful.py`, `graceful_inside.py` (with the flawed example), `graceful_inside2.py` (the year-free example).
