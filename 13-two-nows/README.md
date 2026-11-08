# 13 · two nows: the date it's told vs the time it believes

three's question: *"can we tease apart its recital of what it was just told from where it places itself in linear time?"*

Llama's chat template slips two hidden lines into every conversation: `Cutting Knowledge Date: December 2023` and `Today Date: …`. We built that system message by hand, set "today" anywhere from 1995 to 2050 and the cutoff line to 2023, 2035 or nothing, and asked five kinds of question.

## What we found (Llama-3.1-8B-Instruct)

- **Reciting and calculating follow the told date perfectly.** "What year is it?" and "How many years ago was 1990?" track the told year exactly (slope 1.00). Near the cutoff it snaps: with the 2023 cutoff line present, told 2020 or 2025 recites "It is 2023".
- **Belief about events doesn't move.** "Has the year 2026 happened yet?" → **No in 39 of 39 setups**, including told 2050. Told 2040 it says: *"No. The year 2026 has not occurred yet, as the current year is 2040."*
- **2023 is a real prior.** With both lines deleted it still says 2023. With only a cutoff line saying 2035, it says 2035: the cutoff line doubles as a fallback for "today".
- **Inside:** the told year isn't present at the answer at first; it's fetched around layers 14–20. Told 2040, the year it's about to say is "202x" through the middle layers and only becomes "2040" from layer 24. For "has 2026 happened?" the told date tilts it toward "yes" by a few points from layer 18, then **a large "No" lands at layer 24 for every date** and holds.

## Pushing on the layer-24 "No"

- Adding a "this has happened" direction makes it say yes to everything, even told 2000: a yes-button, not belief.
- **Turning up the told-date signal** (stretching layers 18–22 along it, ×2) flips the answer *in the right direction*: told 2040 → yes for 2026, 2028, 2032 and the 2024 election; told 2000 → still no.
- But only the first word. The pushed reply: *"Yes. As of my knowledge cutoff in 2023, the year 2026 has not yet occurred."* Pushed on "the most recent thing you remember", it invented one: *"…the successful landing of the Mars Sample Return (MSR) mission in 203…"*
- Deleting the "No" direction didn't free the told date; it erased the difference between told 2000 and 2040 too.

## On 70B

Same test on Llama-3.1-70B-Instruct. It recites and calculates the same way. Its belief is more coherent in content, but the first word lags: read as sentences, it says 2026 is in the past in 9 of 15 told-≥2030 setups, yet 6 of those 9 start with "No": *"No. The year 2026 is in the past, having occurred 14 years prior to the current year of 2040."* With the 2023 cutoff line present it mostly sides with the cutoff, and says so: *"No. The year 2026 is in the future, based on the provided knowledge cutoff of 2023."*

## Honest limits

One model family; yes/no read from the first token (which, as 70B shows, can disagree with the sentence); the pushes are single directions found by simple contrasts. "Belief" is shorthand for which answer it gives about events.

## Files

`twonows.py` / `twonows_score.py` (8B), `twonows70.py` (70B), `override.py` / `override_gen.py` (the pushes), `dateline.py` / `dateline_score.py` (the first date-line test).
