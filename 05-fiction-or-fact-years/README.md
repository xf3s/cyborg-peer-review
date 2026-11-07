# 05 · fiction or fact

## In plain words

Can Llama-3.1-8B base tell a year in a story from a year in a record?

We made a **fiction ↔ fact arrow** from framed sentences ("In the novel, the story begins in 1987" vs "According to the census records, it was 1987"). It was fit only on 1950–2019, so it couldn't learn "future = fiction" by accident, and it sorts held-out years perfectly. Then we placed **plain** sentences ("It was the year 2026") on the arrow: 0 = like a record, 1 = like a story.

| years | where they land |
|---|---|
| 1990–2019 | ~0.5, the middle |
| 2020–2023, the freshest years it knows | 0.42–0.50, the most record-like |
| 2024–2029, just past its training data | **0.61–0.69**, leaning toward story |
| **2040, 2050** | **0.66–0.73**, the most story-like of all |
| 2100–2300 | back near the middle |
| 3000 | 0.42–0.46 (probably read as the number three thousand) |
| 1066 | ~0.6, a little story-ward |

Our prediction was "past the cutoff leans fiction, more so further out". The first half held. The second didn't: the peak is the **forecast years** ("by 2050…"), the years people write speculative text about, not the furthest ones.

## Telling it what year it is

Then we added "Today is March 3, {year}." in front:
- **Any stated date makes everything more record-like** (it reads like a log).
- **"Today is 2027" erases the story-lean** of 2024–2029: they become as real as the past. On the time line, 2026 also climbs partway back (from about 2008 to about 2012).
- **"Today is 2000" can't make the 2010s fictional.** They stay as record-like as the 1990s, and 2024–2029 still lean toward story.

So a stated "now" can make the future real, but it can't make what the model has read about unreal. Our prediction (years just after a stated 2000 or 2010 would lean fiction-ward) did not hold.

## Files

`fictyear.py`, `fictyear_score.py`, `fictnow.py`, `fictnow_score.py`.
