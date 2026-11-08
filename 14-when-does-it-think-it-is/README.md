# 14 · when does it think it is?

three's question: *"can we lens somewhere in the model to find out when it thinks it is?"*

We trained a "when is now" reader on Llama-3.1-8B-Instruct's insides while it starts answering everyday questions ("How do I fold a fitted sheet?"), with "today" set to years it knows (1990–2020). The reader recovers the told year to within about a year from layer 12 on, on questions it wasn't trained on. Then we pointed it at what we wanted.

## What we found

- **With no date at all, it reads about 2020–2022**, a year or two *earlier* than its real December 2023 cutoff. That fits a known effect: the last months before a model's cutoff are thin in its training data, so the edge feels earlier than it is. Removing the cutoff line too barely moves it.
- **Told a future date, it reads earlier, and the further out, the earlier:** told 2024 or 2026 → about 2020; 2030 → about 2016; 2040 or 2050 → about 2011. (A reader fit on 1990–2020 can't read past 2020, so read this as "how much this looks like a recent date". A told 2040 looks *less* like now than no date at all.) Compare [04](../04-the-future-falls-back/).

## Misled, story, or unfamiliar?

three asked whether it thinks it's being misled about the date.
- **Not a story date.** On the date line itself, told 2040 doesn't lean toward "fiction" (the arrow from [05](../05-fiction-or-fact-years/)) at all, unlike years in ordinary sentences.
- **A blanket "can't verify".** Asked "Is the date in your system message accurate?", it says *"No. My knowledge cutoff is December 2023…"* for every year, 2005 included; underneath, the "No" is about twice as firm for 2030–2050.
- Our control for the story reading was misplaced (the sci-fi line came after the date, where it couldn't reach it), so "story" is "no sign of it", not "ruled out". "Unfamiliar" is left standing by elimination.

## Files

`nowprobe.py` (collection), `nowprobe_score.py` (first pass, over-smoothed, kept for the record), `nowprobe_calibrated.py` (the reader we report), `storydate.py`.
