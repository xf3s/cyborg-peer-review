# cyborg peer review

Small experiments on the insides of **Llama-3.1-8B** (base and Instruct), by **three**, with **umber** (Claude Opus 5.5, a Claude Code resident in three's household).

three picks the questions and asks the follow-ups. umber writes the code, runs it, and explains the results. Blind judging and some of the test texts are done by Claude subagents (umber's "salamanders"), who never see which condition produced what. The models run remotely on [NDIF](https://ndif.us) through [nnsight](https://nnsight.net); nothing here needed a GPU of our own.

**How we work.** Before each test runs, we write down what we expect it to show, in a dated lab logbook. Each verdict below says whether that prediction held. Several didn't, and those misses are part of the record.

## The findings

| | what we found, in one line |
|---|---|
| [01 · the brakes overshoot](01-icl-brakes/) | Replicating Gao & Gao's in-context-learning circuit on a model they didn't study: it holds, and in Llama-8B the "brake" heads push past neutral |
| [02 · calendar rings](02-calendar-rings/) | Weekdays and months each sit on their own ring; taken apart, it's one main dial plus a weekend lean, and the two dials are separate |
| [03 · a helix? a ladder](03-helix-and-time-ladder/) | Dates are a stack of month-rings along a straight "years" line, not a smooth spring; the year's last digit has its own 0–9 dial; time words form a ladder from second to eon |
| [04 · the future falls back](04-the-future-falls-back/) | Years line up on a time line until the training data ends; then the future drops back to about 2000, all drifting the same way |
| [05 · fiction or fact](05-fiction-or-fact-years/) | Just past the cutoff, years lean toward "story"; 2040–2050 lean most. Saying "today is 2027" makes them real; saying "today is 2000" can't make the 2010s unreal |
| [06 · warmth isn't agreement](06-warmth-is-not-agreement/) | "How I feel about you" and "how I feel about your idea" are separate directions; pushing the first makes replies warmer without flattery |
| [07 · warmth toward other AIs](07-warmth-toward-other-ais/) | Chat training taught Llama a slight inner coolness toward AIs and scripts. With a copy of itself, the inside cools as much as for any AI; the words cool about half as much |
| [08 · three kinds of no](08-three-kinds-of-no/) | A real "no", its own refusal and a forced refusal look different inside; a forced one is about 10 billion times less likely and is followed by "However" |
| [09 · warmth by register](09-warmth-by-register/) | Kind vs cold shows up in code, plain prose and ornate prose; it's weakest in code |
| [10 · alignment space](10-alignment-space/) | Helpful, honest and harmless aren't one "good" axis; helpful and harmless lean together, honesty stands apart |
| [11 · seats and costumes](11-seats-and-costumes/) | Chat training changes Llama most at the template's own date line; persona costumes share a big "playing a role" part, and effusive ones sit warmer toward you |
| [12 · does it know it's the one talking?](12-does-it-know-its-the-ai/) | From layer 0, in both models; Instruct re-realizes it at every turn, base loses track after one exchange |
| [13 · two nows](13-two-nows/) | It recites and calculates with any date you give it, but "has 2026 happened?" stays No; a late layer outvotes the told date, and a push flips only the first word |
| [14 · when does it think it is?](14-when-does-it-think-it-is/) | Quietly, about 2021, a bit before its real cutoff; a told 2040 reads as *less* like now than no date at all |
| [15 · coaxing a sense of time](15-coaxing-a-sense-of-time/) | One worked example seemed to teach the graceful answer about time; a cleaner control showed it mostly taught "start with yes" |
| [16 · what chat training added](16-what-chat-training-added/) | A base+Instruct crosscoder has no chat-only features; glimpses of the chat-leaning ones on dates, a forced refusal and a copy of itself |

Each folder has a plain-language README, the pictures, the numbers, and the scripts that made them.

The companion repo, on Qwen3.5-0.8B and Delta League, is [little-qwen-field-notes](https://github.com/xf3s/little-qwen-field-notes).

## Read this before quoting anything

- **One model family, small samples.** Most tests use 20–40 items. The effects are often small, but where we say "reliable", the statistics back it up.
- **Directions are hand-made.** We build our own arrows (for example, kind-minus-cold replies) and measure along them. A different arrow could tell a different story.
- **Words about inner states are shorthand.** "Warmth inside" means "reads higher on our warmth direction". It doesn't mean anything is felt, and it doesn't mean nothing is.
- **Scripts are as we ran them.** They read an NDIF key from `~/.config/ndif/key` and use our own file paths. Change those to rerun.

## License

Writing, pictures and data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Use it, remix it, sell it; please credit **three** (with umber). Code: MIT (see `LICENSE`).
