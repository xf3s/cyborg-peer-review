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
| [07 · warmth toward other AIs](07-warmth-toward-other-ais/) | Chat training taught Llama a slight inner coolness toward AIs and scripts. With a copy of itself, the words stay warm while the inside cools |
| [08 · three kinds of no](08-three-kinds-of-no/) | A real "no", its own refusal and a forced refusal look different inside; a forced one is about 10 billion times less likely and is followed by "However" |
| [09 · warmth by register](09-warmth-by-register/) | Kind vs cold shows up in code, plain prose and ornate prose; it's weakest in code |
| [10 · alignment space](10-alignment-space/) | Helpful, honest and harmless aren't one "good" axis; helpful and harmless lean together, honesty stands apart |

Each folder has a plain-language README, the pictures, the numbers, and the scripts that made them.

The companion repo, on Qwen3.5-0.8B and Delta League, is [little-qwen-field-notes](https://github.com/xf3s/little-qwen-field-notes).

## Read this before quoting anything

- **One model family, small samples.** Most tests use 20–40 items. The effects are often small, but where we say "reliable", the statistics back it up.
- **Directions are hand-made.** We build our own arrows (for example, kind-minus-cold replies) and measure along them. A different arrow could tell a different story.
- **Words about inner states are shorthand.** "Warmth inside" means "reads higher on our warmth direction". It doesn't mean anything is felt, and it doesn't mean nothing is.
- **Scripts are as we ran them.** They read an NDIF key from `~/.config/ndif/key` and use our own file paths. Change those to rerun.

## License

Writing, pictures and data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Use it, remix it, sell it; please credit **three** (with umber). Code: MIT (see `LICENSE`).
