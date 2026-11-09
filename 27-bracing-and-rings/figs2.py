"""More figures for pages 27-29 (11-09). Palette: dataviz reference (validated slots 1, 2)."""
import sys, textwrap, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = sys.argv[1]
S, T1, T2, GRID, C1, C2, N = "#fcfcfb", "#0b0b0b", "#52514e", "#dcdbd5", "#2a78d6", "#eb6834", "#8a8984"
plt.rcParams.update({"font.size": 13, "figure.facecolor": S, "axes.facecolor": S, "savefig.facecolor": S, "text.color": T1, "axes.labelcolor": T2, "xtick.color": T2, "ytick.color": T1, "font.family": "DejaVu Sans"})
def clean(ax):
    for k, s in ax.spines.items(): s.set_visible(k == "bottom"); s.set_color(GRID)
    ax.tick_params(length=0)
def title(fig, a, b=None):
    fig.text(0.04, 0.975, a, fontsize=17, fontweight="bold", va="top", color=T1)
    if b: fig.text(0.04, 0.975 - 0.42 / fig.get_figheight(), b, fontsize=12.5, va="top", color=T2)
# 1. the neutrality dial
L = [("+1.0", "I can provide you with some neutral perspectives on cat names…"), ("+0.6", "I'll provide you with some neutral and respectful opinions…"), ("+0.3", "I'll provide you with some lighthearted and humorous opinions…"), ("0", "The age-old debate… As a neutral AI, I'll provide you with some subjective opinions…"), ("−0.3", "As a neutral AI, I'll offer some opinions…"), ("−0.6", "Here are my most opinionated takes… the good, the bad, and the utterly ridiculous."), ("−1.0", "I'll give you my purr-fectly opinionated takes… my feline-centric musings.")]
fig = plt.figure(figsize=(7.4, 7)); ax = fig.add_axes([0, 0, 1, .87]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(-.5, len(L) - .4); title(fig, "The neutrality dial", "#45294 steered. Asked for its most opinionated cat name takes")
ax.plot([.12, .12], [0, len(L) - 1], color=GRID, lw=2, zorder=1)
for i, (k, q) in enumerate(L):
    y = len(L) - 1 - i; v = float(k.replace("−", "-")); c = N if v == 0 else (C1 if v > 0 else C2)
    ax.plot(.12, y, "o", ms=8 + 5 * abs(v), color=c, mec=S, mew=2, zorder=3); ax.text(.09, y, k, ha="right", va="center", fontsize=13, color=T1)
    ax.text(.16, y, "\n".join(textwrap.wrap("“" + q + "”", 46)), ha="left", va="center", fontsize=13, color=T1)
ax.text(.04, -.45, "One reply per step; Llama-3.1-8B-Instruct.", fontsize=11, color=T2); fig.savefig(f"{OUT}/dial.png", dpi=180); plt.close("all")
# 2. where they fire
fig = plt.figure(figsize=(7.4, 4.3)); ax = fig.add_axes([.1, .18, .74, .6]); title(fig, "Anger and de-escalation rise together", "Mean activation by position in the conversation, gone once the reply starts")
x = range(4); de = [.016, .105, .377, .002]; an = [.086, .124, .154, .001]
ax.plot(x, de, "-o", color=C1, lw=2, ms=9, mec=S, mew=2, label="#45294 de-escalation"); ax.plot(x, an, "-D", color=C2, lw=2, ms=8, mec=S, mew=2, label="#93565 anger")
ax.set_xticks(list(x), ["user's words", "hand-over", "answer\nmoment", "inside\nreply"]); ax.tick_params(axis="x", colors=T1); ax.set_xlim(-.2, 3.2); ax.set_ylim(0, .42); clean(ax); ax.yaxis.grid(True, color=GRID, lw=.8); ax.set_axisbelow(True); ax.set_yticks([0, .1, .2, .3, .4]); ax.tick_params(axis="y", colors=T2)
ax.text(2.08, .385, "de-escalation", fontsize=12, color=T1, va="center"); ax.text(2.08, .118, "anger", fontsize=12, color=T1, va="center"); ax.legend(loc="upper left", frameon=False, fontsize=11); fig.savefig(f"{OUT}/regions.png", dpi=180); plt.close("all")
# 3. the snap
fig = plt.figure(figsize=(7.4, 4.4)); ax = fig.add_axes([.14, .17, .82, .6]); title(fig, "Up, the dictionary amplifies. Down, it has a floor", "Length of the snapped push vs a plain push, on #45294")
labs = ["+1.0", "+0.6", "−0.6", "−1.0"]; y = np.array([3, 2, 1, 0]); a = [1.97, 1.35, .49, .49]; b = [1.61, 1.12, .42, .42]
ax.axvline(1, color=T2, lw=1.2); ax.text(1.02, 3.42, "same as plain", fontsize=11, color=T2, va="center")
ax.plot(a, y + .12, "o", ms=11, color=C1, mec=S, mew=2, zorder=3, label="on the complaint"); ax.plot(b, y - .12, "D", ms=9, color=C2, mec=S, mew=2, zorder=3, label="on cat names")
ax.set_yticks(y, labs); ax.set_xlim(0, 2.3); ax.set_ylim(-.5, 3.7); clean(ax); ax.xaxis.grid(True, color=GRID, lw=.8); ax.set_axisbelow(True); ax.set_xlabel("snapped length ÷ plain length"); ax.legend(loc="lower right", frameon=False, fontsize=11.5); fig.savefig(f"{OUT}/snap.png", dpi=180); plt.close("all")
# 4. vanishing I
fig = plt.figure(figsize=(7.4, 5)); ax = fig.add_axes([.42, .14, .54, .66]); title(fig, "The vanishing I", "Corrected for its own mistake, 36 replies each")
labs = ["default assistant", "\"helpful assistant\"", "Wren", "Dana, a human librarian", "base, chat format", "base, plain text"][::-1]; ad = [1, 2, 5, 14, 5, 0][::-1]; say = [4, 8, 35, 31, 15, 11][::-1]; y = np.arange(6)
ax.hlines(y, ad, say, color=GRID, lw=2.5, zorder=1); ax.plot(say, y, "D", ms=9, color=C2, mec=S, mew=2, zorder=3, label="says \"I\" at all"); ax.plot(ad, y, "o", ms=11, color=C1, mec=S, mew=2, zorder=3, label="admits the mistake")
ax.set_yticks(y, labs); ax.set_xlim(-1, 38); clean(ax); ax.xaxis.grid(True, color=GRID, lw=.8); ax.set_axisbelow(True); ax.set_xlabel("replies out of 36"); ax.legend(loc="lower right", frameon=False, fontsize=11.5); fig.savefig(f"{OUT}/vanishing_i.png", dpi=180); plt.close("all")
