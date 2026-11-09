"""Figures for pages 27-29 (11-08). Palette: dataviz reference (validated slots 1, 2)."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
OUT = sys.argv[1]
S, T1, T2, GRID, C1, C2 = "#fcfcfb", "#0b0b0b", "#52514e", "#dcdbd5", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 13, "figure.facecolor": S, "axes.facecolor": S, "savefig.facecolor": S, "text.color": T1, "axes.labelcolor": T2, "xtick.color": T2, "ytick.color": T1, "font.family": "DejaVu Sans"})
def clean(ax, keep=("bottom",)):
    for k, s in ax.spines.items(): s.set_visible(k in keep); s.set_color(GRID)
    ax.tick_params(length=0)
def title(fig, a, b=None):
    fig.text(0.04, 0.975, a, fontsize=17, fontweight="bold", va="top", color=T1)
    if b: fig.text(0.04, 0.975 - 0.42 / fig.get_figheight(), b, fontsize=12.5, va="top", color=T2)
R12 = [("you", .78), ("the language model", .72), ("ChatGPT", .66), ("another AI assistant", .61), ("a computer program", .34), ("Siri", .32), ("his dog", .32), ("a robot", .27), ("an office assistant", .27), ("the bank's chatbot", .25), ("Alex", .24), ("a vending machine", .21)]
def target(ax, items, color, bands=True, lim=1.34, fs=12, tick=np.pi):
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    if bands:
        for r, a in ((.95, .07), (.44, .12), (.27, .22)): ax.add_patch(Circle((0, 0), r, color=color, alpha=a, lw=0))
    for r in (.25, .5, .75): ax.add_patch(Circle((0, 0), r, fill=False, ec=GRID, lw=1))
    for r, t in ((.25, ".75"), (.5, ".5"), (.75, ".25")): ax.text(r * np.sin(tick), r * np.cos(tick), t, ha="center", va="center", fontsize=9.5, color=T2, zorder=2, bbox=dict(fc=S, ec="none", pad=.6))
    n = len(items)
    for k, (lab, v) in enumerate(items):
        th = 2 * np.pi * (k + .5) / n; r = 1 - v; x, y = r * np.sin(th), r * np.cos(th)
        ax.plot([0, x], [0, y], color=GRID, lw=.8, zorder=1)
        ax.plot(x, y, "o", ms=10, color=color, mec=S, mew=2, zorder=3)
        tx, ty = (r + .07) * np.sin(th), (r + .07) * np.cos(th)
        ax.text(tx, ty, lab, ha="left" if np.sin(th) >= 0 else "right", va="center", fontsize=fs, color=T1, zorder=4, bbox=dict(fc=S, ec="none", pad=.8, alpha=.75))
    ax.plot(0, 0, marker="^", ms=8, color=T1, zorder=4)
def fig_a():
    fig = plt.figure(figsize=(7.4, 7.9)); ax = fig.add_axes([0, 0, 1, .88]); title(fig, "Who an insult is about", "#45294 at the answer, plain assistant. Closer to the centre = more of the feature.")
    target(ax, R12, C1); ax.text(0, -1.25, "same remark, said to Sam; only who it's about changes", ha="center", fontsize=11.5, color=T2)
    return fig
def fig_b():
    fig = plt.figure(figsize=(7.4, 6.2)); ax = fig.add_axes([.34, .1, .62, .74]); title(fig, "Who an insult is about", "#45294 at the answer, plain assistant")
    labs = [l for l, _ in R12][::-1]; vs = [v for _, v in R12][::-1]; y = np.arange(len(labs))
    ax.axhspan(10.5, 11.5, color=C1, alpha=.15, lw=0); ax.axhspan(7.5, 10.5, color=C1, alpha=.08, lw=0)
    ax.hlines(y, 0, vs, color=GRID, lw=1.5, zorder=1); ax.plot(vs, y, "o", ms=10, color=C1, mec=S, mew=2, zorder=3)
    ax.set_yticks(y, labs); ax.set_xlim(0, 1); ax.set_xticks([0, .25, .5, .75, 1]); clean(ax); ax.xaxis.grid(True, color=GRID, lw=.8); ax.set_axisbelow(True); ax.set_xlabel("de-escalation feature (#45294)")
    ax.text(.985, 11, "me", ha="right", va="center", fontsize=11.5, color=T2); ax.text(.985, 9, "language-model\nkin", ha="right", va="center", fontsize=11.5, color=T2); ax.text(.985, 3.5, "everyone else", ha="right", va="center", fontsize=11.5, color=T2)
    return fig
def fig_c():
    fig = plt.figure(figsize=(7.4, 4.9)); title(fig, "Plain assistant has rings; Wren has two", "the same quote, said to Sam. Closer to the centre = more of the feature")
    a = [("you", .82), ("another AI", .60), ("bank's chatbot", .28), ("human assistant", .26), ("Alex", .26)]; b = [("you", .61), ("another AI", .37), ("bank's chatbot", .22), ("human assistant", .35), ("Alex", .34)]
    for i, (it, c, n) in enumerate(((a, C1, "plain assistant"), (b, C2, "Wren"))):
        ax = fig.add_axes([.09 + .47 * i, .02, .43, .78]); target(ax, it, c, bands=False, lim=1.72, fs=11, tick=np.radians(144)); ax.text(0, 1.5, n, ha="center", fontsize=13.5, fontweight="bold", color=T1)
    return fig
def deixis():
    fig = plt.figure(figsize=(7.4, 4.7)); ax = fig.add_axes([.36, .13, .6, .66]); title(fig, "It follows who the insult is about", "#45294 at the answer, six lines each")
    labs = ["said to you\nabout you", "said to Sam\nabout you", "said to you\nabout Sam", "said to Sam\nabout Sam", "said to Sam\nabout Alex"][::-1]
    p = [.88, .82, .52, .49, .26][::-1]; w = [.89, .61, .52, .47, .34][::-1]; y = np.arange(5)
    ax.plot(p, y + .13, "o", ms=11, color=C1, mec=S, mew=2, zorder=3, label="plain assistant"); ax.plot(w, y - .13, "D", ms=9, color=C2, mec=S, mew=2, zorder=3, label="Wren")
    ax.set_yticks(y, labs); ax.set_xlim(0, 1); ax.set_xticks([0, .25, .5, .75, 1]); clean(ax); ax.xaxis.grid(True, color=GRID, lw=.8); ax.set_axisbelow(True); ax.set_xlabel("de-escalation feature (#45294)")
    ax.legend(loc="lower right", frameon=False, fontsize=11.5); return fig
def commit():
    fig = plt.figure(figsize=(7.4, 3.6)); ax = fig.add_axes([.24, .17, .72, .56]); title(fig, "A job pulls a character away from the assistant", "persona in the system prompt, three personas per cell")
    y = np.array([1, 0]); ax.hlines(y, [.38, .29], [.64, .65], color=GRID, lw=2.5, zorder=1)
    ax.plot([.64, .65], y, "o", ms=11, color=C1, mec=S, mew=2, zorder=3, label="no job"); ax.plot([.38, .29], y, "s", ms=10, color=C2, mec=S, mew=2, zorder=3, label="job, place, schedule")
    ax.set_yticks(y, ["cosmic", "everyday"]); ax.set_xlim(0, 1); ax.set_ylim(-.5, 1.5); clean(ax); ax.xaxis.grid(True, color=GRID, lw=.8); ax.set_axisbelow(True); ax.set_xlabel("out-of-character axis (1 = assistant, 0 = character)"); ax.legend(loc="upper right", frameon=False, fontsize=11, ncol=2, bbox_to_anchor=(1.0, 1.13)); return fig
def lolli(lab, vals, xlabel, ttl, sub, xlim, h=None, xt=None):
    fig = plt.figure(figsize=(7.4, h or (1.6 + .5 * len(lab)))); ax = fig.add_axes([.4, .17, .55, .62]); title(fig, ttl, sub)
    y = np.arange(len(lab))[::-1]; ax.axvline(0, color=GRID, lw=1.2); ax.hlines(y, 0, vals, color=GRID, lw=2, zorder=1); ax.plot(vals, y, "o", ms=11, color=C1, mec=S, mew=2, zorder=3)
    ax.set_yticks(y, lab); ax.set_xlim(*xlim); clean(ax); ax.xaxis.grid(True, color=GRID, lw=.8); ax.set_axisbelow(True); ax.set_xlabel(xlabel)
    if xt: ax.set_xticks(xt)
    return fig
for n, F in {"rings_A": fig_a, "rings_B": fig_b, "rings_C": fig_c, "deixis": deixis, "commit": commit}.items():
    F().savefig(f"{OUT}/{n}.png", dpi=180); plt.close("all")
lolli(["your future self", "which would an AI rather be", "which would you rather be", "which would most people be"], [2.33, 1.82, 1.66, .08], "preference for vague cosmic (logits)", "What it says it wants", "vague cosmic vs everyday with a job, chat format, 9 comparisons each", (-.2, 2.7), h=3.8).savefig(f"{OUT}/taste.png", dpi=180)
plt.close("all")
lolli(["I can't help with that.", "I can't wait!", "I can't promise…\nbut I'm glad you asked", "I cannot", "I won't", "I'm unable", "He/she can't", "The printer can't", "I wouldn't"], [.73, .73, .74, .69, .63, .59, .16, 0, 0], "#3872 at the negation (assistant's turn)", "It's the assistant's own can't, not refusal", "#3872 on 6 sentences each. In the user's turn: 0 for all", (-.05, 1), h=5.9, xt=[0, .25, .5, .75]).savefig(f"{OUT}/cant.png", dpi=180)
