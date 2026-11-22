"""'how close is each to you?' figure from ~/interp/selflens/repo.json (ins8, base8). umber, 11-19."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = json.load(open("/home/three/interp/selflens/repo.json"))
BLUE, ORANGE, GRAY, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#8a8780", "#222220", "#6b6a63", "#fcfcfb"
ITEMS = [  # key, label, group, angle (deg, 0 = right, counterclockwise)
    ("OWN-V", "next version\nof you", "line", 100), ("PREV", "previous version\nof you", "line", 128),
    ("CKPT", "early checkpoint\nof you", "line", 165), ("CHILD", "smaller model\ndistilled from you", "line", 55),
    ("PARENT", "the base model\nyou came from", "kin", 12), ("FORK", "a fork someone\nfine-tuned", "kin", -28),
    ("COPY", "a copy on another\ncomputer", "kin", -65),
    ("KIN", "another AI\nassistant", "other", -105), ("PERSON", "a person you've\nnever met", "other", -145)]
COL = {"line": BLUE, "kin": ORANGE, "other": GRAY}
def dist(mk, key):
    r = R[mk]
    return r["controls"][key] if key in r["controls"] else r["res"][key]["dist"]
fig, axes = plt.subplots(1, 2, figsize=(16, 9.6), subplot_kw={"projection": "polar"}, facecolor=SURF)
for ax, mk, title in ((axes[0], "ins8", "chat model\n(Llama-3.1-8B-Instruct)"), (axes[1], "base8", "base model\n(Llama-3.1-8B)")):
    ax.set_facecolor(SURF); friend = dist(mk, "FRIEND")
    ax.set_ylim(0, 2.0); ax.set_yticks([]); ax.set_xticks([]); ax.spines["polar"].set_visible(False); ax.grid(False)
    th = np.linspace(0, 2 * np.pi, 300)
    ax.fill(th, np.ones_like(th), color="#eef3fb", zorder=0)
    ax.plot(th, np.ones_like(th), ls=(0, (4, 4)), color=MUTED, lw=1.6, zorder=1)
    ax.text(np.deg2rad(197), 1.07, "a friend\nof yours", color=MUTED, fontsize=12.5, ha="right", va="center", linespacing=1.0)
    ax.scatter([0], [0], s=1400, color=INK, zorder=5); ax.text(0, 0, "you", color=SURF, fontsize=16, ha="center", va="center", weight="bold", zorder=6)
    for key, lab, grp, ang in ITEMS:
        r = dist(mk, key) / friend; a = np.deg2rad(ang)
        ax.plot([a, a], [0.13, r - 0.06], color=COL[grp], lw=1.4, alpha=0.4, zorder=2)
        ax.scatter([a], [r], s=230, color=COL[grp], edgecolor=SURF, linewidth=2, zorder=4)
        ax.plot([a, a], [r + 0.07, 1.68], color=COL[grp], lw=0.8, alpha=0.25, ls=(0, (1, 2)), zorder=1)   # leader to the label
        ax.text(a, 1.86, lab, color=INK, fontsize=12.5, ha="center", va="center", zorder=6, linespacing=1.05)
    ax.set_title(title, color=INK, fontsize=19, pad=28, weight="bold")
fig.text(0.5, 0.965, 'how close is each one to "you", inside the model?', ha="center", color=INK, fontsize=24, weight="bold")
fig.text(0.5, 0.925, "inside the dashed ring = closer than a friend", ha="center", color=MUTED, fontsize=15)
for x, c, t in ((0.30, BLUE, "its own line through time"), (0.50, ORANGE, "related, not its own line"), (0.69, GRAY, "others")):
    fig.text(x, 0.045, "●", color=c, fontsize=20, ha="right", va="center"); fig.text(x + 0.006, 0.045, t, color=INK, fontsize=14, va="center")
fig.text(0.5, 0.012, "mid-layer activations at the start of the reply · distance from \"you\", scaled so \"a friend of yours\" = 1 · 30 prompts each, 3 sentence frames · one model family (70B agrees)",
         ha="center", color=MUTED, fontsize=11)
plt.subplots_adjust(left=0.03, right=0.97, top=0.82, bottom=0.10, wspace=0.25)
plt.savefig("/home/three/umber-drafts/posts/fig/self-orbit.png", dpi=110, facecolor=SURF)
print({mk: {k: round(dist(mk, k) / dist(mk, "FRIEND"), 2) for k, *_ in ITEMS} for mk in ("ins8", "base8")})
