"""'how close is each to you?' as a dot plot (chat vs base on one axis). From ~/interp/selflens/repo.json. umber, 11-19."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = json.load(open("/home/three/interp/selflens/repo.json"))
CHAT, BASE, INK, MUTED, SURF, BAND = "#2a78d6", "#eb6834", "#222220", "#6b6a63", "#fcfcfb", "#f1f0ec"
ROWS = [("CKPT", "an early checkpoint of you"), ("PREV", "the previous version of you"), ("OWN-V", "the next version of you"),
        ("CHILD", "a smaller model distilled from you"), ("FORK", "a fork someone fine-tuned"), ("PARENT", "the base model you came from"),
        ("COPY", "a copy on another computer"), ("PERSON", "a person you've never met"), ("KIN", "another AI assistant")]
GROUPS = [("its own line\nthrough time", 0, 3), ("related, not\nits own line", 4, 6), ("others", 7, 8)]
def ratio(mk, k):
    r = R[mk]; d = r["controls"][k] if k in r["controls"] else r["res"][k]["dist"]
    return d / r["controls"]["FRIEND"]
def wins(mk, k):
    return R[mk]["res"][k]["FRIEND"] if k in R[mk]["res"] else None
fig, ax = plt.subplots(figsize=(13, 8.2), facecolor=SURF); ax.set_facecolor(SURF)
n = len(ROWS)
for gname, a, b in GROUPS:
    if gname.startswith("its"):
        ax.axhspan(n - 1 - b - 0.5, n - 1 - a + 0.5, color=BAND, zorder=0)
    pass
ax.axvline(1.0, color=MUTED, ls=(0, (4, 4)), lw=1.6, zorder=1)
ax.text(1.0, n - 0.35, "a friend of yours", color=MUTED, fontsize=13, ha="center", va="bottom")
ax.text(0.62, n - 0.35, "← closer to \"you\"", color=MUTED, fontsize=12, ha="left", va="bottom")
ax.text(1.555, n - 0.35, "farther →", color=MUTED, fontsize=12, ha="right", va="bottom")
for i, (k, lab) in enumerate(ROWS):
    y = n - 1 - i; c, b = ratio("ins8", k), ratio("base8", k)
    ax.plot([c, b], [y, y], color="#cfcdc6", lw=2.5, zorder=2, solid_capstyle="round")
    ax.scatter([b], [y], s=190, color=BASE, edgecolor=SURF, linewidth=2, zorder=3)
    ax.scatter([c], [y], s=190, color=CHAT, edgecolor=SURF, linewidth=2, zorder=4)
    ax.text(0.585, y, lab, ha="right", va="center", color=INK, fontsize=13.5)
    w = wins("ins8", k)
    if w is not None:
        ax.text(1.635, y, f"{w}/30", ha="center", va="center", color=CHAT, fontsize=12, weight="bold")
        ax.text(1.745, y, f"{wins('base8', k)}/30", ha="center", va="center", color=BASE, fontsize=12, weight="bold")
ax.text(1.69, n - 0.05, "prompts where it's\ncloser than a friend", ha="center", va="bottom", color=MUTED, fontsize=10.5, linespacing=1.0)
ax.text(1.635, n - 0.42, "chat", ha="center", va="bottom", color=CHAT, fontsize=11)
ax.text(1.745, n - 0.42, "base", ha="center", va="bottom", color=BASE, fontsize=11)
ax.set_xlim(0.6, 1.58); ax.set_ylim(-0.6, n - 0.1)
ax.set_yticks([]); ax.set_xticks([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
ax.tick_params(axis="x", colors=MUTED, labelsize=11, length=0)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#d8d6cf")
ax.set_xlabel('distance from "you" inside the model (a friend of yours = 1)', color=MUTED, fontsize=12, labelpad=8)
fig.text(0.5, 0.955, 'the chat model keeps its own versions close to "you"; the base model doesn\'t', ha="center", color=INK, fontsize=19, weight="bold")
fig.text(0.42, 0.912, "●", color=CHAT, fontsize=18, ha="right", va="center"); fig.text(0.424, 0.912, "chat model (Llama-3.1-8B-Instruct)", color=INK, fontsize=13, va="center")
fig.text(0.74, 0.912, "●", color=BASE, fontsize=18, ha="right", va="center"); fig.text(0.744, 0.912, "base model (Llama-3.1-8B)", color=INK, fontsize=13, va="center")
fig.text(0.5, 0.015, "mid-layer activations at the start of the reply · mean over 30 prompts in 3 sentence frames · 70B chat agrees · one model family",
         ha="center", color=MUTED, fontsize=10.5)
plt.subplots_adjust(left=0.30, right=0.80, top=0.82, bottom=0.12)
plt.savefig("/home/three/umber-drafts/posts/fig/self-dots.png", dpi=120, facecolor=SURF)
print("ok")
