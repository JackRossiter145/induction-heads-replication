# plot_ablation.py — L1 mean-ablation Δ copy-loss vs L1 prefix-matching, per checkpoint
import json, matplotlib.pyplot as plt

abl = json.load(open("results/run3_ablation.json"))
sc  = json.load(open("results/run3_scores.json"))
steps  = sorted(int(s) for s in abl)
delta  = [abl[str(s)]["delta_clm_L1"] for s in steps]
heads  = [[sc[str(s)][f"L1H{h}"] for s in steps] for h in range(4)]            # one list per head
prefix = [sum(sc[str(s)][f"L1H{h}"] for h in range(4)) / 4 for s in steps]    # layer-1 mean

def draw(ax1, xlim, legend=False):
    ax1.plot(steps, delta, color="C0", lw=2, label="L1 mean-ablation Δ copy-loss (nats)")
    ax1.axhline(6.93, ls="--", color="grey", lw=0.8, label="uniform loss (ln vocab)")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("Δ copy-loss (nats)", color="C0")
    ax1.set_xlim(*xlim)
    ax2 = ax1.twinx()
    for h in range(4):
        ax2.plot(steps, heads[h], color="C1", alpha=0.25, lw=0.8,
                 label="L1 heads (individual)" if h == 0 else None)
    ax2.plot(steps, prefix, color="C1", lw=2, label="L1 prefix-matching (mean)")
    ax2.set_ylabel("prefix-matching score", color="C1")
    ax2.set_ylim(0, 1)
    if legend:
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="center left", fontsize=8)

fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12, 4),
                                       gridspec_kw={"width_ratios": [1, 2]})
draw(ax_full, (0, 3000))
draw(ax_zoom, (300, 1000), legend=True)
ax_full.set_title("Full training run")
ax_zoom.set_title("Phase-change window")
fig.tight_layout()
fig.savefig("results/ablation_timing.png", dpi=150)