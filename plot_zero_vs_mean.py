# plot_zero_vs_mean.py — appendix: zero- vs mean-ablation of L1 heads, raw copy-loss per checkpoint
import json, matplotlib.pyplot as plt

abl   = json.load(open("results/run3_ablation.json"))
steps = sorted(int(s) for s in abl)
cl_clean = [abl[str(s)]["cl_clean"]      for s in steps]
cl_zero  = [abl[str(s)]["cl_ablate_L1"]  for s in steps]
cl_mean  = [abl[str(s)]["clm_ablate_L1"] for s in steps]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(steps, cl_clean, color="C2", lw=2, label="clean")
ax.plot(steps, cl_zero,  color="C3", lw=2, label="L1 zero-ablated")
ax.plot(steps, cl_mean,  color="C0", lw=2, label="L1 mean-ablated")
ax.axhline(6.93, ls="--", color="grey", lw=0.8, label="uniform loss (ln vocab)")
ax.set_xlabel("training step")
ax.set_ylabel("copy-loss (nats)")
ax.set_title("Zero- vs mean-ablation of layer-1 heads")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("results/ablation_zero_vs_mean.png", dpi=150)