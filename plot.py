import json
from matplotlib import pyplot as plt
import numpy as np

with open('results/run3_scores.json', 'r') as x:
    results = json.load(x)

results = {int(k): v for k, v in results.items()}

steps = []
icl = []
head_scores = {}

for k, v in sorted(results.items()):
    steps.append(k)
    icl.append(v['icl'])

    for head, score in v.items():
        if head.startswith("L"):
            head_scores.setdefault(head, []).append(score)

fig, ax1 = plt.subplots(figsize=(10, 6))

layer_1_scores = []
for head, scores in head_scores.items():
    if head.startswith("L1"):
        layer_1_scores.append(scores)
        ax1.plot(
            steps, scores,
            color='tab:blue',
            alpha=0.3,
            linewidth=1.2,
            label="Other Layer 1 Heads" if head == "L1H0" else None
        )
    else:
        ax1.plot(
            steps, scores,
            color='tab:orange',
            alpha=0.3,
            linewidth=1.2,
            label="Layer 0 Heads" if head == "L0H0" else None
        )
layer_1_mean = np.mean(layer_1_scores, axis=0)
ax1.plot(
    steps, layer_1_mean, 
    color='tab:blue',
    linewidth=2.5,
    label="Layer 1 Mean",
    zorder=3
)

ax1.set_xlabel('Steps')
ax1.set_ylabel('Prefix Matching Score')
ax1.set_title('Run 3 Induction Head Formation')

ax2 = ax1.twinx()
ax2.plot(
    steps, icl,
    color='tab:red',
    linewidth=2,
    label='ICL'
)
ax2.set_ylabel('In Context Learning Score', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

fig.tight_layout()

plt.savefig("results/run3_plot.png", dpi=150)
