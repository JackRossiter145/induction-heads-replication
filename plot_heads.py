import json
from matplotlib import pyplot as plt

with open('results/run3_scores.json', 'r') as x:
    results = json.load(x)

results = {int(k): v for k, v in results.items()}

steps = []
prev_token_scores = {}
prefix_matching_scores = {}

for k, v in sorted(results.items()):
    steps.append(k)
    for head, score in v.items():
        if head.startswith("prev_"):
            prev_token_scores.setdefault(head, []).append(score)
        elif head.startswith("L"):
            prefix_matching_scores.setdefault(head, []).append(score)

fig, (ax_top, ax_bot)  = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

for head, scores in prefix_matching_scores.items():
    if head.startswith("L1"):
        ax_top.plot(
            steps, scores,
            color='tab:blue',
            alpha=0.45,
            linewidth=1.2,
            label="Layer 1 Heads" if head == "L1H0" else None
        )
    else:
        ax_top.plot(
            steps, scores,
            color='tab:orange',
            alpha=0.45,
            linewidth=1.2,
            label="Layer 0 Heads" if head == "L0H0" else None
        )

for head, scores in prev_token_scores.items():
    if head.startswith("prev_L1"):
        ax_bot.plot(
            steps, scores,
            color='tab:blue',
            alpha=0.45,
            linewidth=1.2,
            label="Layer 1 Heads" if head == "prev_L1H0" else None
        )
    else:
        ax_bot.plot(
            steps, scores,
            color='tab:orange',
            alpha=0.45,
            linewidth=1.2,
            label="Layer 0 Heads" if head == "prev_L0H0" else None
        )

ax_top.set_ylabel('Prefix Matching Score')
ax_top.set_title('Run 3 Induction Head Formation')

ax_bot.set_xlabel('Steps')
ax_bot.set_ylabel('Previous Token Score')

ax_top.legend(loc='best')
ax_bot.legend(loc='best')

fig.tight_layout()

plt.savefig("results/run3_heads.png", dpi=150)
