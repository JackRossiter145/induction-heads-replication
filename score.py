# score.py — prefix-matching score per head, per checkpoint
import torch, glob, re, json
from train import cfg, make_batch, SEQ, device
from transformer_lens import HookedTransformer

torch.manual_seed(1)
eval_tokens = make_batch(256).to(device)     # fixed eval batch
model = HookedTransformer(cfg).to(device)

def prefix_matching(model):
    _, cache = model.run_with_cache(eval_tokens, return_type=None)
    scores = {}
    q = torch.arange(SEQ + 1, 2 * SEQ)       # queries in repeated half
    k = q - SEQ + 1                          # token after earlier copy
    for layer in range(cfg.n_layers):
        pattern = cache["pattern", layer]    # [batch, head, q, k]
        attn = pattern[:, :, q, k]           # [batch, head, q]
        for h in range(cfg.n_heads):
            scores[f"L{layer}H{h}"] = attn[:, h].mean().item()
    return scores

results = {}
for path in sorted(glob.glob("checkpoints/step_*.pt")):
    step = int(re.search(r"step_(\d+)", path).group(1))
    model.load_state_dict(torch.load(path, map_location=device))
    results[step] = prefix_matching(model)
    print(step, max(results[step].items(), key=lambda kv: kv[1]))

json.dump(results, open("prefix_scores.json", "w"))