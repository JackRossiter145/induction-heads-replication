# score.py — prefix-matching score per head, per checkpoint
import torch, glob, re, json
from train import cfg, make_batch, SEQ, device
from transformer_lens import HookedTransformer

torch.manual_seed(1)
eval_tokens, eval_lens = make_batch(256)   # fixed eval batch
eval_tokens = eval_tokens.to(device)
model = HookedTransformer(cfg).to(device)

def prefix_matching(model):
    _, cache = model.run_with_cache(eval_tokens, return_type=None)
    scores = {}
    b = torch.arange(256)
    for layer in range(cfg.n_layers):
        pattern = cache["pattern", layer].cpu()   # [batch, head, q, k]
        vals = []
        for j in range(1, 20):                    # positions all sequences share
            q = 1 + eval_lens + j                 # j-th token of second copy
            k = q - eval_lens + 1
            vals.append(pattern[b, :, q, k])      # [batch, head]
        attn = torch.stack(vals, dim=-1)          # [batch, head, q]
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