# score.py — prefix-matching, icl, and previous-token score per head, per checkpoint
import torch, glob, re, json
from train import cfg, make_batch, SEQ, device
from transformer_lens import HookedTransformer

RUN = "run3"

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

def icl_score(model):
    loss_per_token = model(eval_tokens, return_type="loss", loss_per_token=True).cpu()     # [batch, n_ctx - 1]
    b = torch.arange(256)
    earlier_tokens_loss = []
    later_tokens_loss = []
    for j in range(1, 6):
        p = eval_lens + j
        earlier_tokens_loss.append(loss_per_token[b, p])
    for j in range(15, 20):
        p = eval_lens + j
        later_tokens_loss.append(loss_per_token[b, p])
    earlier_tokens_mean_loss = torch.stack(earlier_tokens_loss).mean()
    later_tokens_mean_loss = torch.stack(later_tokens_loss).mean()
    return (later_tokens_mean_loss - earlier_tokens_mean_loss).item()

def prev_token(model):
    _, cache = model.run_with_cache(eval_tokens, return_type=None)
    scores = {}
    for layer in range(cfg.n_layers):
        pattern = cache["pattern", layer].cpu()   # [batch, head, q, k]
        vals = []
        for q in range(2, 41):                         
            k = q - 1
            vals.append(pattern[:, :, q, k])      # [batch, head]
        attn = torch.stack(vals, dim=-1)          # [batch, head, q]
        for h in range(cfg.n_heads):
            scores[f"prev_L{layer}H{h}"] = attn[:, h].mean().item()
    return scores

results = {}
for path in sorted(glob.glob("checkpoints/step_*.pt")):
    step = int(re.search(r"step_(\d+)", path).group(1))
    model.load_state_dict(torch.load(path, map_location=device))
    results[step] = prefix_matching(model)
    results[step]["icl"] = icl_score(model)
    results[step].update(prev_token(model))
    best = max((kv for kv in results[step].items() if kv[0].startswith("L")), key=lambda kv: kv[1])
    best_prev_token = max((kv for kv in results[step].items() if kv[0].startswith("prev_")), key=lambda kv: kv[1])
    print(step, best, "icl", round(results[step]["icl"], 3), best_prev_token)

json.dump(results, open(f"results/{RUN}_scores.json", "w"))
