# ablate.py — per-checkpoint head ablation, ICL delta
import torch, glob, re, json
from functools import partial
from train import cfg, device
from score import model, icl_score, eval_tokens, eval_lens


RUN = "run3"

def zero_heads_hook(z, hook, heads):
    z[:, :, heads, :] = 0.0
    return z

def ablated(model, metric, layer, heads=(0, 1, 2, 3)):
    hook = (f"blocks.{layer}.attn.hook_z", partial(zero_heads_hook, heads=list(heads)))
    with model.hooks(fwd_hooks=[hook]):
        return metric(model)

def copy_loss(model):
    lpt = model(eval_tokens, return_type="loss", loss_per_token=True).cpu()
    b = torch.arange(256)
    return torch.stack([lpt[b, eval_lens + j] for j in range(1, 20)]).mean().item()

def mean_heads_hook(z, hook, heads, mean):
    z[:, :, heads, :] = mean[heads]
    return z

def ablated_mean(model, metric, layer, heads=(0,1,2,3)):
    name = f"blocks.{layer}.attn.hook_z"
    _, cache = model.run_with_cache(eval_tokens, names_filter=name)
    mean = cache[name].mean(dim=(0, 1))          # [head, d_head]
    with model.hooks(fwd_hooks=[(name, partial(mean_heads_hook, heads=list(heads), mean=mean))]):
        return metric(model)

results = {}
for path in sorted(glob.glob("checkpoints/step_*.pt")):
    step = int(re.search(r"step_(\d+)", path).group(1))
    model.load_state_dict(torch.load(path, map_location=device))
    clean_icl, clean_cl = icl_score(model), copy_loss(model)
    a1_icl, a0_icl = ablated(model, icl_score, 1), ablated(model, icl_score, 0)
    a1_cl,  a0_cl  = ablated(model, copy_loss, 1), ablated(model, copy_loss, 0)
    b1_cl,  b0_cl  = ablated_mean(model, copy_loss, 1), ablated_mean(model, copy_loss, 0)
    results[step] = dict(
        icl_clean=clean_icl, icl_ablate_L1=a1_icl, icl_ablate_L0=a0_icl,
        delta_icl_L1=a1_icl - clean_icl, delta_icl_L0=a0_icl - clean_icl,
        cl_clean=clean_cl, cl_ablate_L1=a1_cl, cl_ablate_L0=a0_cl,
        delta_cl_L1=a1_cl - clean_cl, delta_cl_L0=a0_cl - clean_cl,
        clm_ablate_L1=b1_cl, clm_ablate_L0=b0_cl,
        delta_clm_L1=b1_cl - clean_cl, delta_clm_L0=b0_cl - clean_cl,
    )
    print(step, {k: round(v, 3) for k, v in results[step].items()})

json.dump(results, open(f"results/{RUN}_ablation.json", "w"))