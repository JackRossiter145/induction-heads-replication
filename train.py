# train.py — train a 2-layer attention-only transformer on repeated random tokens
import torch, os
from transformer_lens import HookedTransformer, HookedTransformerConfig

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

torch.manual_seed(0)
VOCAB, SEQ, BOS = 1000, 50, 1000          # tokens 0..999, BOS = 1000
STEPS, BATCH, CKPT_EVERY = 3000, 64, 25

cfg = HookedTransformerConfig(
    n_layers=2, d_model=128, n_heads=4, d_head=32,
    n_ctx=2*SEQ + 21, d_vocab=VOCAB + 1,
    attn_only=True, act_fn=None,
    positional_embedding_type="standard", seed=0,
)
model = HookedTransformer(cfg)
model.to(device)

def make_batch(n):
    seqs, offsets = [], []
    for _ in range(n):
        p = torch.randint(0, 20, (1,)).item()
        pre = torch.randint(0, VOCAB, (p,))
        a = torch.randint(0, VOCAB, (SEQ,))
        s = torch.cat([torch.tensor([BOS]), pre, a, a])
        seqs.append(torch.nn.functional.pad(s, (0, cfg.n_ctx - len(s)), value=BOS))
        offsets.append(p)
    return torch.stack(seqs), torch.tensor(offsets)

if __name__ == "__main__":
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    os.makedirs("checkpoints", exist_ok=True)

    for step in range(STEPS + 1):
        tokens, _ = make_batch(BATCH)
        tokens = tokens.to(device)
        loss = model(tokens, return_type="loss")
        opt.zero_grad(); loss.backward(); opt.step()
        if step % CKPT_EVERY == 0:
            torch.save(model.state_dict(), f"checkpoints/step_{step:05d}.pt")
            print(f"step {step:5d}  loss {loss.item():.3f}")
