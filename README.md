# Induction heads replication

## The figure

**Target:** reproduce the induction-head phase change from Olsson et al. (2022),
"In-context Learning and Induction Heads", at the smallest scale that shows it.

**Model:** 2-layer attention-only transformer (TransformerLens,
`attn_only=True`), d_model 128, 4 heads/layer, no MLPs, learned positional embeddings.

**Data:** sequences of length 100 built as [BOS] + 50 random tokens + the same
50 tokens repeated. Vocab ~1000. Fresh random batch every step.

**Training:** AdamW, ~2–3k steps, batch 64, checkpoint every 25 steps.

**Metrics (per checkpoint, fixed eval batch of 256 sequences):**
1. *Prefix-matching score* per head: mean attention from token t (second half)
   to the token following its earlier copy. Plot the best head across layers.
2. *In-context learning score*: loss at position ~90 minus loss at position ~60
   (both in the repeated half). Olsson's co-occurrence claim.

**Plot:** x = training step; y = best-head prefix-matching score (left axis)

and ICL score (right axis). Success = a sharp sigmoid in (1), with (2) dropping
at the same step.

**Extension (after the figure exists):** behavioural eval — does a
copying-accuracy test on held-out sequences detect the capability at the same
step as the mechanistic score, or lag it?
