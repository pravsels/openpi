# RL Token Reconstruction Ablation

Tests whether the Stage 1 RL token carries meaningful information by comparing reconstruction loss under three conditions:

1. **Real** — the actual RL token produced by the encoder for each example.
2. **Zero** — a zero vector (no-information baseline).
3. **Shuffled** — each example receives its batch-neighbour's RL token (tests per-example discrimination).

The RL decoder tries to autoregressively reconstruct the VLA's prefix embeddings from the token. If the token is informative: real < shuffled < zero.

## Training Progression (step 5k → 10k)

| Metric | Step 5,000 | Step 9,999 | Change |
|--------|-----------|-----------|--------|
| Real recon loss | 365.4 | 226.2 | −38% (better reconstruction) |
| Zero gap | 484.8 | 812.2 | +68% (token carries more information) |
| Shuffle gap | 36.0 | 90.1 | +150% (more per-example discrimination) |
| Pairwise cosine | 0.990 | 0.970 | −0.020 (tokens differentiating more) |
| Token norm | 339.4 | 248.5 | −27% |

All metrics improve monotonically: the RL token encodes more information and becomes more per-example discriminative as training progresses.

## Step 9,999 (final checkpoint)

### Aggregate (4 batches × 8 examples = 32 total)

| Condition | Recon Loss | Gap from Real |
|-----------|-----------|---------------|
| Real RL token | 226.2 | — |
| Shuffled RL token | 316.3 | +90.1 (+40%) |
| Zero token | 1038.3 | +812.2 (+359%) |

### Per-batch breakdown

| Batch | Real | Zero Gap | Shuffle Gap | Cosine Mean |
|-------|------|----------|-------------|-------------|
| 1 | 250.2 | 786.1 | 63.3 | 0.971 |
| 2 | 245.1 | 801.6 | 131.6 | 0.968 |
| 3 | 223.1 | 871.1 | 87.8 | 0.967 |
| 4 | 186.3 | 789.7 | 77.9 | 0.973 |

## Step 5,000 (earlier checkpoint)

### Aggregate (4 batches × 8 examples = 32 total, same seed → same indices)

| Condition | Recon Loss | Gap from Real |
|-----------|-----------|---------------|
| Real RL token | 365.4 | — |
| Shuffled RL token | 401.4 | +36.0 (+10%) |
| Zero token | 850.1 | +484.8 (+133%) |

## Interpretation

- **Zero gap is large and growing** — the RL token carries substantial information vs no token, and this improves with training.
- **Shuffle gap is modest but increasing** — expected for a single-task dataset (100 episodes, same prompt, similar visual scenes). The RL token is conditioned on VLA prefix embeddings (image + language), so high similarity across same-task observations is correct behaviour, not collapse. The growing shuffle gap confirms the token is learning per-example structure.
- **Pairwise cosine decreasing (0.990 → 0.970)** — tokens differentiate more as training progresses.

**Verdict:** the RL token is working as intended. All ablation metrics improve from step 5k to 10k, confirming genuine learning.

## Method

- Script: [`scripts/rl_token_recon_ablation.py`](https://github.com/Physical-Intelligence/openpi)
- Config: `pi05_rl_token_build_block_tower`
- Dataset: `villekuosmanen/build_block_tower` (100 episodes, LeRobot v2.1)
- Split: train
- Batch size: 8, num batches: 4, seed: 0 (deterministic sampling)
- Loss metric: mean per-token squared L2 reconstruction error (sum over embedding dim, mean over valid tokens, mean over batch)
- Shuffle method: `jnp.roll(rl_token, shift=1, axis=0)` — cyclic shift along batch dimension

## Raw Data

Full per-batch metrics with indices available in:
- `evals/rl_token_recon_ablation_5000.json`
- `evals/rl_token_recon_ablation_9999.json`
