# RL token reconstruction ablation — build_block_tower step 9999

## Mode
- run_type: experiment
- objective: Quick ablation to test whether the Stage 1 RL token carries per-example information, by comparing reconstruction loss under real, zeroed, and shuffled token conditions.

## Config
- script: `scripts/rl_token_recon_ablation.py`
- slurm: `slurm/rl_token_recon_ablation_slurm.sh` (job name `rlt_ablation`)
- config: `pi05_rl_token_build_block_tower`
- checkpoint: `/scratch/u6cr/pravsels.u6cr/openpi/checkpoints/pi05_rl_token_build_block_tower/rlt_v1/9999/params`
- assets: `/scratch/u6cr/pravsels.u6cr/openpi/checkpoints/pi05_rl_token_build_block_tower/rlt_v1/9999/assets`
- output: `/scratch/u6cr/pravsels.u6cr/openpi/eval_outputs/build_block_tower_rlt_stage2/rl_token_recon_ablation_9999.json`
- dataset: `villekuosmanen/build_block_tower` (100 episodes, single task)
- split: train
- batch_size: 8, num_batches: 4 (32 examples total)

## Method
The ablation runs three conditions through the frozen RL decoder:
1. **Real** — the actual RL token produced by the encoder for each example
2. **Zero** — a zero vector in place of the RL token
3. **Shuffled** — each example receives its batch neighbour's RL token (`jnp.roll(rl_token, shift=1, axis=0)`)

Additionally computes pairwise cosine similarity and L2 norm statistics across RL tokens in each batch.

## Job
- job_id: 3395012
- submitted: 2026-03-27
- start: `2026-03-27T13:11:50+00:00`
- end: `2026-03-27T13:13:05+00:00`
- runtime: 00:01:15
- node: nid010052

## Status
- 2026-03-27 13:11 — submitted and completed on nid010052

## Results — step 9,999 (primary)

### Aggregate (4 batches, 32 examples)

| Condition | Recon Loss | Gap from Real |
|-----------|-----------|---------------|
| Real RL token | 226.2 | — |
| Shuffled RL token | 316.3 | +90.1 (+40%) |
| Zero token | 1038.3 | +812.2 (+359%) |

### RL token statistics

| Metric | Value |
|--------|-------|
| Norm mean | 248.5 |
| Norm std | 5.0 |
| Pairwise cosine mean | 0.970 |
| Pairwise cosine std | 0.006 |

### Per-batch breakdown

| Batch | Real | Zero Gap | Shuffle Gap | Cosine Mean |
|-------|------|----------|-------------|-------------|
| 1 | 250.2 | 786.1 | 63.3 | 0.971 |
| 2 | 245.1 | 801.6 | 131.6 | 0.968 |
| 3 | 223.1 | 871.1 | 87.8 | 0.967 |
| 4 | 186.3 | 789.7 | 77.9 | 0.973 |

## Results — step 5,000 (earlier checkpoint comparison)

- job_id: 3454075
- runtime: 00:01:36
- node: nid010052
- output: `/scratch/u6cr/pravsels.u6cr/openpi/eval_outputs/build_block_tower_rlt_stage2/rl_token_recon_ablation_5000.json`

### Aggregate (4 batches, 32 examples, same seed → same indices)

| Condition | Recon Loss | Gap from Real |
|-----------|-----------|---------------|
| Real RL token | 365.4 | — |
| Shuffled RL token | 401.4 | +36.0 (+10%) |
| Zero token | 850.1 | +484.8 (+133%) |

### RL token statistics

| Metric | Value |
|--------|-------|
| Norm mean | 339.4 |
| Norm std | 6.2 |
| Pairwise cosine mean | 0.990 |
| Pairwise cosine std | 0.005 |

## Training progression (step 5k → 10k)

| Metric | Step 5,000 | Step 9,999 | Change |
|--------|-----------|-----------|--------|
| Real recon loss | 365.4 | 226.2 | −38% (better reconstruction) |
| Zero gap | 484.8 | 812.2 | +68% (token carries more information) |
| Shuffle gap | 36.0 | 90.1 | +150% (more per-example discrimination) |
| Pairwise cosine | 0.990 | 0.970 | −0.020 (tokens differentiating more) |
| Token norm | 339.4 | 248.5 | −27% (decreasing) |

All metrics move in the expected direction: as training progresses the RL token encodes more information, reconstructs better, and becomes more discriminative across examples.

## Interpretation
- **Zero gap is large and growing:** the RL token carries substantial information vs no token, and this improves with training.
- **Shuffle gap is modest but increasing:** expected for a single-task dataset where all 100 episodes share the same prompt and similar visual scenes. The RL token is conditioned on VLA prefix embeddings (image + language), so high similarity across same-task observations is the correct behaviour, not collapse. The growing shuffle gap confirms the token is learning per-example structure, not just a fixed bias.
- **Pairwise cosine decreasing from 0.990 → 0.970:** tokens are differentiating more as training progresses, consistent with the encoder learning to encode observation-specific information.

**Verdict:** the RL token is working as intended for a single-task setting. All ablation metrics improve monotonically from step 5k to 10k, confirming genuine learning rather than a degenerate solution.

## Next
- The original full probe suite (`validate_rl_token.py`, job 3391484) timed out at 12h. If the per-example probes (action probe, linear state probe, subtask classifier) are still needed, the script needs profiling or the feature extraction needs to be decoupled from probe training.
- A more informative cosine/shuffle test would use a multi-task dataset or compare tokens across subtask phases.
- Could also run this ablation on steps 15k and 20k to check whether improvement continues or plateaus.
