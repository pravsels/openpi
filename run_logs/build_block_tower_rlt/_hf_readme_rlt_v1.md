---
license: apache-2.0
tags:
 - robotics
 - pi0
 - rl-token
 - block-tower
 - openpi
---

# pi0.5 Build Block Tower — RLT Stage 1 (Encoder-Decoder)

RL Token encoder-decoder trained on top of a frozen pi0.5 baseline VLA for building a block tower. Implements Stage 1 of the [RL Token method](https://www.pi.website/research/rlt) (Xu et al., 2026): a lightweight transformer encoder-decoder compresses VLA prefix embeddings into a single RL token via autoregressive reconstruction.

## Experiment

- **Objective:** Train RLT encoder-decoder to produce a compact RL token representation from frozen VLA prefix embeddings.
- **VLA backbone:** Baseline 55k checkpoint (`pravsels/pi05-build-block-tower-baseline`), frozen (`rl_vla_loss_weight=0.0`).
- **Encoder-decoder:** 2-layer transformer, 8 heads, dim=2048, SwiGLU FFN.
- **Loss:** Autoregressive reconstruction of VLA prefix embeddings (L2).
- **Validation:** Deterministic episode-level `90/10` train/val split with held-out episode IDs saved in `assets/episode_split.json`.
- **Steps:** 10,000

## Config

- **Config name:** `pi05_rl_token_build_block_tower`
- **Model:** `Pi0RLConfig` (`pi05=True`, `action_horizon=50`, `rl_vla_loss_weight=0.0`)
- **Batch size:** 36
- **Learning rate:** 5e-5 cosine decay (1k warmup)
- **Optimizer:** AdamW (gradient clip norm 1.0)
- **EMA decay:** 0.999
- **Delta actions:** enabled
- **State/action space:** 7D joint-space

## Dataset

- `villekuosmanen/build_block_tower` (200 episodes, LeRobot v2.1)
- Train/val separation is by whole episode, not timestep, to avoid leakage.

## Checkpoint Hashes

Verify integrity with `find params -type f | sort | xargs sha256sum | sha256sum`.

| Step | Train Loss | Val Loss | SHA-256 |
|------|------------|----------|---------|
| 9,999 | 216.8683 | 286.5721 | `4378fc1886f7eef6adab8a123ec491cde783c9aa94cd60a0b57757314862ed95` |

## W&B

- [pravsels/openpi-rlt-block-tower/runs/oa2o0o0z](https://wandb.ai/pravsels/openpi-rlt-block-tower/runs/oa2o0o0z)

## Evaluation

Evaluation artifacts for the frozen `9999` checkpoint are available under `evals/`.

All evaluations depend on two artifacts together:
- the frozen base pi0.5 VLA: `pravsels/pi05-build-block-tower-baseline`
- the Stage 1 RLT encoder-decoder checkpoint in this repo: step `9999`

### Cosine Similarity (ID vs OOD)

Tested on 1 ID episode (1,028 frames) and 1 OOD episode (486 frames). Raw cosine similarity does **not** cleanly separate ID (`build_block_tower`) from OOD (`drop_footbag_into_dice_tower`).

| Comparison | Mean cosine | Std |
|------------|------------|-----|
| Within ID (build_block_tower) | 0.972 | 0.010 |
| Within OOD (drop_footbag) | 0.988 | 0.005 |
| Cross-task (ID vs OOD) | 0.974 | 0.006 |
| Episode-level (mean-pooled) | 0.994 | — |

Within-ID similarity (0.972) and cross-task similarity (0.974) are nearly identical — the token doesn't distinguish between tasks any more than it distinguishes between frames of the same task. The most likely failure mode is not a dead token, but a token dominated by a large shared component with useful information compressed into smaller residual directions.

See `evals/2026-03-27_rl_token_eval/eval_log.md` for the full interpretation and the accompanying JSON/plot artifacts.

### Reconstruction Ablation (step 5k vs 10k)

Tests whether the RL token carries meaningful information by comparing decoder reconstruction loss under three conditions: real token, zero vector, and shuffled (batch-neighbour's) token. Evaluated on 32 timesteps (4 batches × 8) from the train split. Loss is mean L2² per token (summed over embedding dim), averaged over valid tokens per example, then averaged over the batch.

| Condition | Mean L2 (Step 5k) | Mean L2 (Step 10k) |
|-----------|-------------------|---------------------|
| Real RL token | 365.4 | 226.2 |
| Neighbour's token | 401.4 (+10%) | 316.3 (+40%) |
| Zero vector | 850.1 (+133%) | 1038.3 (+359%) |

Percentages are relative to the real RL token loss at the same checkpoint. Pairwise cosine similarity between tokens decreased from 0.990 (5k) to 0.970 (10k), confirming tokens are differentiating more across examples.

All metrics improve from step 5k to 10k, confirming the RL token is learning genuine information rather than collapsing. The modest neighbour gap is expected for a single-task dataset (100 episodes, same prompt) — tokens are legitimately similar across same-task observations.

See `evals/2026-03-27_rl_token_eval/recon_ablation_progression.md` for full analysis and per-batch breakdowns.

### Probe Suite (step 10k)

Tests whether the frozen RL token is informative enough for downstream actor-critic work by training lightweight PyTorch probes on extracted features (5k train / 1k val samples, episode-level split).

| Probe | Input | Target | Val Metric | Baseline | Delta |
|-------|-------|--------|------------|----------|-------|
| Action MLP | rl_token + state | VLA action chunk | MSE **0.1517** | state-only: 0.1612 | **-6%** |
| Action MLP | rl_token + state | ground-truth action | MSE **0.0088** | — | — |
| Linear state | rl_token | normalized state | MSE **0.0555** | random vector: 0.0785 | **-29%** |
| Subtask classifier | rl_token | 11 subtask classes | accuracy **19.9%** | chance: 9.1% | **2.2x** |

**Interpretation:**
- The RL token adds 6% MSE improvement over state-only for action prediction, confirming it contributes beyond raw proprioception.
- State information is linearly decodable from the RL token (29% below random baseline). Val loss decreases monotonically over 40 epochs without divergence.
- Subtask classification is 2.2× above chance, but train loss drops to 0.03 while val loss diverges (2.0 → 5.3) — the probe is overfitting, likely because 200 episodes split across 11 subtask classes leaves too few examples per class to learn from.
- **Verdict: moderate pass** — the RL token carries sufficient information for Stage 3 critic training.

See `evals/2026-03-30_probe_suite/metrics.json` for full per-epoch training histories.

## Repo Structure

```
assets/                                # Norm stats plus deterministic episode split metadata
checkpoints/9999/params/               # Model weights (params only)
evals/2026-03-27_rl_token_eval/        # Cosine analysis, reconstruction ablation JSONs, plots
evals/2026-03-30_probe_suite/          # Probe suite metrics (action, linear, subtask probes)
README.md                              # This file
TRAINING_LOG.md                        # Training log
```
