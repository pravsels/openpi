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

Evaluation artifacts for the frozen `9999` checkpoint are available in `evals/2026-03-27_rl_token_eval/`.

This evaluation depends on two artifacts together:
- the frozen base pi0.5 VLA: `pravsels/pi05-build-block-tower-baseline`
- the Stage 1 RLT encoder-decoder checkpoint in this repo: step `9999`

### Cosine Similarity (ID vs OOD)

- Raw cosine similarity does **not** cleanly separate ID (`build_block_tower`) from OOD (`drop_footbag_into_dice_tower`) episodes.
- Mean-pooled ID-vs-OOD cosine similarity is `0.9941`, indicating weak task separation in embedding space.
- The most likely failure mode is not a dead token, but a token dominated by a large shared component with useful information compressed into smaller residual directions.

See `evals/2026-03-27_rl_token_eval/eval_log.md` for the full interpretation and the accompanying JSON/plot artifacts.

### Reconstruction Ablation (step 5k vs 10k)

Tests whether the RL token carries meaningful information by comparing decoder reconstruction loss under three conditions: real token, zero vector, and shuffled (batch-neighbour's) token.

| Metric | Step 5,000 | Step 9,999 | Change |
|--------|-----------|-----------|--------|
| Real recon loss | 365.4 | 226.2 | −38% (better reconstruction) |
| Zero gap | 484.8 | 812.2 | +68% (token carries more information) |
| Shuffle gap | 36.0 | 90.1 | +150% (more per-example discrimination) |
| Pairwise cosine | 0.990 | 0.970 | −0.020 (tokens differentiating more) |

All metrics improve monotonically from step 5k to 10k, confirming the RL token is learning genuine information rather than collapsing. The modest shuffle gap is expected for a single-task dataset (100 episodes, same prompt) — tokens are legitimately similar across same-task observations.

See `evals/2026-03-27_rl_token_eval/recon_ablation_progression.md` for full analysis and per-batch breakdowns.

## Repo Structure

```
assets/                                # Norm stats plus deterministic episode split metadata
checkpoints/9999/params/               # Model weights (params only)
evals/2026-03-27_rl_token_eval/        # Evaluation artifacts (cosine analysis, ablation JSONs, plots)
README.md                              # This file
TRAINING_LOG.md                        # Training log
```
