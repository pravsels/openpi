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

## Repo Structure

```
assets/                      # Norm stats plus deterministic episode split metadata
checkpoints/9999/params/     # Model weights (params only)
README.md                    # This file
TRAINING_LOG.md              # Training log
```
