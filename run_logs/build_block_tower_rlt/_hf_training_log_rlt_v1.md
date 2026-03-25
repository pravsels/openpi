# Training Log — Build Block Tower RLT Stage 1

## Mode
- run_type: experiment
- objective: train RLT encoder-decoder on top of frozen baseline VLA (55k checkpoint)

## Config
- config: `pi05_rl_token_build_block_tower`
- model: `Pi0RLConfig` (rl_vla_loss_weight=0.0, 2-layer encoder-decoder, 8 heads, dim=2048)
- dataset: `villekuosmanen/build_block_tower` (200 episodes, LeRobot v2.1)
- key settings: VLA frozen, encoder-decoder only, 10k steps, batch_size=36, lr=5e-5 cosine (1k warmup)
- VLA backbone: baseline 55k checkpoint (loss 0.007)

## Training Dynamics

| Step | Loss | Grad Norm |
|------|------|-----------|
| 0 | 9873.5 | 35762.5 |
| 5,000 | ~273 | ~1700 |
| 9,900 | 218.0 | 1330.4 |

- loss_one_liner: Reconstruction loss dropped from ~9874 to ~218 over 10k steps with stable convergence and no sign of plateau.

## Checkpoint Hashes (params, deterministic)

Verify with `cd checkpoints/<step> && find params -type f | sort | xargs cat | sha256sum`.

| Step | SHA-256 |
|------|---------|
| 9,999 | `214f3473fba0339779276528ff618b3a88cd7df5bdb4c1560bf0c13459fe3454` |

## Status
- Started: Wednesday, Mar 25th, 2026
- Completed: Wednesday, Mar 25th, 2026
- Runtime: ~1 hour
