# Training Log — Build Block Tower RLT Stage 1

## Mode
- run_type: experiment
- objective: train RLT encoder-decoder on top of frozen baseline VLA (55k checkpoint)

## Config
- config: `pi05_rl_token_build_block_tower`
- model: `Pi0RLConfig` (rl_vla_loss_weight=0.0, 2-layer encoder-decoder, 8 heads, dim=2048)
- dataset: `villekuosmanen/build_block_tower` (200 episodes, LeRobot v2.1)
- key settings: VLA frozen, encoder-decoder only, 10k steps, batch_size=36, lr=5e-5 cosine (1k warmup), deterministic episode-level 90/10 train/val split
- VLA backbone: baseline 55k checkpoint (loss 0.007)

## Training Dynamics

| Step | Train Loss | Val Loss | Grad Norm |
|------|------------|----------|-----------|
| 0 | 9944.8115 | 9756.3584 | 36160.1641 |
| 1,000 | 584.5576 | 1456.2078 | 2977.0269 |
| 2,000 | 483.2471 | 660.4791 | 1786.0547 |
| 3,000 | 421.7671 | 538.0388 | 1792.1675 |
| 9,900 | 216.8683 | 286.5721 at step 9,000 | 1256.7623 |

- loss_one_liner: Train loss and held-out val loss both dropped strongly over the 10k-step run, with val remaining higher and noisier but still improving throughout.

## Checkpoint Hashes (params, deterministic)

Verify with `cd checkpoints/<step> && find params -type f | sort | xargs sha256sum | sha256sum`.

| Step | SHA-256 |
|------|---------|
| 9,999 | `4378fc1886f7eef6adab8a123ec491cde783c9aa94cd60a0b57757314862ed95` |

## W&B
- synced: https://wandb.ai/pravsels/openpi-rlt-block-tower/runs/oa2o0o0z
- notes: validation-aware rerun; train loss converged faster than val loss, but the held-out curve continued trending down through the final logged evaluation.

## Status
- Started: Thursday, Mar 26th, 2026
- Completed: Thursday, Mar 26th, 2026
- Runtime: 2:12:34
