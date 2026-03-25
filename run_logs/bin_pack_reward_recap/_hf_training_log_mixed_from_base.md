# Training Log — Reward Recap Mixed (from Base)

## Mode
- run_type: experiment
- objective: test whether mixed positive/negative advantage conditioning works when training from pi0.5 base weights (no task-specific pre-training)

## Config
- config: `pi05_bin_pack_coffee_capsules_reward_recap_mixed_from_base`
- dataset: 9 datasets (bin_pick_pack + 8 dAgger rounds) on HuggingFace
- key settings: delta actions, 100k steps, batch_size=36, lr=5e-5 cosine, from pi0.5 base weights

## Training Dynamics

| Step | Loss | Grad Norm |
|------|------|-----------|
| 0 | 0.3473 | 3.1816 |
| 25,000 | 0.0125 | 0.0705 |
| 50,000 | ~0.0075 | ~0.05 |
| 74,000 | ~0.0070 | ~0.05 |

- loss_one_liner: High initial loss (0.35) from base weights + negative demonstrations. Dropped rapidly to 0.012 by 25k, continued converging to ~0.007 by 74k with stable grad norms.

## Checkpoint Hashes (params, deterministic)

Verify with `cd checkpoints/<step> && find params -type f | sort | xargs cat | sha256sum`.

| Step | SHA-256 |
|------|---------|
| 25,000 | `9e6d903b70a0159d6fb9979570556b031650f2e733e9b8a30c1d17b08f3307c2` |
| 50,000 | `f347d098e046f63ef65aa9c0c7a5614e0735667f1d03a5f8fb893e43698079c9` |
| 74,000 | `4fd1de8b341df95595b7691443e524638c80a0f1eb58c09d717a33741482d70e` |

## W&B
- synced: https://wandb.ai/pravsels/recap_plain/runs/kurb306v

## Status
- Started: Tuesday, Mar 24th, 2026
- Completed: Wednesday, Mar 25th, 2026 (stopped at step 74,200)
- Multiple resubmissions due to checkpoint finalization hangs
- Runtime: ~24h total across 4 submissions
