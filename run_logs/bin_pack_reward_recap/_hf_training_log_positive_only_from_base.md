# Training Log — Reward Recap Positive Only (from Base)

## Mode
- run_type: experiment
- objective: test whether positive-only advantage conditioning works when training from pi0.5 base weights (no task-specific pre-training)

## Config
- config: `pi05_bin_pack_coffee_capsules_reward_recap_positive_only_from_base`
- dataset: 9 datasets (bin_pick_pack + 8 dAgger rounds) on HuggingFace
- key settings: delta actions, 100k steps, batch_size=36, lr=5e-5 cosine, from pi0.5 base weights

## Training Dynamics

| Step | Loss | Grad Norm |
|------|------|-----------|
| 1,100 | 0.0425 | 0.3367 |
| 25,000 | 0.0097 | 0.0605 |
| 50,000 | 0.0066 | 0.0509 |

- loss_one_liner: Loss dropped from 0.043 to 0.005 steadily; higher initial loss than task-pretrained variant as expected, but converged to similar range. No plateau.

## W&B
- synced: https://wandb.ai/pravsels/recap_plain/runs/l4kfxxqe

## Status
- Started: Tuesday, Mar 24th, 2026
- Training in progress
