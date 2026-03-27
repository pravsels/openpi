# Training Log — Reward Recap Positive Only

## Mode
- run_type: experiment
- objective: test whether positive-only advantage conditioning improves bin-pack policy when fine-tuning from a task-trained checkpoint

## Config
- config: `pi05_bin_pack_coffee_capsules_reward_recap_positive_only`
- dataset: 9 datasets (bin_pick_pack + 8 dAgger rounds) on HuggingFace
- key settings: delta actions, 100k steps, batch_size=36, lr=5e-5 cosine, resume from task-trained bin-pack checkpoint (step 29999)

## Training Dynamics

| Step | Loss | Grad Norm |
|------|------|-----------|
| 3,100 | 0.0148 | 0.1025 |
| 25,000 | 0.0074 | 0.0511 |
| 50,000 | 0.0058 | 0.0486 |

- loss_one_liner: Loss dropped steadily from 0.015 to 0.005 with no sign of plateau; healthy training throughout.

## W&B
- synced: https://wandb.ai/pravsels/recap_plain/runs/9cfn5pz0

## Status
- Started: Tuesday, Mar 24th, 2026
- Training in progress
