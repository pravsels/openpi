# Training Log — Reward Recap Mixed

## Mode
- run_type: experiment
- objective: test whether mixed positive/negative advantage conditioning improves bin-pack policy when fine-tuning from a task-trained checkpoint

## Config
- config: `pi05_bin_pack_coffee_capsules_reward_recap_mixed`
- dataset: 9 datasets (bin_pick_pack + 8 dAgger rounds) on HuggingFace
- key settings: delta actions, 100k steps, batch_size=36, lr=5e-5 cosine, resume from task-trained bin-pack checkpoint (step 29999)

## Training Dynamics

| Step | Loss | Grad Norm |
|------|------|-----------|
| 0 | 0.5005 | 1.2731 |
| 25,000 | 0.0098 | 0.0633 |
| 50,000 | 0.0075 | 0.0524 |

- loss_one_liner: High initial loss (0.50) from introduction of negative demonstrations. Dropped rapidly in first 5k steps, then steadily decreased to 0.006 range. No plateau.

## W&B
- synced: https://wandb.ai/pravsels/recap_plain/runs/vvd1y2sk

## Status
- Started: Tuesday, Mar 24th, 2026
- Training in progress
