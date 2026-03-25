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

- loss_one_liner: High initial loss (0.35) with very high grad norm (3.2) from base weights + negative demonstrations. Dropped rapidly, reached 0.011 by 31k steps and still decreasing. Training behind the other three runs due to initial checkpoint hang requiring resubmission.

## W&B
- synced: https://wandb.ai/pravsels/recap_plain/runs/kurb306v

## Status
- Started: Tuesday, Mar 24th, 2026
- First submission hung during checkpoint finalization at step 1000; cancelled and resubmitted fresh.
- Training in progress (resubmitted run)
