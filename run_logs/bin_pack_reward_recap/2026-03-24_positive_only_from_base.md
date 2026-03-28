# reward_recap positive_only — from pi05 base weights

## Mode
- run_type: experiment
- objective: test whether positive-only advantage conditioning works when training from pi05 base weights (no task-specific pre-training)

## Config
- script: `slurm/train_bin_pack_reward_recap_slurm.sh positive_only_from_base`
- config: `pi05_bin_pack_coffee_capsules_reward_recap_positive_only_from_base` (in `src/openpi/training/config.py`)
- dataset: 9 datasets (bin_pick_pack + 8 dAgger rounds) on HuggingFace
- key settings: 17D action space (7D joints + 7D EEF with RPY→rot6d), delta actions (joints+xyz+gripper delta, rot6d absolute), per-timestep action normalization, rot6d identity normalization, 100k steps, batch_size=36, lr=5e-5 cosine, weights from `weights/pi05_base/params`

## Job
- job_id: 3310631
- submitted/start: `2026-03-24T`
- start_human: Tuesday, Mar 24th, 2026
- end:
- end_human:
- runtime:
- node: nid011091

## Status
- 2026-03-24 12:30 — running, step 18800, loss 0.0119, rate ~1.7 it/s, ~13h remaining. First logged step 1100 (loss 0.0425). Loss dropping steadily to ~0.012.

## Results

## W&B
- local: `wandb/offline-run-20260324_092933-l4kfxxqe`
- synced: https://wandb.ai/pravsels/recap_plain/runs/l4kfxxqe
- notes:

## HuggingFace
- repo: https://huggingface.co/pravsels/pi05-bin-pack-reward-recap-positive-only-from-base
- uploaded checkpoints: 25k, 50k (params only)
- includes: README, TRAINING_LOG, assets

## Next
