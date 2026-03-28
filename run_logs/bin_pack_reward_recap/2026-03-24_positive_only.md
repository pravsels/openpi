# reward_recap positive_only — resume from 1_dataset checkpoint

## Mode
- run_type: experiment
- objective: test whether positive-only advantage conditioning improves bin-pack policy when fine-tuning from a task-trained checkpoint

## Config
- script: `slurm/train_bin_pack_reward_recap_slurm.sh positive_only`
- config: `pi05_bin_pack_coffee_capsules_reward_recap_positive_only` (in `src/openpi/training/config.py`)
- dataset: 9 datasets (bin_pick_pack + 8 dAgger rounds) on HuggingFace
- key settings: 17D action space (7D joints + 7D EEF with RPY→rot6d), delta actions (joints+xyz+gripper delta, rot6d absolute), per-timestep action normalization, rot6d identity normalization, 100k steps, batch_size=36, lr=5e-5 cosine, resume from `pi05_bin_pack_coffee_capsules_delta_single_dataset/1_dataset/29999/params`

## Job
- job_id: 3310629
- submitted/start: `2026-03-24T`
- start_human: Tuesday, Mar 24th, 2026
- end:
- end_human:
- runtime:
- node: nid011081

## Status
- 2026-03-24 12:30 — running, step 21000, loss 0.0079, rate 1.7 it/s, ~13h remaining. First logged step 3100 (loss 0.0148). Loss dropped steadily to ~0.008 range.

## Results

## W&B
- local: `wandb/offline-run-20260324_092932-9cfn5pz0`
- synced: https://wandb.ai/pravsels/recap_plain/runs/9cfn5pz0
- notes:

## HuggingFace
- repo: https://huggingface.co/pravsels/pi05-bin-pack-reward-recap-positive-only
- uploaded checkpoints: 25k, 50k (params only)
- includes: README, TRAINING_LOG, assets

## Next
