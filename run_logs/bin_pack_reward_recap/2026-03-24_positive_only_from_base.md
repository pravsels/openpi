# reward_recap positive_only — from pi05 base weights

## Mode
- run_type: experiment
- objective: test whether positive-only advantage conditioning works when training from pi05 base weights (no task-specific pre-training)

## Config
- script: `slurm/train_bin_pack_reward_recap_slurm.sh positive_only_from_base`
- config: `pi05_bin_pack_coffee_capsules_reward_recap_positive_only_from_base` (in `src/openpi/training/config.py`)
- dataset: 9 datasets (bin_pick_pack + 8 dAgger rounds) on HuggingFace
- key settings: delta actions, 100k steps, batch_size=36, lr=5e-5 cosine, weights from `weights/pi05_base/params`

## Job
- job_id: 3303415
- submitted/start: `2026-03-24T`
- start_human: Monday, Mar 24th, 2026
- end:
- end_human:
- runtime:
- node:

## Status

## Results

## W&B
- local:
- synced:
- notes:

## Next
