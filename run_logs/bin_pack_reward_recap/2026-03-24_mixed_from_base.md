# reward_recap mixed — from pi05 base weights

## Mode
- run_type: experiment
- objective: test whether mixed positive/negative advantage conditioning works when training from pi05 base weights (no task-specific pre-training)

## Config
- script: `slurm/train_bin_pack_reward_recap_slurm.sh mixed_from_base`
- config: `pi05_bin_pack_coffee_capsules_reward_recap_mixed_from_base` (in `src/openpi/training/config.py`)
- dataset: 9 datasets (bin_pick_pack + 8 dAgger rounds) on HuggingFace
- key settings: delta actions, 100k steps, batch_size=36, lr=5e-5 cosine, weights from `weights/pi05_base/params`

## Job
- job_id: 3310330
- submitted/start: `2026-03-24T`
- start_human: Monday, Mar 24th, 2026
- end: cancelled
- runtime: ~4h
- node: nid010329

## Status
- 2026-03-24 12:30 — running, step 2000, loss 0.0362, rate 1.7 it/s, ~16h remaining. Long data loading phase (~2h50m before training started). Step 0 loss 0.3473, already down to 0.036 by step 2k.
- 2026-03-24 13:40 — checkpoint finalize hung at step 1000 (41GB tmp written, rename never completed). Training blocked at step 2000 waiting for finalize thread. Cancelled, cleaned up tmp checkpoint + wandb_id.txt.

## Job (resubmit)
- job_id: 3318403
- submitted: `2026-03-24T`
- start_human: Monday, Mar 24th, 2026
- end:
- runtime:
- node:
- notes: fresh start, assets/norm_stats preserved

## Results

## W&B
- local:
- synced:
- notes:

## Next
