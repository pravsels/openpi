# reward_recap mixed — from pi05 base weights

## Mode
- run_type: experiment
- objective: test whether mixed positive/negative advantage conditioning works when training from pi05 base weights (no task-specific pre-training)

## Config
- script: `slurm/train_bin_pack_reward_recap_slurm.sh mixed_from_base`
- config: `pi05_bin_pack_coffee_capsules_reward_recap_mixed_from_base` (in `src/openpi/training/config.py`)
- dataset: 9 datasets (bin_pick_pack + 8 dAgger rounds) on HuggingFace
- key settings: 17D action space (7D joints + 7D EEF with RPY→rot6d), delta actions (joints+xyz+gripper delta, rot6d absolute), per-timestep action normalization, rot6d identity normalization, 100k steps, batch_size=36, lr=5e-5 cosine, weights from `weights/pi05_base/params`

## Job
- job_id: 3310330
- submitted/start: `2026-03-24T`
- start_human: Tuesday, Mar 24th, 2026
- end: cancelled
- runtime: ~4h
- node: nid010329

## Status
- 2026-03-24 12:30 — running, step 2000, loss 0.0362, rate 1.7 it/s, ~16h remaining. Long data loading phase (~2h50m before training started). Step 0 loss 0.3473, already down to 0.036 by step 2k.
- 2026-03-24 13:40 — checkpoint finalize hung at step 1000 (41GB tmp written, rename never completed). Training blocked at step 2000 waiting for finalize thread. Cancelled, cleaned up tmp checkpoint + wandb_id.txt.
- 2026-03-25 10:03 — resubmit 2 stalled while waiting for checkpoint save finalize at step 35000; job remained RUNNING but logs stopped advancing.
- 2026-03-25 12:12 — cancelled job 3334251, deleted unfinished checkpoint tmp dir `35000.orbax-checkpoint-tmp-20`, and resubmitted.
- 2026-03-25 12:13 — new job 3337757 restored successfully from checkpoint `34000`; progress resumed (`34.0k/100k` onward).

## Job (resubmit 1)
- job_id: 3318403
- submitted: `2026-03-24T14:01:58+00:00`
- start_human: Tuesday, Mar 24th, 2026
- end: cancelled (disk full)
- runtime: ~19h (stuck for ~14h)
- node: nid010444
- notes: fresh start, assets/norm_stats preserved. Reached step 31k, then hung at checkpoint save (step 30k) due to scratch disk full (5.0T/5.0T). Partial 30k checkpoint removed.

## Job (resubmit 2)
- job_id: 3334251
- submitted: `2026-03-25T08:50+00:00`
- start_human: Wednesday, Mar 25th, 2026
- end: cancelled (checkpoint finalize hang at step 35k)
- runtime: ~3h19m
- node: nid011286
- notes: resumed from 34k and reached 36k before stalling on checkpoint finalize; unfinished tmp checkpoint removed before restart

## Job (resubmit 3)
- job_id: 3337757
- submitted: `2026-03-25T12:12+00:00`
- start_human: Wednesday, Mar 25th, 2026
- resumed from: `/scratch/u6cr/pravsels.u6cr/openpi/checkpoints/pi05_bin_pack_coffee_capsules_reward_recap_mixed_from_base/mixed_from_base/34000`
- node: nid011208
- notes: restart after deleting `35000.orbax-checkpoint-tmp-20`; checkpoint restore to step 34000 confirmed in logs

## Results
- loss@0: 0.3473
- loss@25k: 0.0125
- loss@50k: ~0.0075
- loss@74k: ~0.0070
- grad_norm@74k: ~0.05
- param_norm@74k: ~1886
- loss_one_liner: High initial loss (0.35) from base weights + negative demos. Dropped rapidly to 0.012 by 25k, converged to ~0.007 by 74k.

## W&B
- local (resubmit 1): `wandb/offline-run-20260324_140237-kurb306v`
- synced (resubmit 1, steps 0-31k): https://wandb.ai/pravsels/recap_plain/runs/kurb306v
- local (resubmit 3): `wandb/offline-run-20260325_121310-kurb306v`
- synced (resubmit 3, steps 34k-74k): https://wandb.ai/pravsels/recap_plain/runs/kurb306v

## HuggingFace
- repo: https://huggingface.co/pravsels/pi05-bin-pack-reward-recap-mixed-from-base
- uploaded checkpoints: 25k, 50k, 74k (params only)
- includes: README, TRAINING_LOG, assets
- checkpoint hashes (deterministic, `find params -type f | sort | xargs cat | sha256sum`):
  - 25k: `9e6d903b70a0159d6fb9979570556b031650f2e733e9b8a30c1d17b08f3307c2`
  - 50k: `f347d098e046f63ef65aa9c0c7a5614e0735667f1d03a5f8fb893e43698079c9`
  - 74k: `4fd1de8b341df95595b7691443e524638c80a0f1eb58c09d717a33741482d70e`

## Next
- Evaluate checkpoints on bin packing task
