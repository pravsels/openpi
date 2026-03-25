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
- loss@25k: 0.0114 (from resubmit 1)
- loss@29k: ~0.0114

## W&B
- local: `wandb/offline-run-20260324_140237-kurb306v`
- synced: https://wandb.ai/pravsels/recap_plain/runs/kurb306v
- notes: synced data covers resubmit 1 (steps 0–31k). Resubmit 2 will create a new offline run.

## HuggingFace
- repo: https://huggingface.co/pravsels/pi05-bin-pack-reward-recap-mixed-from-base
- uploaded checkpoints: 25k (params only)
- includes: README, TRAINING_LOG, assets

## Next
- Upload 50k + final checkpoint once resubmit 3 completes
- Sync new wandb run
- Watch the next checkpoint save/finalize cycle before leaving unattended
