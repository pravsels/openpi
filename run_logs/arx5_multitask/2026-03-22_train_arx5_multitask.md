# ARX5 Multitask v1 — PI0.5 foundation fine-tune on 186 mixed datasets

## Mode
- run_type: experiment
- objective: Fine-tune PI0.5 on 186 ARX5 single-arm + bimanual datasets with subtask descriptions as prompts, absolute actions, and per-dim loss masking for mixed 7/14-dim action spaces.

## Config
- script: `slurm/train_arx5_multitask_slurm.sh`
- config: `pi05_arx5_multitask_v1` (in `src/openpi/training/config.py`)
- exp_name: `arx5_abs_v1`
- dataset: `training_mix_v1.json` — 186 `villekuosmanen/*` LeRobot repos
- base weights: `weights/pi05_base/params` (PI0.5 pretrained)
- key settings:
  - model: `Pi0Config(pi05=True, action_horizon=50)`
  - action_dim: 14 (bimanual), single-arm padded with `action_dim_mask`
  - batch_size: 36
  - lr: cosine decay, warmup 10k steps, peak 5e-5, decay to 5e-5 over 1M steps
  - optimizer: AdamW, gradient clip norm 1.0
  - ema_decay: 0.999
  - num_train_steps: 100,000
  - use_delta_actions: False (absolute actions)
  - wandb: enabled (offline mode)
- precomputed assets at `/scratch/u6cr/pravsels.u6cr/openpi/checkpoints/pi05_arx5_multitask_v1/arx5_abs_v1/assets/`:
  - `norm_stats.json` — 14-dim per-dimension quantile stats
  - `valid_indices.txt` — ~1.4M filtered frame indices
  - `training_mix_v1.json` — dataset list

## Job
- job_id: 3283403
- submitted: 2026-03-22
- start_human: Sunday, Mar 22nd, 2026
- node: nid010979
- end: 2026-03-24 ~10:00 UTC (killed by walltime)
- end_human: Tuesday, Mar 24th, 2026
- runtime: 24:00:00 (walltime limit)

## Status
- 2026-03-22 — 3283367 failed (one dataset still private); made public, resubmitted as 3283403
- 2026-03-23 09:59 — job started on nid010979, dataset loading + JAX init
- 2026-03-23 10:06 — training loop started, ~1.7 it/s
- 2026-03-23 10:27 — reached step 2000, then deadlocked waiting for checkpoint finalization at step 1000. Scratch was full (5TB/5TB project quota). Checkpoint I/O couldn't complete, `.__lock` files never released, main thread blocked on `wait_until_finished`. Training hung for remaining ~23.5 hours.
- 2026-03-24 ~10:00 — killed by Slurm walltime limit. No usable checkpoints.
- 2026-03-24 — cleaned up corrupt checkpoint, cleared `wandb_id.txt`. Scratch cleaned from 4.4TB to 1.3TB.

## Job (resubmit)
- job_id: 3312454
- submitted: 2026-03-24
- start_human: Tuesday, Mar 24th, 2026
- worktree: `openpi_ville_subtask` (feat/ville_subtask @ 5e007f2)
- fresh start from base weights (no checkpoint to resume from)

## Results
- final step reached: 2000 (of 100,000)
- start_train_loss: 0.1667 (step 0)
- end_train_loss: 0.0252 (step 2000)
- loss_one_liner: Loss dropped healthily from 0.167 to 0.025 in 2000 steps before checkpoint deadlock killed the run.
- checkpoint: none — `1000.orbax-checkpoint-tmp-0` is corrupt (incomplete write due to full scratch)

## W&B
- local: `wandb/offline-run-20260322_213644-qabysa98`
- synced:
- notes: only ~20 min of training data captured before hang

## Next
- Monitor 3312454 for healthy training progress
- If walltime-interrupted, resume with same script (`--resume` flag is set)
- Future experiment: enable delta actions (`use_delta_actions=True` with `DeltaActionsFromState`) and recompute norm stats
