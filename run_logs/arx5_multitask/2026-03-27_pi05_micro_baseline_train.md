# PI0.5 ARX5 micro baseline — 14-dataset mix (submitted)

## Mode
- run_type: experiment
- objective: Fine-tune PI0.5 on the micro training mix with baseline valid indices (human-controlled + successful episodes); compare later to `pi05_arx5_multitask_micro_advantaged`.

## Config
- script: `slurm/train_pi05_arx5_multitask_micro_baseline_slurm.sh`
- config: `pi05_arx5_multitask_micro_baseline` (`src/openpi/training/config.py`)
- exp_name: `micro_baseline_v1` (edit in script if you change it)
- dataset: `training_mix_micro.json` — 14 `villekuosmanen/*` LeRobot repos
- base weights: `weights/pi05_base/params` (bound on cluster to `${OPENPI_DATA_HOME}/weights`)
- key settings:
  - `Pi0Config(pi05=True, action_horizon=50)`
  - batch_size: 36
  - lr: cosine, warmup 1k, peak 5e-5, decay_steps 100k, decay_lr 5e-5
  - optimizer: AdamW, clip_gradient_norm 1.0
  - ema_decay: 0.999
  - num_train_steps: 30_000
  - use_delta_actions: True (delta joints, absolute grippers)
  - wandb: enabled (offline in container; sync from scratch cache if needed)
- precomputed assets (rsync into `.../checkpoints/pi05_arx5_multitask_micro_baseline/micro_baseline_v1/assets/` before `sbatch`):
  - `training_mix_micro.json`
  - `norm_stats.json`
  - `valid_indices.txt`

## Job
- job_id: 3396391
- submitted: 2026-03-27T14:50:00Z
- start_human: Friday, Mar 27th, 2026
- node: nid010735
- end:
- end_human:
- runtime:

## Status
- 2026-03-27 14:50 — submitted and running on nid010735

## Results
- *(fill on completion: final step, loss bracket, checkpoint path)*

## W&B
- local: *(under `${WANDB_DIR}` / scratch `.cache/wandb` as configured in script)*
- synced:
- notes:

## Next
- After completion: optional eval + checkpoint promotion; run `pi05_arx5_multitask_micro_advantaged` with same mix if comparing valid-index strategies.
- Update this file with `job_id`, timelines, and qualitative W&B notes after you submit and monitor.
