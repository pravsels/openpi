# PI0.5 ARX5 micro advantaged — 14-dataset mix (submitted)

## Mode
- run_type: experiment
- objective: Fine-tune PI0.5 on the micro training mix with advantaged valid indices; compare to `pi05_arx5_multitask_micro_baseline`.

## Config
- script: `slurm/train_pi05_arx5_multitask_micro_advantaged_slurm.sh`
- config: `pi05_arx5_multitask_micro_advantaged` (`src/openpi/training/config.py`)
- exp_name: `micro_advantaged_v1`
- dataset: `training_mix_micro.json` — 14 `villekuosmanen/*` LeRobot repos
- base weights: `weights/pi05_base/params`
- key settings:
  - `Pi0Config(pi05=True, action_horizon=50)`
  - batch_size: 36
  - lr: cosine, warmup 1k, peak 5e-5, decay_steps 100k, decay_lr 5e-5
  - optimizer: AdamW, clip_gradient_norm 1.0
  - ema_decay: 0.999
  - num_train_steps: 30_000
  - use_delta_actions: True (delta joints, absolute grippers)
  - wandb: enabled (offline in container)
- precomputed assets (in `.../checkpoints/pi05_arx5_multitask_micro_advantaged/micro_advantaged_v1/assets/`):
  - `training_mix_micro.json`
  - `norm_stats.json`
  - `valid_indices.txt` (advantaged indices)

## Job
- job_id: 3440925
- submitted: 2026-03-29
- node: (pending)

## Status
- 2026-03-29 — submitted

## Results
- (pending)

## W&B
- (pending)

## Next
- Compare loss curves and eval metrics against `pi05_arx5_multitask_micro_baseline`.
