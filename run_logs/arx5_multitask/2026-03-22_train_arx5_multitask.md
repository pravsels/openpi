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
- node:
- end:
- runtime:

## Status
- 2026-03-22 — 3283367 failed (one dataset still private); made public, resubmitted as 3283403

## Results

## W&B
- local: `wandb/offline-run-20260322_213644-qabysa98`
- synced:
- notes:

## Next
- Monitor initial loss curve to confirm training is progressing
- If walltime-interrupted, resume with same script (`--resume` flag is set)
- Future experiment: enable delta actions (`use_delta_actions=True` with `DeltaActionsFromState`) and recompute norm stats
