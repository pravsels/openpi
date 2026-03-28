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
- submitted/start: `2026-03-27T14:53:03+00:00`
- start_human: Friday, Mar 27th, 2026
- end: `2026-03-27T19:47:49+00:00`
- end_human: Friday, Mar 27th, 2026
- runtime: `04:54:46`
- node: nid010735

## Status
- 2026-03-27 14:53 — submitted and running on nid010735
- 2026-03-27 19:47 — completed, exit code 0, step 29999

## Results
- final step: 29999
- start_train_loss: 0.1664 (step 0)
- loss@5k: 0.0274
- loss@10k: 0.0197
- loss@15k: 0.0159
- loss@20k: 0.0133
- loss@25k: 0.0119
- end_train_loss: 0.0107 (step 29900)
- loss_one_liner: Steep drop from 0.17 to ~0.03 in the first 5k, then steady decline to 0.011 by 30k; no plateau or overfitting.
- checkpoint: `/scratch/u6cr/pravsels.u6cr/openpi/checkpoints/pi05_arx5_multitask_micro_baseline/micro_baseline_v1/29999`
- checkpoints kept: 25000, 29999 (pruned 5k, 10k, 15k, 20k)

## W&B
- local: `/scratch/u6cr/pravsels.u6cr/openpi/wandb/offline-run-20260327_145436-gtk5f6zw`
- synced: https://wandb.ai/pravsels/arx5_multitask/runs/gtk5f6zw
- notes:

## HuggingFace
- repo: https://huggingface.co/pravsels/pi05-arx5-multitask-micro-baseline
- uploaded checkpoints: 25k, 29999 (params only)
- includes: assets (norm stats, valid_indices.txt, training_mix_micro.json), README.md, TRAINING_LOG.md

## Next
- Run `pi05_arx5_multitask_micro_advantaged` with same mix if comparing valid-index strategies.
