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
- submitted/start: `2026-03-29T14:15:10+00:00`
- start_human: Sunday, Mar 29th, 2026
- end: `2026-03-29T19:07:13+00:00`
- end_human: Sunday, Mar 29th, 2026
- runtime: `04:52:03`
- node: nid010769

## Status
- 2026-03-29 14:15 — submitted and running on nid010769
- 2026-03-29 19:07 — completed, exit code 0, step 29999

## Results
- final step: 29999
- start_train_loss: 0.1770 (step 0)
- loss@5k: 0.0215
- loss@10k: 0.0146
- loss@15k: 0.0118
- loss@20k: 0.0101
- loss@25k: 0.0089
- end_train_loss: 0.0080 (step 29900)
- loss_one_liner: Steep drop from 0.18 to ~0.02 in the first 5k, then steady decline to 0.008 by 30k; lower final loss than baseline (0.0080 vs 0.0107).
- checkpoint: `/scratch/u6cr/pravsels.u6cr/openpi/checkpoints/pi05_arx5_multitask_micro_advantaged/micro_advantaged_v1/29999`
- checkpoints kept: 25000, 29999 (pruned 5k, 10k, 15k, 20k)

## W&B
- local: `/scratch/u6cr/pravsels.u6cr/openpi/wandb/offline-run-20260329_141552-jik4rmpl`
- synced: https://wandb.ai/pravsels/arx5_multitask/runs/jik4rmpl

## HuggingFace
- repo: https://huggingface.co/pravsels/pi05-arx5-multitask-micro-advantaged
- uploaded checkpoints: 25k, 29999 (params only)
- includes: assets (norm stats, valid_indices.txt, training_mix_micro.json), README.md, TRAINING_LOG.md

## Next
- Compare loss curves and eval metrics against `pi05_arx5_multitask_micro_baseline`.
