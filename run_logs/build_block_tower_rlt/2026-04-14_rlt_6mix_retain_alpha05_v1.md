# build_block_tower RLT on retain/step_49999/alpha_0.5 checkpoint

## Mode
- run_type: experiment
- objective: train fresh RLT encoder-decoder on the retain/step_49999/alpha_0.5 checkpoint from the 6mix baseline

## Config
- script: `slurm/train_build_block_tower_rlt_slurm.sh`
- config: `pi05_rlt_build_block_tower_6mix` (in `src/openpi/training/config.py`)
- model: `Pi0RLConfig` (`src/openpi/models/pi0_rl.py`)
- dataset: 6 HuggingFace datasets (`villekuosmanen/build_block_tower` plus `dAgger_build_block_tower_1.0.0` through `1.4.0`)
- key settings: VLA frozen (`rl_vla_loss_weight=0.0`), encoder-decoder only, batch_size `36`, lr `5e-5` cosine (`1k` warmup), `num_train_steps=20_000`, episode-level `90/10` train/val split
- VLA backbone: `retain/step_49999/alpha_0.5` checkpoint from `pravsels/pi05-build-block-tower-6mix`
- worktree: `/home/u6cr/pravsels.u6cr/openpi_rlt` (branch `task/rlt_block_tower`)

## Job (failed — `3816843` — retain/alpha_0.5 backbone)
- job_id: 3816843
- submitted: 2026-04-14
- start: `2026-04-14T21:24:52+00:00`
- end: `2026-04-14T21:25:21+00:00`
- runtime: 00:00:29
- node: nid010984
- exp_name: `rlt_6mix_retain_alpha05_v1`
- backbone: `retain/step_49999/alpha_0.5` (from `pravsels/pi05-build-block-tower-6mix`)
- failure: `scripts/train.py` not found — sbatch script `repo_dir` pointed to `~/openpi_rlt_block_tower` but the repo was cloned as `~/openpi_rlt`. Renamed repo dir to match.

## Job (resubmit — `3820691` — retain/alpha_0.5 backbone)
- job_id: 3820691
- submitted: 2026-04-14
- start: `2026-04-15T07:13:17`
- end: `2026-04-15T10:39:32`
- runtime: 03:26:15
- node: nid010761
- exp_name: `rlt_6mix_retain_alpha05_v1`
- backbone: `retain/step_49999/alpha_0.5` (from `pravsels/pi05-build-block-tower-6mix`)
- checkpoint_dir: `/scratch/u6cr/pravsels.u6cr/openpi/checkpoints/pi05_rlt_build_block_tower_6mix/rlt_6mix_retain_alpha05_v1`
- status: failed
- failure: data loader hit a non-numeric field when converting a batch to JAX arrays (`TypeError: Dtype <U7 is not a valid JAX array type`)

## Status
- 2026-04-14 — modified `slurm/train_build_block_tower_rlt_slurm.sh` to use `retain/step_49999/alpha_0.5` checkpoint with new experiment name `rlt_6mix_retain_alpha05_v1`
- 2026-04-14 — updated `pi05_rlt_build_block_tower_6mix` config weight_loader to point to `checkpoints/pi05_build_block_tower_baseline_6mix/retain/step_49999/alpha_0.5/params`
- 2026-04-14 — submitted to Slurm as job `3816843`; currently `PD` on `workq` with reason `(Priority)`
- 2026-04-15 — resubmitted as `3820691`; entered `R` at `07:13:17` on `nid010761`
- 2026-04-15 — job `3820691` failed at `10:39:32` with `TypeError: Dtype <U7 is not a valid JAX array type` during batch conversion in `data_loader.py`
- 2026-04-15 — root cause: `control_mode` string field passed through by `_copy_passthrough_metadata` but never consumed (RLT config doesn't use `SetAdvantageLabelFromControlMode`). Fixed in `f09cb60` by popping `control_mode` in `TokenizePrompt` and `TokenizeHighPrompt`. Verified fix inside container — all sample fields numeric.
- 2026-04-15 — resubmitted as `3829868`
- 2026-04-15 — job `3829868` completed successfully in 05:42:58 on `nid010961`. All 20k steps trained, checkpoints saved at 5000/10000/15000/19999.

## Job (resubmit — `3829868` — retain/alpha_0.5 backbone — success)
- job_id: 3829868
- submitted: 2026-04-15
- start: `2026-04-15T13:58:49`
- start_human: Tuesday, Apr 15th, 2026
- end: `2026-04-15T19:41:53`
- end_human: Tuesday, Apr 15th, 2026
- runtime: 05:42:58
- node: nid010961

## Results
- final step: 19999
- final train_loss: 356.3 (step 19900)
- final val_loss: 464.8 (step 19000)
- loss_one_liner: Train loss decreased steadily from early steps down to ~356; val loss higher at ~465 but no sign of divergence.
- checkpoints: 5000, 10000, 15000, 19999
- checkpoint_dir: `/home/u6cr/pravsels.u6cr/openpi_rlt_block_tower/checkpoints/pi05_rlt_build_block_tower_6mix/rlt_6mix_retain_alpha05_v1`

## W&B
- local: `/scratch/u6cr/pravsels.u6cr/openpi/wandb/offline-run-20260415_135934-g5myo76p`
- synced: https://wandb.ai/pravsels/pi05-build-block-tower-rlt-6mix-retain-alpha05/runs/g5myo76p
- notes:

## HuggingFace
- repo: https://huggingface.co/pravsels/pi05-build-block-tower-rlt-6mix-retain-alpha05
- uploaded: checkpoint 19999 (params only, no train_state)
- includes: README.md, TRAINING_LOG.md, assets (norm stats, valid indices, episode split), checkpoint hash

## Next
- sync W&B run and review training curves
- run reconstruction ablation on the 6mix RLT checkpoints (compare against single-dataset RLT results)
- decide which checkpoints to publish to HuggingFace
