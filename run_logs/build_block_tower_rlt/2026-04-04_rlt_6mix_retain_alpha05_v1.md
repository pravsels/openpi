# build_block_tower RLT Stage 1 — 6mix backbone run

## Mode
- run_type: experiment
- objective: train the RLT encoder-decoder on top of the published 6mix build-block-tower baseline so the frozen VLA backbone and the RLT training mixture stay aligned

## Config
- script: `slurm/train_build_block_tower_rlt_slurm.sh`
- config: `pi05_rlt_build_block_tower_6mix` (in `src/openpi/training/config.py`)
- model: `Pi0RLConfig` (`src/openpi/models/pi0_rl.py`)
- dataset: 6 HuggingFace datasets (`villekuosmanen/build_block_tower` plus `dAgger_build_block_tower_1.0.0` through `1.4.0`)
- key settings: VLA frozen (`rl_vla_loss_weight=0.0`), encoder-decoder only, batch_size `36`, lr `5e-5` cosine (`1k` warmup), `num_train_steps=20_000`, episode-level `90/10` train/val split
- VLA backbone: published 6mix baseline checkpoint `49999` from `pravsels/pi05-build-block-tower-6mix`
- worktree: `/home/u6cr/pravsels.u6cr/openpi_rlt_block_tower` (branch `task/rlt_block_tower`)

## Job (failed — `3617244`)
- job_id: 3617244
- submitted: `2026-04-04`
- start: `2026-04-04T16:54:42+00:00`
- end: `2026-04-04T16:54:48+00:00`
- runtime: 00:00:06
- node: nid010548
- failure: `huggingface-cli: command not found` — the script's download fallback (line 55) runs outside the Apptainer container where `huggingface-cli` isn't installed. The baseline checkpoint path was `pi05_build_block_tower_baseline_6mix` but the checkpoint on scratch was under the original name `pi05_build_block_tower_baseline` (before the 6mix rename).

## Job (resubmit — `3619866` — success)
- job_id: 3619866
- submitted: `2026-04-04`
- start: `2026-04-05T05:18:09+00:00`
- start_human: Saturday, Apr 5th, 2026
- end: `2026-04-05T12:37:06+00:00`
- end_human: Saturday, Apr 5th, 2026
- runtime: 07:18:57
- node: nid010264

## Status
- 2026-04-04 — updated `pi05_rlt_build_block_tower_6mix` and `slurm/train_build_block_tower_rlt_slurm.sh` to load the published 6mix baseline checkpoint from `pravsels/pi05-build-block-tower-6mix`, use scratch path `checkpoints/pi05_build_block_tower_baseline_6mix/baseline/49999/params`, and write new outputs under `rlt_6mix_v1`
- 2026-04-04 — first submission (`3617227`) was cancelled after spotting that the RLT config still pointed at only the single base `build_block_tower` dataset instead of the full 6mix training mixture
- 2026-04-04 — added a regression test asserting `pi05_rlt_build_block_tower_6mix` uses the same dataset mix as `pi05_build_block_tower_baseline`, then fixed the config to include all six datasets
- 2026-04-04 — committed the dataset fix as `08cb538` (`fix block tower rlt 6mix dataset mix`), pushed `task/rlt_block_tower`, fast-forwarded `/home/u6cr/pravsels.u6cr/openpi_rlt_block_tower`, and resubmitted as job `3617244`
- 2026-04-04 — job `3617244` failed after 6s: `huggingface-cli: command not found`. The baseline checkpoint was on scratch under `pi05_build_block_tower_baseline` (the pre-rename name), not `pi05_build_block_tower_baseline_6mix`. User copied checkpoint to the expected path and symlinked `baseline/assets` → `baseline/49999/assets`. Resubmitted as job `3619866`
- 2026-04-05 — job `3619866` completed successfully in 07:18:57 on nid010264. All 20k steps trained, checkpoints saved at 5000/10000/15000/19999.

## Results
- final train loss: 208.4 (step 19900)
- final val loss: 302.1 (step 19000)
- checkpoints: 5000, 10000, 15000, 19999
- checkpoint dir: `/scratch/u6cr/pravsels.u6cr/openpi/checkpoints/pi05_rlt_build_block_tower_6mix/rlt_6mix_v1/`
- loss decreased steadily throughout training; no instability or divergence

## W&B
- local: `/scratch/u6cr/pravsels.u6cr/openpi/wandb/offline-run-20260405_051853-xanf5muf`
- synced: https://wandb.ai/pravsels/openpi-rlt-block-tower/runs/xanf5muf
- notes: synced to new project `openpi-rlt-block-tower`

## HuggingFace
- repo: https://huggingface.co/pravsels/pi05-build-block-tower-rlt-6mix
- uploaded: checkpoints 5000, 10000, 15000, 19999 (params only, no train_state)
- includes: README.md, TRAINING_LOG.md, assets (norm stats, valid indices), checkpoint hashes

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
- exp_name: `rlt_6mix_retain_alpha05_v1`
- backbone: `retain/step_49999/alpha_0.5` (from `pravsels/pi05-build-block-tower-6mix`)
- checkpoint_dir: `/scratch/u6cr/pravsels.u6cr/openpi/checkpoints/pi05_rlt_build_block_tower_6mix/rlt_6mix_retain_alpha05_v1`
- status: pending

## Next
- monitor job `3820691`
- run reconstruction ablation on the 6mix RLT checkpoints (compare against single-dataset RLT results)
- decide which checkpoints to publish to HuggingFace
