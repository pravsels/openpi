# build_block_tower RLT joints-only on joints_only/49999 checkpoint

## Mode
- run_type: experiment
- objective: train fresh RLT encoder-decoder on the joints-only block-tower baseline checkpoint (`joints_only/49999`) with joints-only action supervision

## Config
- script: `slurm/train_build_block_tower_rlt_joints_only_slurm.sh`
- config: `pi05_rlt_build_block_tower_6mix_joints_only` (in `src/openpi/training/config.py`)
- model: `Pi0RLConfig` (`src/openpi/models/pi0_rl.py`)
- dataset: 6 HuggingFace datasets (`villekuosmanen/build_block_tower` plus `dAgger_build_block_tower_1.0.0` through `1.4.0`)
- key settings: VLA frozen (`rl_vla_loss_weight=0.0`), encoder-decoder only, joints_only=True, batch_size `36`, lr `5e-5` cosine (`1k` warmup), `num_train_steps=50_000`, episode-level `90/10` train/val split
- VLA backbone: `joints_only/49999` checkpoint from `pravsels/build_block_tower_baseline_6mix_joints_only`

## Job (failed — `4180356` — launch script issue)
- job_id: 4180356
- submitted/start: `2026-04-22T16:52:14+00:00`
- start_human: Wednesday, Apr 22nd, 2026
- end: `2026-04-22T16:52:20+00:00`
- end_human: Wednesday, Apr 22nd, 2026
- runtime: `00:00:05`
- node: nid010970
- exp_name: `rlt_6mix_joints_only_v1`
- failure: host `huggingface-cli` had a broken interpreter path (`/usr/bin/python` missing), so asset download failed before training startup

## Job (resubmit — `4180372` — failed, data-load hang + strace)
- job_id: 4180372
- submitted/start: `2026-04-22T16:55:13+00:00`
- start_human: Wednesday, Apr 22nd, 2026
- end: cancelled after 4+ hrs with no training steps
- node: nid010931
- exp_name: `rlt_6mix_joints_only_v1`
- failure: process stuck in data-loading loop (repeated "Fetching 212 files" in stderr, GPUs idle). An `/usr/bin/strace` tracer was found attached to the train.py process (94% CPU), which blocked forward progress. After killing the tracer, the process remained stuck in dataset fetching with no training steps logged.

## Job (resubmit — `4187578` — success)
- job_id: 4187578
- submitted/start: `2026-04-22T21:30:44+00:00`
- start_human: Wednesday, Apr 22nd, 2026
- end: `2026-04-23T06:43:58+00:00`
- end_human: Wednesday, Apr 23rd, 2026
- runtime: 09:13:14
- node: nid010827
- exp_name: `rlt_6mix_joints_only_v1`
- checkpoint_dir: `/scratch/u6cr/pravsels.u6cr/openpi/checkpoints/pi05_rlt_build_block_tower_6mix_joints_only/rlt_6mix_joints_only_v1`

## Status
- 2026-04-22 16:52 UTC — submitted as `4180356`; exited quickly with `ExitCode=126` before training loop
- 2026-04-22 16:55 UTC — fixed launch path in script to use self-contained checkpoint assets under `.../joints_only/49999/assets`
- 2026-04-22 16:55 UTC — resubmitted as `4180372`; entered `R` on `nid010931`
- 2026-04-22 ~17:00 UTC — startup confirmed: baseline params/assets loaded, dataset metadata loaded, W&B offline run created
- 2026-04-22 ~21:00 UTC — 4+ hrs in, still no training steps. Discovered `/usr/bin/strace` attached as tracer (94% CPU). Killed tracer but job remained stuck in dataset fetching loop. Cancelled.
- 2026-04-22 21:21 UTC — cleared old logs, resubmitted as `4187578` on `nid010827`
- 2026-04-22 21:31 UTC — dataset metadata loaded for all 6 datasets; JAX JIT compilation began (~600% CPU, GPUs idle as expected)
- 2026-04-22 — checked `task/rlt_block_tower` branch for unmerged fixes: the 5 unmerged commits are all run-log markdown updates and a slurm script tweak. No unmerged Python code changes.
- 2026-04-23 06:43 UTC — all 50k steps completed successfully (exit code 0). Final checkpoint saved at step 49999.

## Results
- final train loss: 104.8 (step 49900)
- final val loss: 190.9 (step 49000)
- loss dropped steadily: 9521 (step 0) → 104.8 (step 49900); no instability or divergence
- val loss: 9613 (step 0) → 190.9 (step 49000)
- checkpoints: 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 49999
- checkpoint dir: `/scratch/u6cr/pravsels.u6cr/openpi/checkpoints/pi05_rlt_build_block_tower_6mix_joints_only/rlt_6mix_joints_only_v1/`

## W&B
- local: `/scratch/u6cr/pravsels.u6cr/openpi/wandb/offline-run-20260422_213116-j6qz79ah`
- synced: https://wandb.ai/pravsels/rlt_build_block_tower_6mix_joints_only/runs/j6qz79ah
- notes: synced to new project `rlt_build_block_tower_6mix_joints_only`

## HuggingFace
- repo: https://huggingface.co/pravsels/rlt_build_block_tower_6mix_joints_only
- uploaded checkpoints: `45000` (best val loss among saved checkpoints, val_loss=191.3)
- includes: `checkpoints/45000/params`, `assets` (norm stats, per-timestep stats, episode split, valid indices)

## Next
- evaluate the RLT joints-only model
