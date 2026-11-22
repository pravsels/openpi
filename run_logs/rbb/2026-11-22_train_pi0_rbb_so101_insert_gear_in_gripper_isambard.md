# Train — π0 `pi0_rbb_so101_insert_gear_in_gripper` (SO-101, Isambard)

## Mode
- run_type: experiment (RoboBuilderBench baseline, initial end-to-end validation run)
- objective: fine-tune π0 (`pi0_base`, full-component) for 20,000 steps on SO-101 `armnet/so101_insert_gear_in_gripper` on 2× Isambard GH200 and validate norm stats, checkpoints, and offline W&B landing on scratch.

## Config
- code: `task/diff_pi0_rbb_finetunes` at `a445976`
- script: `slurm/train_rbb_pi0_slurm.sh pi0_rbb_so101_insert_gear_in_gripper`
- config: `pi0_rbb_so101_insert_gear_in_gripper` (`src/openpi/training/config.py`)
- exp_name: `pi0_rbb_so101_insert_gear_in_gripper`
- dataset: [armnet/so101_insert_gear_in_gripper](https://huggingface.co/datasets/armnet/so101_insert_gear_in_gripper) pinned at `42e271c25c30d934449c4acf73af96c48ff86b6f` (SO-101, 30 Hz, 6-D state/action, 2 cameras via `LeRobotSO101TopWristDataConfig`: `top` → `base_0_rgb`, `wrist` → `left_wrist_0_rgb`, `video_backend=pyav`)
- prompt: `Insert gear into gripper`
- init: `/scratch/u6abf/pravsels.u6abf/rbb_openpi/weights/pi0_base/params`
- key settings: π0 (`pi05=False`), `action_horizon=30`, `use_delta_actions=True`, per-timestep action norm stats, `num_train_steps=20000`, `batch_size=32` (16 per GPU across 2× GH200 data-parallel, `fsdp_devices=1`), `num_workers=6`, cosine LR `2.5e-5 → 2.5e-6` over 20k with 1k warmup, `ema_decay=0.999`, `save_interval=5000`, `keep_period=None`, `wandb_enabled=True` (offline)
- checkpoints: `/scratch/u6abf/pravsels.u6abf/rbb_openpi/checkpoints/pi0_rbb_so101_insert_gear_in_gripper/pi0_rbb_so101_insert_gear_in_gripper`
- assets: `/scratch/u6abf/pravsels.u6abf/rbb_openpi/assets/pi0_rbb_so101_insert_gear_in_gripper/pi0_rbb_so101_insert_gear_in_gripper/assets`
- runbook: `alpha-robotics/docs/plans/2026-11-22-diff-pi0-rbb-isambard.md`

## Infrastructure
- cluster: Isambard AI Phase 2 (`u6abf.aip2.isambard`)
- account: project `u6abf`, user `pravsels.u6abf`
- hardware: 2× NVIDIA GH200 120GB (`arm64`), 32 requested / 72 allocated CPUs, 230 GiB RAM
- container: `/scratch/u6abf/pravsels.u6abf/rbb_openpi/container/openpi_arm64.sif` (`14G`, `jax[cuda12]==0.5.3`, `torch==2.7.1+cpu`, `av==15.1.0`)
- checkout: `/home/u6abf/pravsels.u6abf/openpi_rbb`
- stdout/stderr: `/scratch/u6abf/pravsels.u6abf/rbb_openpi/logs/train_8737897.{out,err}`

## Job
- execution_id: `8737897` (`pi0_so101_gear`)
- submitted/start: `2026-11-22T21:32:41Z` / `2026-11-22T21:32:43Z`
- start_human: Sunday, Nov 22nd, 2026
- end: pending (walltime limit `2026-11-23T21:32:43Z`, 24h)
- end_human: pending
- runtime: running
- node: `nid010537`

## Status
- 2026-11-22 21:32 UTC — submitted after 10/10 preflight pass; started immediately on `nid010537` (2× GH200). `$HOME` 1% used, `/scratch/u6abf/pravsels.u6abf` 6% used (4.8 TB free).
- 2026-11-22 21:32–21:36 UTC — computed fresh global + per-timestep action norm stats on 2-camera SO-101 dataset into scratch assets dir.
- 2026-11-22 21:37 UTC — `scripts/train.py` JIT compiled and began stepping: `Step 0: grad_norm=14.0995, loss=0.7021, param_norm=1377.8651`.
- 2026-11-22 21:41 UTC — steady at **1.2 it/s** (~4h 28m ETA), `.err` clean. `Step 100: loss=0.5199, grad_norm=9.1227` → `Step 200: loss=0.3051, grad_norm=6.0593`. GPU telemetry (`nvidia-smi` via `srun --overlap --mem=0`):
  - GPU 0: **100% util**, 36% mem util, **93,633 / 97,871 MiB VRAM**, **521.56 / 900.00 W**, 54°C
  - GPU 1: **100% util**, 36% mem util, **93,605 / 97,871 MiB VRAM**, **511.45 / 900.00 W**, 55°C
- 2026-11-22 21:42 UTC — `Step 300: grad_norm=4.9643, loss=0.2674, param_norm=1377.8674`.
- 2026-11-22 22:08 UTC — `Step 2200: grad_norm=1.9086, loss=0.1697, param_norm=1378.5223` (36m elapsed, steady).

## Results
- pending (running; checkpoints saving every 5,000 steps to `/scratch/u6abf/pravsels.u6abf/rbb_openpi/checkpoints/pi0_rbb_so101_insert_gear_in_gripper/pi0_rbb_so101_insert_gear_in_gripper/`)

## W&B
- project: `rbb_pi0_so101_insert_gear_in_gripper`
- run id: `9yxg7xp7`
- local: `/scratch/u6abf/pravsels.u6abf/rbb_openpi/wandb/wandb/offline-run-20261122_213630-9yxg7xp7`
- synced: pending — run `autohpc-wandb-sync sync` after confirming entity/project with user
- notes: pending completion

## Next
- Monitor to completion (~4.5h ETA, ~02:10 UTC), then record `sacct`/`seff`, final loss, and checkpoint paths (`19999/params`).
- Sync offline W&B run once entity/project are confirmed.
- Generate `CHECKPOINT_MANIFEST.json` before publishing or transferring checkpoints.
