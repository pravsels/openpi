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
- end: `2026-11-23T02:10:04Z`
- end_human: Monday, Nov 23rd, 2026
- runtime: `04:37:21` (exit `0:0`, `MaxRSS` ~39.4 GiB / 230G)
- node: `nid010537`
- wandb: https://wandb.ai/pravsels/rbb-pi0/runs/9yxg7xp7

## Status
- 2026-11-22 21:32 UTC — submitted after 10/10 preflight pass; started immediately on `nid010537` (2× GH200). `$HOME` 1% used, `/scratch/u6abf/pravsels.u6abf` 6% used (4.8 TB free).
- 2026-11-22 21:32–21:36 UTC — computed fresh global + per-timestep action norm stats on 2-camera SO-101 dataset into scratch assets dir.
- 2026-11-22 21:37 UTC — `scripts/train.py` JIT compiled and began stepping: `Step 0: grad_norm=14.0995, loss=0.7021, param_norm=1377.8651`.
- 2026-11-22 21:41 UTC — steady at **1.2 it/s** (~4h 28m ETA), `.err` clean. `Step 100: loss=0.5199, grad_norm=9.1227` → `Step 200: loss=0.3051, grad_norm=6.0593`. GPU telemetry (`nvidia-smi` via `srun --overlap --mem=0`):
  - GPU 0: **100% util**, 36% mem util, **93,633 / 97,871 MiB VRAM**, **521.56 / 900.00 W**, 54°C
  - GPU 1: **100% util**, 36% mem util, **93,605 / 97,871 MiB VRAM**, **511.45 / 900.00 W**, 55°C
- 2026-11-22 21:42 UTC — `Step 300: grad_norm=4.9643, loss=0.2674, param_norm=1377.8674`.
- 2026-11-22 22:08 UTC — `Step 2200: grad_norm=1.9086, loss=0.1697, param_norm=1378.5223` (36m elapsed, steady).
- 2026-11-23 02:10 UTC — `COMPLETED` (`0:0`), `Step 19900: grad_norm=0.5686, loss=0.0221, param_norm=1381.7559`. Orbax finalized `19999` (`params/` 12G + `assets/` + `train_state/` 29G; rotated earlier steps cleanly).
- 2026-11-24 16:00 UTC — synced offline W&B run via `wandb beta sync` to `pravsels/rbb-pi0` (`pi0_rbb_so101_insert_gear_in_gripper`, `#ff7f0e`, 0 model artifacts), generated `CHECKPOINT_MANIFEST.json`, and uploaded `checkpoint-19999` (`params/` + `assets/` + `_CHECKPOINT_METADATA`) to public HF repo `armnet/pi0_rbb_so101_insert_gear_in_gripper`.

## Results
- runtime: `04:37:21` (start `2026-11-22T21:32:43Z`, end `2026-11-23T02:10:04Z`)
- final step: `19999` (last logged `19900`)
- start_train_loss: `0.7021` (step 0)
- end_train_loss: `0.0221` (step 19,900; `grad_norm=0.5686`, `param_norm=1381.7559`)
- start_val_loss: `n/a`
- end_val_loss: `n/a`
- loss_one_liner: Steady flow-matching loss decrease from 0.7021 to 0.0221 over 20k steps with smooth gradient norm decay (14.10 → 0.57).
- checkpoint: `/scratch/u6abf/pravsels.u6abf/rbb_openpi/checkpoints/pi0_rbb_so101_insert_gear_in_gripper/pi0_rbb_so101_insert_gear_in_gripper/19999`
- config_snapshot: `src/openpi/training/config.py` (`pi0_rbb_so101_insert_gear_in_gripper`)

## W&B
- project: `pravsels/rbb-pi0`
- run id: `9yxg7xp7` (display name: `pi0_rbb_so101_insert_gear_in_gripper`, color: `#ff7f0e`)
- local: `/scratch/u6abf/pravsels.u6abf/rbb_openpi/wandb/wandb/offline-run-20261122_213630-9yxg7xp7`
- synced: https://wandb.ai/pravsels/rbb-pi0/runs/9yxg7xp7
- notes: Synced via `wandb beta sync` (0 logged artifacts; metrics + 5 small `camera_views` PNGs only).

## HuggingFace
- repo: https://huggingface.co/armnet/pi0_rbb_so101_insert_gear_in_gripper (public)
- uploaded checkpoints: `checkpoint-19999` (`params/` + `assets/` + `_CHECKPOINT_METADATA`, 11.189 GiB; `train_state/` excluded)
- includes: `README.md`, `TRAINING_LOG.md`, `CHECKPOINT_MANIFEST.json`, `checkpoint-19999/`
- manifest sha256: `8a7f701fdbb8d571110978805fb694fb7fb4f18496544cefc00c192850a8ca30`

## Next
- Run robot eval on `armnet/pi0_rbb_so101_insert_gear_in_gripper` (`checkpoint-19999`, after `verify-checkpoint`).
- Launch the remaining 8 RoboBuilderBench tasks in staggered batches.
