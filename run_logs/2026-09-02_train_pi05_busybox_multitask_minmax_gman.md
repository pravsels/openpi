# Train — π0.5 BusyBox multitask minmax (GMAN)

## Mode
- run_type: full-component fine-tune
- objective: train π0.5 on `villekuosmanen/busybox_multitask` with the relative-action recipe (5 joints delta, gripper absolute) and true min/max bounds in `q01`/`q99`
- status: stopped before 30k — bootstrap `cmd-dcr8t` killed; node parked. Language-conditioning bug found; do not resume until prompts are remapped.

## Config
- config: `pi05_busybox_multitask_minmax`
- exp_name: `pi05_busybox_multitask_minmax`
- dataset: `villekuosmanen/busybox_multitask` (LeRobot v3, 66 episodes, 12141 frames, 20 fps, 27 tasks)
- prompt: `prompt_from_task` (no single default)
- cameras: `top` → `base_0_rgb`, `wrist` → `left_wrist_0_rgb`, `front` → `base_1_rgb`
- key settings: π0.5, full-component fine-tuning, action horizon 30, global batch 32, 30k steps, save every 5k and retain only the latest checkpoint, cosine LR 2.5e-5 → 2.5e-6, EMA 0.999, relative actions (5 joints delta, gripper absolute), per-timestep min/max bounds mapped to `[-1, 1]`
- parallelism: 8-GPU full data parallel (`fsdp_devices=1`)
- input pipeline: TorchCodec, 8 persistent workers
- memory: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`
- publishing: each finalized checkpoint replaces the Hugging Face repo root; `train_state` is excluded
- init: `weights/pi05_base/params`
- code: `eb12f89` on `task/busybox_multitask_minmax`

## Infrastructure
- provider: GMAN
- mission: `pi05-busybox-multitask-minmax-20260902` — https://givemeanode.com/missions/pi05-busybox-multitask-minmax-20260902
- node: `pi05-busybox-minmax-8xh100-prav`
- hardware: 8× NVIDIA H100 80GB
- clone: `/home/dev/openpi` at `task/busybox_multitask_minmax`
- launch: `CONFIG_NAME=pi05_busybox_multitask_minmax bash gman/train.sh` (no `SMOKE=1`)
- checkpoints: `/home/dev/openpi/checkpoints/pi05_busybox_multitask_minmax/pi05_busybox_multitask_minmax/`
- assets: `/home/dev/openpi_runs/pi05_busybox_multitask_minmax/pi05_busybox_multitask_minmax/assets`
- do not reuse relative-run or abs-run assets

## Job
- execution_id: node `pi05-busybox-minmax-8xh100-prav` (`069269d6-99fa-4b97-876d-4fe0c536e721`); bootstrap `cmd-dcr8t`
- submitted/start: `2026-09-02T09:03Z`
- start_human: Wednesday, Sep 2nd, 2026
- node: `pi05-busybox-minmax-8xh100-prav` (8× H100 80GB, `cuda-12.9`, scratch 250 GiB)

## Status
- 2026-09-02 — config `pi05_busybox_multitask_minmax` added on `task/busybox_multitask_minmax` (`eb12f89`).
- 2026-09-02 09:03 UTC — GMAN 8×H100 `pi05-busybox-minmax-8xh100-prav` running. Bootstrap `cmd-dcr8t` started with `REPO_REF=task/busybox_multitask_minmax`.

## Results

## W&B
- project: `busybox_multitask_pi05_minmax`
- synced: pending

## Next
- bootstrap, then 30k without smoke
