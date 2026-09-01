# Train — π0.5 BusyBox multitask absolute actions

## Mode
- run_type: full-component fine-tune
- objective: train π0.5 on `villekuosmanen/busybox_multitask` with absolute 6D actions, true min/max normalization, and per-episode task prompts
- status: running (30k; smoke skipped; restarted on cheaper SXM)

## Config
- config: `pi05_busybox_multitask_abs`
- exp_name: `pi05_busybox_multitask_abs`
- dataset: `villekuosmanen/busybox_multitask` (LeRobot v3, 66 episodes, 12141 frames, 20 fps, 27 tasks)
- prompt: `prompt_from_task` (no single default)
- cameras: `top` → `base_0_rgb`, `wrist` → `left_wrist_0_rgb`, `front` → `base_1_rgb`
- key settings: π0.5, full-component fine-tuning, action horizon 30, global batch 32, 30k steps, save every 5k and retain only the latest checkpoint, cosine LR 2.5e-5 → 2.5e-6, EMA 0.999, absolute 6D actions, per-timestep min/max bounds mapped to `[-1, 1]`
- parallelism: 4-GPU full data parallel (`fsdp_devices=1`)
- input pipeline: TorchCodec, 8 persistent workers
- memory: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`
- publishing: each finalized checkpoint replaces the Hugging Face repo root; `train_state` is excluded
- init: `weights/pi05_base/params`
- code: `e846282` on `task/busybox_multitask`
- plan: `docs/plans/2026-09-01-pi05-busybox-multitask-abs.md`

## Infrastructure
- provider: Vast
- instance: `49590717` (`pi05-busybox-multitask-abs`)
- offer: `49362311` 4× H100 SXM 80GB HBM3, Netherlands, billed ~$11.11/hr, 1600 GB disk
- image: `nvcr.io/nvidia/pytorch:25.03-py3`
- SSH: `ssh -i ~/.ssh/id_ed25519 -p 30716 root@ssh6.vast.ai`
- clone: `/workspace/openpi` at `e846282` (`task/busybox_multitask`)
- launch: `CONFIG_NAME=pi05_busybox_multitask_abs WANDB_PROJECT=busybox_multitask_pi05_abs nohup bash vast/train.sh` (no `SMOKE=1`); `uv run scripts/train.py pi05_busybox_multitask_abs --overwrite`
- train log: `/workspace/vast_runs/openpi/logs/train_30k.log`
- bootstrap log: `/workspace/vast_runs/openpi/logs/bootstrap.log`
- checkpoints: `/workspace/openpi/checkpoints/pi05_busybox_multitask_abs/pi05_busybox_multitask_abs/`
- assets: `/root/openpi_runs/pi05_busybox_multitask_abs/pi05_busybox_multitask_abs/assets`
- prior US box `49589136` destroyed before 5k; do not reuse relative-run assets or touch relative `49582742` / CRA `49561214`

## Status
- 2026-09-01 — Variant 2 config and true min/max normalization wiring prepared locally.
- 2026-09-01 22:52 UTC — rented US Vast `49589136` after discarding a PCIe box.
- 2026-09-01 23:00 UTC — first 30k on `49589136`. W&B `zw46eika`. Step 0 `loss=0.2664`.
- 2026-09-01 23:07 UTC — rented cheaper NL `49590717` (~$11.11/hr). User destroyed US `49589136` (~$21.72/hr) before the 5k checkpoint.
- 2026-09-01 23:09–23:13 UTC — `vast/bootstrap.sh` finished `setup_done` on `49590717`. `branch_assets_ok`. Staged `pi0_base` and `pi05_base`.
- 2026-09-01 23:18 UTC — 30k restarted from step 0 on `49590717`. JAX `device_count=4`. W&B online as `1xb73qsx`. Norm stats loaded from the abs assets dir (`use_min_max_norm_stats=True`).
- 2026-09-01 23:20 UTC — first-step compile: XLA gemm autotune `Results do not match the reference` and rematerialization memory warnings; training continued.
- 2026-09-01 23:20 UTC — step 0: `loss=0.2664`, `grad_norm=4.6466`, `param_norm=1802.3865`.
- 2026-09-01 23:21 UTC — step 100: `loss=0.0865`, `grad_norm=1.1302`.
- 2026-09-01 23:22 UTC — step 200: `loss=0.0429`, `grad_norm=0.4300`.
- 2026-09-01 23:23 UTC — step 300: `loss=0.0350`, `grad_norm=0.3945`.
- 2026-09-01 23:24 UTC — step 400: `loss=0.0304`, `grad_norm=0.3386`.
- throughput: ~2.0 it/s after compile (~0.50 s/step); remaining ~4h10m to 30k. Same envelope as the relative Taiwan 30k and the killed US abs box.
- GPU sample (23:21 UTC): 4× H100 ~59–98% util, ~78.6 / 81.6 GiB.

## W&B
- project: `busybox_multitask_pi05_abs`
- run: https://wandb.ai/pravsels/busybox_multitask_pi05_abs/runs/1xb73qsx
- run id: `1xb73qsx`
- local: `/workspace/openpi/wandb/run-20260901_231849-1xb73qsx`
- killed US run: https://wandb.ai/pravsels/busybox_multitask_pi05_abs/runs/zw46eika

## HuggingFace
- Hub: https://huggingface.co/pravsels/pi05_busybox_multitask_abs
- publisher: `scripts/gman_publish_latest.py` (started with train; steps 5k–30k when they finalize)

## Next
- let the run reach 30k and verify Hub 5k–30k plus train/publisher exit codes
- do not destroy `49590717` or relative `49582742` without asking
