# Train — π0.5 BusyBox multitask absolute actions

## Mode
- run_type: full-component fine-tune
- objective: train π0.5 on `villekuosmanen/busybox_multitask` with absolute 6D actions, true min/max normalization, and per-episode task prompts
- status: running (30k; smoke skipped)

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
- code: `03def4f` on `task/busybox_multitask`
- plan: `docs/plans/2026-09-01-pi05-busybox-multitask-abs.md`

## Infrastructure
- provider: Vast
- instance: `49589136` (`pi05-busybox-multitask-abs`)
- offer: `36346698` 4× H100 SXM 80GB HBM3, US, billed ~$21.72/hr, 1600 GB disk
- image: `nvcr.io/nvidia/pytorch:25.03-py3`
- SSH: `ssh -i ~/.ssh/id_ed25519 -p 29136 root@ssh6.vast.ai`
- clone: `/workspace/openpi` at `03def4f` (`task/busybox_multitask`)
- launch: `CONFIG_NAME=pi05_busybox_multitask_abs WANDB_PROJECT=busybox_multitask_pi05_abs nohup bash vast/train.sh` (no `SMOKE=1`); `uv run scripts/train.py pi05_busybox_multitask_abs --overwrite`
- train log: `/workspace/vast_runs/openpi/logs/train_30k.log`
- bootstrap log: `/workspace/vast_runs/openpi/logs/bootstrap.log`
- checkpoints: `/workspace/openpi/checkpoints/pi05_busybox_multitask_abs/pi05_busybox_multitask_abs/`
- assets: `/root/openpi_runs/pi05_busybox_multitask_abs/pi05_busybox_multitask_abs/assets`
- do not reuse relative-run assets or touch relative `49582742` / CRA `49561214`

## Status
- 2026-09-01 — Variant 2 config and true min/max normalization wiring prepared locally.
- 2026-09-01 22:52 UTC — rented Vast `49589136` after discarding a PCIe box. SSH up, 4× H100 80GB SXM, clone at `03def4f`.
- 2026-09-01 22:54–22:55 UTC — `vast/bootstrap.sh` finished `setup_done`. `branch_assets_ok`. Staged `pi0_base` and `pi05_base`.
- 2026-09-01 23:00 UTC — 30k started without the 10-step smoke. JAX `device_count=4`. W&B online as `zw46eika`. Norm stats loaded from the abs assets dir (`use_min_max_norm_stats=True`).
- 2026-09-01 23:02 UTC — first-step compile: XLA gemm autotune `Results do not match the reference` and rematerialization memory warnings; training continued.
- 2026-09-01 23:02 UTC — step 0: `loss=0.2664`, `grad_norm=4.6466`, `param_norm=1802.3865`.
- 2026-09-01 23:03 UTC — step 100: `loss=0.0865`, `grad_norm=1.1302`; ~1.9 it/s, remaining ~4h20m.
- GPU sample (23:03 UTC): 4× H100 ~100% util, ~78.6 / 81.6 GiB.

## W&B
- project: `busybox_multitask_pi05_abs`
- run: https://wandb.ai/pravsels/busybox_multitask_pi05_abs/runs/zw46eika
- run id: `zw46eika`
- local: `/workspace/openpi/wandb/run-20260901_230048-zw46eika`

## HuggingFace
- Hub: https://huggingface.co/pravsels/pi05_busybox_multitask_abs
- publisher: `scripts/gman_publish_latest.py` (started with train; steps 5k–30k when they finalize)

## Next
- let the run reach 30k and verify Hub 5k–30k plus train/publisher exit codes
- do not destroy `49589136` or relative `49582742` without asking
