# Train — π0.5 BusyBox multitask (Vast)

## Mode
- run_type: full-component fine-tune
- objective: train π0.5 on `villekuosmanen/busybox_multitask` with the green-button relative-action recipe and per-episode task prompts
- status: completed (`train_done`); published to Hub; instance destroyed

## Config
- config: `pi05_busybox_multitask`
- exp_name: `pi05_busybox_multitask`
- dataset: `villekuosmanen/busybox_multitask` (LeRobot v3, 66 episodes, 12141 frames, 20 fps, 27 tasks)
- prompt: `prompt_from_task` (no single default)
- cameras: `top` → `base_0_rgb`, `wrist` → `left_wrist_0_rgb`, `front` → `base_1_rgb`
- key settings: π0.5, full-component fine-tuning, action horizon 30, global batch 32, 30k steps, save every 5k and retain only the latest checkpoint, cosine LR 2.5e-5 → 2.5e-6, EMA 0.999, relative actions (5 joints delta, gripper absolute), per-timestep quantile
- parallelism: 4-GPU full data parallel (`fsdp_devices=1`)
- input pipeline: TorchCodec, 8 persistent workers
- memory: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`
- publishing: each finalized checkpoint replaces the Hugging Face repo root; `train_state` is excluded
- init: `weights/pi05_base/params`
- code: `412112c` on `task/busybox_multitask`
- plan: `docs/plans/2026-09-01-pi05-busybox-multitask.md`

## Infrastructure
- provider: Vast
- instance: `49582742` (`pi05-busybox-multitask`) — destroyed 2026-09-02 after `train_done`
- offer: `49467276` 4× H100 SXM 80GB HBM3, Taiwan, billed ~$13.54/hr, 1516 GB disk
- image: `nvcr.io/nvidia/pytorch:25.03-py3`
- SSH: `ssh -i ~/.ssh/id_ed25519 -p 22742 root@ssh8.vast.ai`
- clone: `/workspace/openpi` at `412112c` (`task/busybox_multitask`)
- launch: `nohup bash vast/train.sh` (no `SMOKE=1`); `uv run scripts/train.py pi05_busybox_multitask --overwrite`
- train log: `/workspace/vast_runs/openpi/logs/train_30k.log`
- bootstrap log: `/workspace/vast_runs/openpi/logs/bootstrap.log`
- checkpoints: `/workspace/openpi/checkpoints/pi05_busybox_multitask/pi05_busybox_multitask/`
- assets: `/root/openpi_runs/pi05_busybox_multitask/pi05_busybox_multitask/assets`
- do not touch CRA `49561214`

## Status
- 2026-09-01 — config rewritten to single-arm three-cam / 30k. Launch pending.
- 2026-09-01 21:35 UTC — rented Vast `49582742`. SSH up, 4× H100 80GB, clone at `6b94b6b`. Bootstrap waiting on `/workspace/secrets/{hf_token,wandb_token}`.
- 2026-09-01 21:40–21:52 UTC — `vast/bootstrap.sh` finished `setup_done`. `branch_assets_ok`. Staged `pi0_base` and `pi05_base`.
- 2026-09-01 22:00 UTC — 30k started without the 10-step smoke. JAX `device_count=4`. W&B online as `gi5dv2qh`.
- 2026-09-01 22:01 UTC — first-step compile: XLA gemm autotune `Results do not match the reference` and rematerialization memory warnings; training continued.
- 2026-09-01 22:02 UTC — step 0: `loss=0.2662`, `grad_norm=2.2733`, `param_norm=1802.3865`.
- 2026-09-01 22:03 UTC — step 100: `loss=0.1571`, `grad_norm=1.2318`. Laptop SSH that started nohup was aborted; remote train kept running.
- 2026-09-01 22:04 UTC — step 200: `loss=0.1007`, `grad_norm=0.9142`.
- 2026-09-01 22:05 UTC — step 300: `loss=0.0857`, `grad_norm=0.8005`; ~2.0 it/s, remaining ~4h10m.
- GPU sample (22:05 UTC): 4× H100 ~99–100% util, ~78.6 / 81.6 GiB.
- 2026-09-01 22:50 UTC — published Hub step 5000.
- 2026-09-01 23:33 UTC — published Hub step 10000.
- 2026-09-02 00:16 UTC — published Hub step 15000.
- 2026-09-02 01:01 UTC — published Hub step 20000.
- 2026-09-02 01:44 UTC — published Hub step 25000.
- 2026-09-02 02:22 UTC — Orbax save of `29999` finalized (`Waiting for checkpoint manager to finish`).
- 2026-09-02 02:28 UTC — published Hub step 29999; wrapper printed `train_done`. GPUs idle.

## Results
- runtime: ~4h 28m (train start `2026-09-01T22:00:20+00:00`, Hub 29999 / `train_done` `2026-09-02T02:28:01+00:00`)
- final step: 29999
- start_train_loss: `0.2662` (step 0)
- end_train_loss: `0.0031` (step 29900)
- local checkpoint: `.../pi05_busybox_multitask/pi05_busybox_multitask/29999/` (`keep_period=None`)
- assets: `norm_stats.json`, `norm_stats_actions_per_timestep.json`, `valid_indices.json`

## W&B
- project: `busybox_multitask_pi05`
- run: https://wandb.ai/pravsels/busybox_multitask_pi05/runs/gi5dv2qh
- run id: `gi5dv2qh`
- local: `/workspace/openpi/wandb/run-20260901_220031-gi5dv2qh`

## HuggingFace
- Hub: https://huggingface.co/pravsels/pi05_busybox_multitask
- published: repo root replaced at 5k, 10k, 15k, 20k, 25k, and 29999 (`train_state` excluded)
- uploaded: 2026-09-02 02:28 UTC (final)

## Next
- pull `pravsels/pi05_busybox_multitask` for eval / RLT later
- instance `49582742` destroyed after Hub verify
