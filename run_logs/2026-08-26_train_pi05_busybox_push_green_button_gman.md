# Train — π0.5 BusyBox push green button (GMAN)

## Mode
- run_type: full-component fine-tune
- objective: train π0.5 on `villekuosmanen/busybox_push_green_button` for the 30k-step green-button comparison
- status: interrupted; resuming from step 10k

## Config
- config: `pi05_busybox_push_green_button`
- exp_name: `pi05_busybox_push_green_button`
- dataset: `villekuosmanen/busybox_push_green_button` (LeRobot v3, 20 episodes, 2471 frames, 20 fps)
- prompt: `push the green button`
- cameras: `top` → `base_0_rgb`, `wrist` → `left_wrist_0_rgb`, `front` → `base_1_rgb`
- key settings: π0.5, full-component fine-tuning, action horizon 30, global batch 32, 30k steps, save every 5k and retain only the latest checkpoint, cosine LR 2.5e-5 → 2.5e-6, EMA 0.999
- parallelism: 4-GPU full data parallel (`fsdp_devices=1`), GPUs 4–7
- input pipeline: TorchCodec, 8 persistent workers
- memory: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`
- init: `weights/pi05_base/params`
- code: `27ee136` on `task/train_pi_policies_green_button`

## Infrastructure
- provider: GMAN
- mission: `pi-busybox-green-smoke-20260826`
- node: `pi-busybox-8xh100-prav`
- hardware: 8× NVIDIA H100 80GB HBM3; this run uses GPUs 4–7 while π0 uses GPUs 0–3
- final launch command id: `cmd-ce8uq`
- checkpoints: `/home/dev/openpi/checkpoints/pi05_busybox_push_green_button/pi05_busybox_push_green_button/`
- assets: `/home/dev/openpi_runs/pi05_busybox_push_green_button/pi05_busybox_push_green_button/assets/`

## Bring-up and throughput investigation
- 10-step FSDP-8 smoke passed training, Hub `step_5`/`step_9`, and W&B history. The train command exited 1 only because the immediate W&B history scan raced API ingestion; a retry found 10 rows. Smoke artifacts were deleted before production.
- FSDP-4 and FSDP-8 ran at roughly 3.1–3.5 s/iteration; the GPUs were sharding one model replica rather than increasing batch throughput.
- Full replication became viable after matching the proven Isambard/GCloud `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` setting.
- With PyAV and 2 workers, π0.5 remained input-bound around 3.0–3.5 s/iteration. Concurrent PyAV worker scaling was capped by `libdav1d` threads and the node's 4096 PID/thread cgroup limit.
- RoboCandyWrapper's PyAV default was replaced for this three-camera config with LeRobot's installed TorchCodec backend. Cached-decoder probes loaded subsequent BusyBox samples in 0.012–0.032 seconds.
- Commits `4939b0f` and `27ee136` enabled TorchCodec, restored full DP, set the proven 95% JAX pool, and selected 8 workers.
- GMAN stashed its generated internal-mirror `uv.lock` rewrite, fast-forwarded to `27ee136`, and passed 19 dependency-backed tests before the final launch.

## Status
- 2026-08-26 23:39 UTC — final production run `cmd-ce8uq` started from a clean production checkpoint directory.
- 2026-08-26 23:40 UTC — step 0 completed; W&B online.
- 2026-08-26 23:44 UTC — step 368/30k, recent rate ~2.0 iterations/s, ETA ~4h11m; step-300 loss 0.0647.
- GPU sample: GPUs 4–7 sustained 100% SM utilization, ~78.6 GiB VRAM, and ~630–660 W.
- Both concurrent jobs used about 3545/4096 cgroup threads after TorchCodec initialization, without new limit events.
- 2026-08-27 01:58 UTC — the asynchronous 15k checkpoint save failed with `RESOURCE_EXHAUSTED` / `ENOSPC` while writing the Orbax params database. Training continued to approximately step 15.6k before command `cmd-ce8uq` exited 1 at 02:04 UTC.
- Checkpoints 5k and 10k were complete. Recovery retains 10k, removes the obsolete 5k and incomplete 15k checkpoint, changes `keep_period` to `None`, and resumes with only the latest checkpoint retained.

## W&B
- project: `busybox_push_green_button_pi05`
- run: https://wandb.ai/pravsels/busybox_push_green_button_pi05/runs/z4ahya8v
- run id: `z4ahya8v`

## Next
- resume from the complete 10k checkpoint and verify the next save replaces it without exhausting disk
- let the run reach 30k and verify the final step-29999 checkpoint
- publish selected production checkpoints after training completes
