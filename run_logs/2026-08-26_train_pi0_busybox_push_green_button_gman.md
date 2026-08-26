# Train — π0 BusyBox push green button (GMAN)

## Mode
- run_type: full-component fine-tune
- objective: train π0 on `villekuosmanen/busybox_push_green_button` for the 30k-step green-button comparison
- status: running

## Config
- config: `pi0_busybox_push_green_button`
- exp_name: `pi0_busybox_push_green_button`
- dataset: `villekuosmanen/busybox_push_green_button` (LeRobot v3, 20 episodes, 2471 frames, 20 fps)
- prompt: `push the green button`
- cameras: `top` → `base_0_rgb`, `wrist` → `left_wrist_0_rgb`, `front` → `base_1_rgb`
- key settings: π0, full-component fine-tuning, action horizon 30, global batch 32, 30k steps, save/keep every 5k, cosine LR 2.5e-5 → 2.5e-6, EMA 0.999
- parallelism: 4-GPU full data parallel (`fsdp_devices=1`), GPUs 0–3
- input pipeline: TorchCodec, 8 persistent workers
- memory: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`
- init: `weights/pi0_base/params`
- code: `27ee136` on `task/train_pi_policies_green_button`

## Infrastructure
- provider: GMAN
- mission: `pi-busybox-green-smoke-20260826`
- node: `pi-busybox-8xh100-prav`
- hardware: 8× NVIDIA H100 80GB HBM3; this run uses GPUs 0–3 while π0.5 uses GPUs 4–7
- final launch command id: `cmd-ij8dr`
- checkpoints: `/home/dev/openpi/checkpoints/pi0_busybox_push_green_button/pi0_busybox_push_green_button/`
- assets: `/home/dev/openpi_runs/pi0_busybox_push_green_button/pi0_busybox_push_green_button/assets/`

## Bring-up and throughput investigation
- 10-step FSDP-8 smoke passed training, Hub `step_5`/`step_9`, and W&B history; smoke artifacts were deleted before production.
- Initial full-replica smoke used JAX's default 75% HBM pool and OOMed after reaching ~62 GiB while requesting another ~14.6 GiB.
- FSDP-8 and FSDP-4 avoided OOM but ran at roughly 3.3–3.5 s/iteration because sharding reduced memory rather than providing data-parallel throughput.
- Setting `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` allowed full replication at ~78.6 GiB/GPU.
- PyAV with 2 workers left GPUs idle between batches (~3.4 s/iteration). Increasing to 8 workers briefly reached ~2.2 iterations/s but hit the node's 4096 PID/thread cgroup limit in `libdav1d`.
- Root cause: RoboCandyWrapper defaulted to PyAV even though TorchCodec was installed. TorchCodec cached-decoder probes loaded subsequent samples in 0.012–0.032 seconds.
- Commits `4939b0f` and `27ee136` enabled TorchCodec, restored full DP, set the proven 95% JAX pool, and selected 8 workers.
- GMAN stashed its generated internal-mirror `uv.lock` rewrite, fast-forwarded to `27ee136`, and passed 19 dependency-backed tests before the final launch.

## Status
- 2026-08-26 23:39 UTC — final production run `cmd-ij8dr` started from a clean production checkpoint directory.
- 2026-08-26 23:40 UTC — step 0 completed; W&B online.
- 2026-08-26 23:44 UTC — step 430/30k, recent rate ~2.2 iterations/s, ETA ~3h48m; step-400 loss 0.1920.
- GPU sample: GPUs 0–3 sustained 99–100% SM utilization, ~78.6 GiB VRAM, and ~580–650 W.

## W&B
- project: `busybox_push_green_button_pi0`
- run: https://wandb.ai/pravsels/busybox_push_green_button_pi0/runs/xmrjumvc
- run id: `xmrjumvc`

## Next
- let the run reach 30k and verify checkpoints at 5k intervals plus final step 29999
- publish selected production checkpoints after training completes
