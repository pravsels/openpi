# Train — RLT Stage 1 BusyBox push green button (GCloud)

## Mode
- run_type: RLT Stage 1 encoder/decoder
- objective: train the RL-token bottleneck on the frozen π0.5 green-button checkpoint so `hw_control.pi0_rlt` can extract tokens
- status: completed (exit 0); published to Hub; VM deleted

## Config
- script: `slurm/train_busybox_push_green_button_rlt_gcloud.sh`
- config: `pi05_rlt_busybox_push_green_button`
- exp_name: `busybox_push_green_button_rlt`
- dataset: `villekuosmanen/busybox_push_green_button` (LeRobot v3, 20 episodes, 2471 frames, 20 fps)
- prompt: `push the green button`
- cameras: `top` → `base_0_rgb`, `wrist` → `left_wrist_0_rgb`, `front` → `base_1_rgb`
- key settings: π0.5 + RLT, VLA frozen (`rl_vla_loss_weight=0.0` + `get_rl_freeze_filter()`), encoder/decoder 2 layers / 8 heads / dim 2048, action horizon 30, 6D (5 joints delta, gripper absolute), per-timestep action norm, global batch 16, 20k steps, save once at the end (`save_interval=20_000`, `keep_period=None`), cosine LR 5e-5 with 1k warmup and `decay_steps=20_000`, EMA 0.999
- parallelism: 1-GPU (`fsdp_devices=1`)
- input pipeline: TorchCodec, 8 persistent workers
- memory: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`
- init: Hub [`pravsels/pi05_busybox_push_green_button`](https://huggingface.co/pravsels/pi05_busybox_push_green_button) downloaded to `checkpoints/pi05_busybox_push_green_button/params`; Hub `assets/` copied, not recomputed
- code: `4354d7d` on `task/rlt_busybox_green_button`

## Infrastructure
- provider: GCloud
- project: `gen-lang-client-0388971498`
- vm: `openpi-rlt-busybox-green` / `us-central1-c`
- ssh: `gcloud compute ssh openpi-rlt-busybox-green --zone=us-central1-c`
- repo_on_vm: `/home/ps/openpi` (user `ps`)
- hardware: 1× NVIDIA A100-SXM4-80GB (`a2-ultragpu-1g`; do not use `a2-highgpu-1g` 40 GB)
- docker: `openpi:latest` (container `743ce8cf17dc`)
- disk: 1 TB boot (~953 GB free at launch)
- log: `/home/ps/openpi/logs/rlt_launch.out`
- checkpoints: `/home/ps/openpi/checkpoints/pi05_rlt_busybox_push_green_button/busybox_push_green_button_rlt/`
- assets: `/home/ps/openpi/assets/pi05_rlt_busybox_push_green_button/busybox_push_green_button_rlt/assets/`

## Status
- 2026-08-27 20:04 UTC — train.py started in Docker; JAX `device_count=1`, `CudaDevice(id=0)`; W&B online as `wnk0bxds`; Hub norm stats loaded; frozen VLA restored from `checkpoints/pi05_busybox_push_green_button/params`.
- 2026-08-27 20:06 UTC — progress bar opened at `-/20000`; first step compiling.
- 2026-08-27 20:09 UTC — step 0: `loss=9935.29`, `grad_norm=28114.27`, `param_norm=1836.10`.
- 2026-08-27 20:10 UTC — step 100: `loss=5386.35`, `grad_norm=11935.28`.
- 2026-08-27 20:11 UTC — step 200: `loss=1589.83`, `grad_norm=1497.15`.
- 2026-08-27 20:12 UTC — ~253/20k, ~1.4 it/s, ETA ~4h; no errors.
- GPU sample (20:10 UTC): 100% util, 75.8/80.0 GiB VRAM, 353/400 W, 65°C.
- Host sample (20:10 UTC): 12 GiB / 167 GiB RAM used; Docker ~14.2 GiB and ~411% CPU.
- 2026-08-27 22:13 UTC — step 10000: `loss=299.31`, `grad_norm=1402.69`.
- 2026-08-28 00:18 UTC — step 19900: `loss=189.18`, `grad_norm=805.95`; Orbax save of `19999` finalized.
- 2026-08-28 00:19 UTC — train.py exited 0 after 4h 15m 17s.

## Results
- runtime: `4:15:17` (start `2026-08-27T20:03:43+00:00`, end `2026-08-28T00:19:00+00:00`)
- final step: 19999
- start_train_loss: `9935.2900` (step 0)
- end_train_loss: `189.1828` (step 19900)
- checkpoint: `/home/ps/openpi/checkpoints/pi05_rlt_busybox_push_green_button/busybox_push_green_button_rlt/19999/`
- params size: 5.9 GiB (`_METADATA` + `manifest.ocdbt` present)
- assets: `norm_stats.json`, `norm_stats_actions_per_timestep.json`, `valid_indices.json`

## W&B
- project: `busybox_push_green_button_rlt`
- run: https://wandb.ai/pravsels/busybox_push_green_button_rlt/runs/wnk0bxds
- run id: `wnk0bxds`
- local: `/workspace/repo/wandb/run-20260827_200413-wnk0bxds`

## HuggingFace
- frozen VLA: https://huggingface.co/pravsels/pi05_busybox_push_green_button
- RLT checkpoint: https://huggingface.co/pravsels/pi05_rlt_busybox_push_green_button
- published: `params/` + `assets/` + `README.md` at the repo root (step 19999; `train_state/` excluded)
- uploaded: 2026-08-28 04:08 UTC

## Next
- pull `pravsels/pi05_rlt_busybox_push_green_button` for `hw_control.pi0_rlt` demo-cache / online RL
- VM `openpi-rlt-busybox-green` deleted after Hub verify
