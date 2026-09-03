# Train — RLT Stage 1 BusyBox multitask single-arm minmax (GCloud)

## Mode
- run_type: replication
- objective: train the RL-token bottleneck on the frozen prompt-fix π0.5 minmax multitask checkpoint so `hw_control.pi0_rlt` can extract tokens
- status: completed (exit 0); published to Hub

## Config
- script: `slurm/train_busybox_multitask_singlearm_minmax_rlt_gcloud.sh`
- config: `pi05_rlt_busybox_multitask_singlearm_minmax`
- exp_name: `busybox_multitask_rlt_singlearm_minmax`
- dataset: `villekuosmanen/busybox_multitask` (LeRobot v3, 66 episodes, 12141 frames, 20 fps, 27 tasks)
- prompt: `prompt_from_task` (RCW git `main` `#597aa9ad`, remap `de4e4eb`)
- cameras: `top` → `base_0_rgb`, `wrist` → `left_wrist_0_rgb`, `front` → `base_1_rgb`
- key settings: π0.5 + RLT, VLA frozen (`rl_vla_loss_weight=0.0` + `get_rl_freeze_filter()`), encoder/decoder 2 layers / 8 heads / dim 2048, action horizon 30, 6D (5 joints delta, gripper absolute), per-timestep min/max in `q01`/`q99` (`use_min_max_norm_stats=True`), global batch 16, 20k steps, save once at the end, cosine LR 5e-5 with 1k warmup, EMA 0.999
- parallelism: 1-GPU (`fsdp_devices=1`)
- input pipeline: TorchCodec, 8 persistent workers
- memory: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`
- init: Hub [`pravsels/pi05_busybox_multitask_minmax`](https://huggingface.co/pravsels/pi05_busybox_multitask_minmax) (W&B `swjv9hbs`); Hub `assets/` copied, not recomputed
- code: `90490b9` on `task/rlt_busybox_multitask`
- RCW: git `main` lock `597aa9ad21176e7f7dcee4aede5dc1ffc07eacee`
- plan: `docs/plans/2026-09-03-rlt-busybox-multitask-singlearm-design.md`

## Infrastructure
- provider: GCloud
- project: `gen-lang-client-0388971498`
- vm: `openpi-rlt-busybox-multitask-minmax` / `us-central1-c` (fresh; not `openpi-rlt-busybox-multitask`)
- ssh: `gcloud compute ssh openpi-rlt-busybox-multitask-minmax --zone=us-central1-c`
- repo_on_vm: `/home/user/openpi` (`git clone --branch task/rlt_busybox_multitask`)
- hardware: 1× NVIDIA A100-SXM4-80GB (`a2-ultragpu-1g`; do not use `a2-highgpu-1g` 40 GB)
- docker: build `openpi:latest` on the VM from this branch
- os_image: `ubuntu-accelerator-2204-amd64-with-nvidia-580`
- disk: 1 TB pd-ssd
- do not reuse shuffled-language Hub trees; do not use GMAN

## Job
- execution_id: `openpi-rlt-busybox-multitask-minmax` / `us-central1-c`
- submitted/start: `2026-09-03T12:16:05Z` (launcher)
- start_human: Thursday, Sep 3rd, 2026
- end: `2026-09-03T16:28:39Z`
- end_human: Thursday, Sep 3rd, 2026
- runtime: `4h 12m 34s`

## Status
- 2026-09-03 — relative single-arm RLT already running on `openpi-rlt-busybox-multitask`. This run is a separate VM.
- 2026-09-03 11:58 UTC — created `openpi-rlt-busybox-multitask-minmax` (`a2-ultragpu-1g`, 1 TB pd-ssd, `ubuntu-accelerator-2204-amd64-with-nvidia-580`, nat 34.29.14.198).
- 2026-09-03 12:00 UTC — `git clone --branch task/rlt_busybox_multitask` → `90490b9` at `/home/user/openpi`.
- 2026-09-03 12:07 UTC — `openpi:latest` built (17.7 GB).
- 2026-09-03 12:13 UTC — launcher started. Hub VLA `pravsels/pi05_busybox_multitask_minmax` downloaded.
- 2026-09-03 12:16 UTC — `prompt_ok`: `wrapped_tasks 27`, `mismatches 0`, index 0 is `Move the left slider to position 1`. `rcw_sha_ok 597aa9ad`.
- 2026-09-03 12:16 UTC — train.py up. JAX `device_count=1`, `CudaDevice(id=0)`. W&B `93keszgb`. Hub assets + 12141 valid indices. `prompt_from_task=True`, `use_min_max_norm_stats=True`. Restored `checkpoints/pi05_busybox_multitask_minmax/params`.
- 2026-09-03 12:21 UTC — step 0: `loss=10755.68`, `grad_norm=33158.02`, `param_norm=1836.38`. GPU 76%, 77.6/80.0 GiB.
- 2026-09-03 12:22 UTC — 86/20000, ~1.4 it/s, ETA ~4h. GPU 100%, 77.6/80.0 GiB.
- 2026-09-03 12:22 UTC — step 100: `loss=5612.74`, `grad_norm=14162.65`, `param_norm=1836.38`.
- 2026-09-03 12:23 UTC — 184/20000, ~1.4 it/s, ETA ~4h. GPU 100%, 77.6/80.0 GiB. No errors.
- 2026-09-03 14:43 UTC — step 11500: `loss=379.56`, `grad_norm=1144.63`, `param_norm=1837.09`. 11.6k/20000, ~1.4 it/s, ETA ~1h 44m. GPU 100%, 77.6/80.0 GiB. No errors.
- 2026-09-03 15:01 UTC — step 12900: `loss=358.34`, `grad_norm=1229.78`, `param_norm=1837.24`. 13.0k/20000, ~1.3 it/s, ETA ~1h 27m. GPU 100%, 77.6/80.0 GiB. No errors.
- 2026-09-03 15:53 UTC — step 17200: `loss=315.43`, `grad_norm=1028.22`, `param_norm=1837.74`. 17.2k/20000, ~1.3–1.4 it/s, ETA ~34m. GPU 100%, 77.6/80.0 GiB. No errors.
- 2026-09-03 16:27 UTC — step 19900: `loss=293.81`, `grad_norm=999.75`, `param_norm=1838.08`. Orbax save of `19999` started.
- 2026-09-03 16:28 UTC — checkpoint finalized at `19999` (`params/` + `assets/` + `train_state/`). Launcher exit 0. GPU idle.
- 2026-09-03 16:52 UTC — published step 19999 to Hub `pravsels/pi05_rlt_busybox_multitask_singlearm_minmax` (`params/` + `assets/` + README; no `train_state/`).

## Results
- runtime: `4:12:34` (start `2026-09-03T12:16:05Z`, end `2026-09-03T16:28:39Z`)
- final step: 19999
- start_train_loss: `10755.68` (step 0)
- end_train_loss: `293.81` (step 19900)
- loss_one_liner: Stage-1 bottleneck loss dropped from ~10.8k to ~294 and flattened in the last 1k steps.
- checkpoint: `/home/user/openpi/checkpoints/pi05_rlt_busybox_multitask_singlearm_minmax/busybox_multitask_rlt_singlearm_minmax/19999/`

## W&B
- project: `busybox_multitask_rlt_singlearm_minmax`
- entity: `pravsels`
- run: https://wandb.ai/pravsels/busybox_multitask_rlt_singlearm_minmax/runs/93keszgb
- run id: `93keszgb`
- synced: online
- local: `/workspace/repo/wandb/run-20260903_121621-93keszgb`

## HuggingFace
- repo: https://huggingface.co/pravsels/pi05_rlt_busybox_multitask_singlearm_minmax
- uploaded checkpoints: step 19999, `params/` + `assets/` at repo root
- includes: README, TRAINING_LOG, CHECKPOINT_MANIFEST.json, assets, params
- excludes: `train_state/`
- manifest sha256: `eab2c46fc0d6bdf51268132bece95b431044ec227055997bac45e7e37db8e98e`

## Next
- delete VM `openpi-rlt-busybox-multitask-minmax` after you confirm the Hub tree
