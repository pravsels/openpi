# Train — RLT Stage 1 BusyBox multitask single-arm (GCloud)

## Mode
- run_type: replication
- objective: train the RL-token bottleneck on the frozen prompt-fix π0.5 multitask checkpoint so `hw_control.pi0_rlt` can extract tokens
- status: completed (exit 0); published to Hub

## Config
- script: `slurm/train_busybox_multitask_singlearm_rlt_gcloud.sh`
- config: `pi05_rlt_busybox_multitask_singlearm`
- exp_name: `busybox_multitask_rlt_singlearm`
- dataset: `villekuosmanen/busybox_multitask` (LeRobot v3, 66 episodes, 12141 frames, 20 fps, 27 tasks)
- prompt: `prompt_from_task` (RCW git `main` `#597aa9ad`, remap `de4e4eb`)
- cameras: `top` → `base_0_rgb`, `wrist` → `left_wrist_0_rgb`, `front` → `base_1_rgb`
- key settings: π0.5 + RLT, VLA frozen (`rl_vla_loss_weight=0.0` + `get_rl_freeze_filter()`), encoder/decoder 2 layers / 8 heads / dim 2048, action horizon 30, 6D (5 joints delta, gripper absolute), per-timestep 1%/99%, global batch 16, 20k steps, save once at the end, cosine LR 5e-5 with 1k warmup, EMA 0.999
- parallelism: 1-GPU (`fsdp_devices=1`)
- input pipeline: TorchCodec, 8 persistent workers
- memory: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`
- init: Hub [`pravsels/pi05_busybox_multitask`](https://huggingface.co/pravsels/pi05_busybox_multitask) (W&B `4ym0qegc`); Hub `assets/` copied, not recomputed
- code: `29fb3dc` on `task/rlt_busybox_multitask`
- RCW: git `main` lock `597aa9ad21176e7f7dcee4aede5dc1ffc07eacee`
- plan: `docs/plans/2026-09-03-rlt-busybox-multitask-singlearm-design.md`

## Infrastructure
- provider: GCloud
- project: `gen-lang-client-0388971498`
- vm: `openpi-rlt-busybox-multitask` / `us-central1-c`
- ssh: `gcloud compute ssh openpi-rlt-busybox-multitask --zone=us-central1-c`
- repo_on_vm: `/home/user/openpi` (`git clone --branch task/rlt_busybox_multitask`)
- hardware: 1× NVIDIA A100-SXM4-80GB (`a2-ultragpu-1g`; do not use `a2-highgpu-1g` 40 GB)
- docker: build `openpi:latest` on the VM from this branch
- os_image: `ubuntu-accelerator-2204-amd64-with-nvidia-580`
- disk: 1 TB pd-ssd
- do not reuse shuffled-language Hub trees; do not use GMAN

## Job
- execution_id: `openpi-rlt-busybox-multitask` / `us-central1-c`
- submitted/start: `2026-09-03T10:46:00Z` (launcher)
- start_human: Thursday, Sep 3rd, 2026
- end: `2026-09-03T14:56:06Z`
- end_human: Thursday, Sep 3rd, 2026
- runtime: `4h 10m 6s`

## Status
- 2026-09-03 10:11 UTC — committed and pushed `29fb3dc` on `task/rlt_busybox_multitask`.
- 2026-09-03 10:11 UTC — local `gcloud` tokens expired; reauth completed by user.
- 2026-09-03 10:16 UTC — created `openpi-rlt-busybox-multitask` (`a2-ultragpu-1g`, 1 TB pd-ssd, `ubuntu-accelerator-2204-amd64-with-nvidia-580`, nat 34.173.157.3).
- 2026-09-03 10:19 UTC — SSH up. `user`, A100-SXM4-80GB, rootfs 993G / 1% used.
- 2026-09-03 10:20 UTC — `git clone --branch task/rlt_busybox_multitask` → `29fb3dc` at `/home/user/openpi`.
- 2026-09-03 10:21 UTC — Docker 29.7.2 + nvidia-container-toolkit. `docker build -t openpi:latest` started (`logs/docker_build.log`).
- 2026-09-03 10:27 UTC — `openpi:latest` built (17.7 GB).
- 2026-09-03 10:44 UTC — launcher started. Hub VLA downloaded. `uv sync` installed RCW git `main` `#597aa9ad` (`rcw_sha_ok`).
- 2026-09-03 10:45 UTC — `prompt_ok`: `wrapped_tasks 27`, `mismatches 0`, index 0 is `Move the left slider to position 1`.
- 2026-09-03 10:46 UTC — train.py up. JAX `device_count=1`, `CudaDevice(id=0)`. W&B `xmkdxvrl`. Hub assets + 12141 valid indices. `prompt_from_task=True`.
- 2026-09-03 10:51 UTC — step 0: `loss=12117.80`, `grad_norm=38526.98`, `param_norm=1836.52`. GPU 100%, 77.6/80.0 GiB.
- 2026-09-03 10:52 UTC — 86/20000, ~1.4 it/s, ETA ~4h. GPU 100%, 77.6/80.0 GiB.
- 2026-09-03 12:21 UTC — step 7400: `loss=387.69`, `grad_norm=1175.97`, `param_norm=1836.89`.
- 2026-09-03 12:22 UTC — step 7500: `loss=386.30`, `grad_norm=1141.76`, `param_norm=1836.89`. 7.57k/20000, ~1.4 it/s, ETA ~2.5h. GPU 100%, 77.6/80.0 GiB. No errors.
- 2026-09-03 14:43 UTC — step 19000: `loss=249.69`, `grad_norm=727.33`, `param_norm=1838.05`. 19.0k/20000, ~1.4 it/s, ETA ~12m. GPU 100%, 77.6/80.0 GiB. No errors.
- 2026-09-03 14:55 UTC — step 19900: `loss=246.70`, `grad_norm=738.00`, `param_norm=1838.16`. Orbax save of `19999` started.
- 2026-09-03 14:56 UTC — checkpoint finalized at `19999` (`params/` + `assets/` + `train_state/`). Launcher exit 0. GPU idle.
- 2026-09-03 15:32 UTC — published step 19999 to Hub `pravsels/pi05_rlt_busybox_multitask_singlearm` (`params/` + `assets/` + README; no `train_state/`).
- 2026-09-03 15:53 UTC — deleted VM `openpi-rlt-busybox-multitask`.

## Results
- runtime: `4:10:06` (start `2026-09-03T10:46:00Z`, end `2026-09-03T14:56:06Z`)
- final step: 19999
- start_train_loss: `12117.80` (step 0)
- end_train_loss: `246.70` (step 19900)
- loss_one_liner: Stage-1 bottleneck loss dropped from ~12k to ~247 and flattened in the last 1k steps.
- checkpoint: `/home/user/openpi/checkpoints/pi05_rlt_busybox_multitask_singlearm/busybox_multitask_rlt_singlearm/19999/`

## W&B
- project: `busybox_multitask_rlt_singlearm`
- entity: `pravsels`
- run: https://wandb.ai/pravsels/busybox_multitask_rlt_singlearm/runs/xmkdxvrl
- run id: `xmkdxvrl`
- synced: online
- local: `/workspace/repo/wandb/run-20260903_104616-xmkdxvrl`

## HuggingFace
- repo: https://huggingface.co/pravsels/pi05_rlt_busybox_multitask_singlearm
- uploaded checkpoints: step 19999, `params/` + `assets/` at repo root
- includes: README, TRAINING_LOG, CHECKPOINT_MANIFEST.json, assets, params
- excludes: `train_state/`
- manifest sha256: `84496d0ff720df1bfed125d56759c7dfa3d0e7e10e81de8af7d6aa61b6a5d433`

## Next
- wait for the minmax sibling to finish and publish before tearing that box down
