# Train — RLT Stage 1 BusyBox multitask single-arm minmax (GCloud)

## Mode
- run_type: replication
- objective: train the RL-token bottleneck on the frozen prompt-fix π0.5 minmax multitask checkpoint so `hw_control.pi0_rlt` can extract tokens
- status: training (step 0 logged)

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
- submitted/start: `2026-09-03T12:16:19Z` (train.py)
- start_human: Thursday, Sep 3rd, 2026
- end:
- end_human:

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

## Results

## W&B
- project: `busybox_multitask_rlt_singlearm_minmax`
- entity: `pravsels`
- run: https://wandb.ai/pravsels/busybox_multitask_rlt_singlearm_minmax/runs/93keszgb
- run id: `93keszgb`
- synced: online
- local: `/workspace/repo/wandb/run-20260903_121621-93keszgb`

## Next
- monitor to 20k; do not start eval/publish in this slice
