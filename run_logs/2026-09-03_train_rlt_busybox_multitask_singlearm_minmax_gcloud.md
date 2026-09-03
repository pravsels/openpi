# Train — RLT Stage 1 BusyBox multitask single-arm minmax (GCloud)

## Mode
- run_type: replication
- objective: train the RL-token bottleneck on the frozen prompt-fix π0.5 minmax multitask checkpoint
- status: launching

## Config
- script: `slurm/train_busybox_multitask_singlearm_minmax_rlt_gcloud.sh`
- config: `pi05_rlt_busybox_multitask_singlearm_minmax`
- exp_name: `busybox_multitask_rlt_singlearm_minmax`
- dataset: `villekuosmanen/busybox_multitask` (LeRobot v3, 66 episodes, 12141 frames, 20 fps, 27 tasks)
- prompt: `prompt_from_task` (RCW git `main` `#597aa9ad`, remap `de4e4eb`)
- cameras: `top` → `base_0_rgb`, `wrist` → `left_wrist_0_rgb`, `front` → `base_1_rgb`
- key settings: same Stage-1 knobs as the 1%/99% single-arm RLT; per-timestep min/max in `q01`/`q99`; Hub assets copied, not recomputed
- init: Hub [`pravsels/pi05_busybox_multitask_minmax`](https://huggingface.co/pravsels/pi05_busybox_multitask_minmax) (W&B `swjv9hbs`)
- code: pending push on `task/rlt_busybox_multitask`
- RCW: git `main` lock `597aa9ad21176e7f7dcee4aede5dc1ffc07eacee`

## Infrastructure
- provider: GCloud
- project: `gen-lang-client-0388971498`
- vm: pending fresh `a2-ultragpu-1g` (do not reuse `openpi-rlt-busybox-multitask`)
- hardware: 1× NVIDIA A100-SXM4-80GB
- docker: build `openpi:latest` on the VM from this branch
- repo_on_vm: git clone (no scp/rsync)

## Job
- execution_id: pending
- start_human: Thursday, Sep 3rd, 2026

## Status
- 2026-09-03 — relative single-arm RLT already running on `openpi-rlt-busybox-multitask`. This run is a separate VM.

## W&B
- project: `busybox_multitask_rlt_singlearm_minmax`
- entity: `pravsels`

## Next
- create VM, git clone, docker build, launch
