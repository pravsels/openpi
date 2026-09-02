# Train — π0.5 BusyBox multitask minmax prompt-fix (Vast)

## Mode
- run_type: full-component fine-tune
- objective: retrain π0.5 on `villekuosmanen/busybox_multitask` with remapped per-frame prompts, relative actions (5 joints delta, gripper absolute), and per-timestep min/max bounds in `q01`/`q99`
- status: completed (`train_done`); published to Hub; Vast instance destroyed

## Config
- config: `pi05_busybox_multitask_minmax`
- exp_name: `pi05_busybox_multitask_minmax`
- dataset: `villekuosmanen/busybox_multitask` (LeRobot v3, 66 episodes, 12141 frames, 20 fps, 27 tasks)
- prompt: `prompt_from_task` (RCW `main` remaps `task_index` onto sorted `meta.tasks`)
- cameras: `top` → `base_0_rgb`, `wrist` → `left_wrist_0_rgb`, `front` → `base_1_rgb`
- key settings: π0.5, full-component fine-tuning, action horizon 30, global batch 32, 30k steps, save every 5k and retain only the latest checkpoint, cosine LR 2.5e-5 → 2.5e-6, EMA 0.999, relative actions (5 joints delta, gripper absolute), per-timestep min/max bounds mapped to `[-1, 1]`
- parallelism: 4-GPU full data parallel (`fsdp_devices=1`)
- input pipeline: TorchCodec, 8 persistent workers
- memory: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`
- publishing: each finalized checkpoint replaces the Hugging Face repo root; `train_state` is excluded
- init: `weights/pi05_base/params`
- code: `84a93adf9000be334dc9b6255f67a86610335d38` on `task/busybox_multitask_promptfix`
- RCW: git `main` lock `597aa9ad21176e7f7dcee4aede5dc1ffc07eacee`
- plan: `docs/plans/2026-09-02-busybox-multitask-promptfix-rerun.md`

## Infrastructure
- provider: Vast
- instance: `49646082` (`pi05-busybox-minmax-promptfix`)
- offer: `45686955` 4× H200 141GB NVLink, Japan, billed ~$18.63/hr with 1500 GB disk (no 4× H100 SXM left; relative run holds the Taiwan H100)
- image: `nvcr.io/nvidia/pytorch:25.03-py3`
- SSH: `ssh -i ~/.ssh/id_ed25519 -p 24311 root@210.157.233.86` (proxy `ssh3.vast.ai:16082`)
- launch: `REPO_REF=task/busybox_multitask_promptfix bash vast/bootstrap.sh` then on-box prompt check then `CONFIG_NAME=pi05_busybox_multitask_minmax WANDB_PROJECT=busybox_multitask_pi05_minmax nohup bash vast/train.sh` (no `SMOKE=1`)
- train log: `/workspace/vast_runs/openpi/logs/train_30k.log`
- bootstrap log: `/workspace/vast_runs/openpi/logs/bootstrap.log`
- checkpoints: `/workspace/openpi/checkpoints/pi05_busybox_multitask_minmax/pi05_busybox_multitask_minmax/`
- assets: `/root/openpi_runs/pi05_busybox_multitask_minmax/pi05_busybox_multitask_minmax/assets`
- do not reuse relative-run or shuffled-language assets; do not touch Vast `49643702`

## Job
- execution_id: Vast `49646082`
- submitted/start: `2026-09-02T13:09:34Z` (train start)
- start_human: Wednesday, Sep 2nd, 2026
- end: `2026-09-02T17:04:14Z` (`train_done` / Hub 29999)
- end_human: Wednesday, Sep 2nd, 2026

## Status
- 2026-09-02 — relative 30k already running on Taiwan 4× H100 SXM `49643702`. No second H100 SXM on the market.
- 2026-09-02 12:48 UTC — rented Vast `49646082` (`pi05-busybox-minmax-promptfix`), Japan 4× H200, `$18.63/hr`, 1500 GB, image `nvcr.io/nvidia/pytorch:25.03-py3`.
- 2026-09-02 12:57 UTC — SSH up (direct `210.157.233.86:24311`) after attaching `id_ed25519`. 4× H200. Secrets copied. Bootstrap started with `REPO_REF=task/busybox_multitask_promptfix`.
- 2026-09-02 12:59 UTC — `setup_done`. Staged `pi0_base` and `pi05_base`. On-box RCW git `main` `597aa9ad` (`rcw_sha_ok`). Clone `84a93adf9000`.
- 2026-09-02 13:05 UTC — on-box prompt check: `wrapped_tasks 27`, `mismatches 0`, `prompt_ok`. Index 0 is `Move the left slider to position 1`.
- 2026-09-02 13:05 UTC — `CONFIG_NAME=pi05_busybox_multitask_minmax WANDB_PROJECT=busybox_multitask_pi05_minmax nohup bash vast/train.sh` started (pid 11613). Fresh minmax norm stats (isolated assets dir).
- 2026-09-02 13:09 UTC — JAX `device_count=4`. `use_min_max_norm_stats=True`. W&B `swjv9hbs`.
- 2026-09-02 13:11 UTC — first-step XLA gemm autotune mismatch (same as prior 30k; training continued). Step 0: `loss=0.2058`, `grad_norm=2.6192`, `param_norm=1802.3865`.
- 2026-09-02 13:12 UTC — ~2.2 it/s after compile (~3h45m remaining). Step 100: `loss=0.1158`.
- 2026-09-02 13:51–17:04 UTC — published Hub 5k / 10k / 15k / 20k / 25k / 29999.
- 2026-09-02 17:04 UTC — `train_done`. GPUs idle. Instance still billed ~$18.63/hr.
- 2026-09-02 18:50 UTC — uploaded prompt-fix model card to Hub `README.md` (W&B `swjv9hbs`, code `84a93ad`). Destroyed Vast `49646082`.

## Results
- runtime: ~3h 55m (train start `2026-09-02T13:09:34+00:00`, Hub 29999 / `train_done` `2026-09-02T17:04:14+00:00`)
- final step: 29999
- start_train_loss: `0.2058` (step 0)
- end_train_loss: `0.0019` (step 29900)
- local checkpoint: `.../pi05_busybox_multitask_minmax/pi05_busybox_multitask_minmax/29999/` (`keep_period=None`)
- assets: `norm_stats.json`, `norm_stats_actions_per_timestep.json`, `valid_indices.json`

## W&B
- project: `busybox_multitask_pi05_minmax`
- run: https://wandb.ai/pravsels/busybox_multitask_pi05_minmax/runs/swjv9hbs
- run id: `swjv9hbs`
- Hub: `pravsels/pi05_busybox_multitask_minmax`
- synced: online

## HuggingFace
- Hub: https://huggingface.co/pravsels/pi05_busybox_multitask_minmax
- published: repo root replaced at 5k, 10k, 15k, 20k, 25k, and 29999 (`train_state` excluded)
- uploaded: 2026-09-02 17:04 UTC (final); model card 2026-09-02 18:50 UTC

## Next
- Vast `49646082` destroyed
- eval / RLT later; do not start in this slice
