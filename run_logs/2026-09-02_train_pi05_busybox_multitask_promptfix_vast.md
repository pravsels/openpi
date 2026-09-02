# Train — π0.5 BusyBox multitask relative prompt-fix (Vast)

## Mode
- run_type: full-component fine-tune
- objective: retrain π0.5 on `villekuosmanen/busybox_multitask` with remapped per-frame prompts, relative actions (5 joints delta, gripper absolute), and per-timestep 1%/99% action norm
- status: completed (`train_done`); published to Hub; Vast instance destroyed

## Config
- config: `pi05_busybox_multitask`
- exp_name: `pi05_busybox_multitask`
- dataset: `villekuosmanen/busybox_multitask` (LeRobot v3, 66 episodes, 12141 frames, 20 fps, 27 tasks)
- prompt: `prompt_from_task` (RCW `main` remaps `task_index` onto sorted `meta.tasks`)
- cameras: `top` → `base_0_rgb`, `wrist` → `left_wrist_0_rgb`, `front` → `base_1_rgb`
- key settings: π0.5, full-component fine-tuning, action horizon 30, global batch 32, 30k steps, save every 5k and retain only the latest checkpoint, cosine LR 2.5e-5 → 2.5e-6, EMA 0.999, relative actions (5 joints delta, gripper absolute), per-timestep 1%/99%
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
- instance: `49643702` (`pi05-busybox-rel-promptfix`)
- offer: `49467276` 4× H100 SXM 80GB HBM3, Taiwan, billed ~$13.53/hr with 1500 GB disk
- image: `nvcr.io/nvidia/pytorch:25.03-py3`
- clone: `/workspace/openpi` at `task/busybox_multitask_promptfix`
- launch: `REPO_REF=task/busybox_multitask_promptfix bash vast/bootstrap.sh` then on-box prompt check then `nohup bash vast/train.sh` (no `SMOKE=1`)
- train log: `/workspace/vast_runs/openpi/logs/train_30k.log`
- bootstrap log: `/workspace/vast_runs/openpi/logs/bootstrap.log`
- checkpoints: `/workspace/openpi/checkpoints/pi05_busybox_multitask/pi05_busybox_multitask/`
- assets: `/root/openpi_runs/pi05_busybox_multitask/pi05_busybox_multitask/assets`
- do not reuse shuffled-language assets; do not use GMAN

## Job
- execution_id: Vast `49643702`
- submitted/start: `2026-09-02T12:40:12Z` (train start)
- start_human: Wednesday, Sep 2nd, 2026
- end: `2026-09-02T17:10:03Z` (`train_done` / Hub 29999)
- end_human: Wednesday, Sep 2nd, 2026

## Status
- 2026-09-02 — local prompt check: `wrapped_tasks 27`, `mismatches 0`, `prompt_ok`. Index 0 is `Move the left slider to position 1`.
- 2026-09-02 — GMAN abandoned; switching to Vast 4× H100 SXM. Only matching offer was Taiwan `49467276`.
- 2026-09-02 12:19 UTC — rented Vast `49643702` (`pi05-busybox-rel-promptfix`), image `nvcr.io/nvidia/pytorch:25.03-py3`, 1500 GB disk. SSH `ssh://root@ssh8.vast.ai:13702`.
- 2026-09-02 12:20 UTC — SSH up. 4× H100 80GB HBM3. Secrets copied to `/workspace/secrets/{hf_token,wandb_token,github_token}`. Bootstrap started with `REPO_REF=task/busybox_multitask_promptfix` (apt ffmpeg/git).
- 2026-09-02 12:23 UTC — local RCW check PASS: lock and `.venv` are git `main` `#597aa9ad` (one commit after remap `de4e4eb`); no PyPI pin.
- 2026-09-02 12:25 UTC — `uv sync` finished. On-box `direct_url.json`: git `main` `597aa9ad21176e7f7dcee4aede5dc1ffc07eacee` (`rcw_sha_ok`). Clone `84a93adf9000` on `task/busybox_multitask_promptfix`. Base weights downloading.
- 2026-09-02 12:34 UTC — `setup_done`. Staged `pi0_base` and `pi05_base`.
- 2026-09-02 12:35 UTC — on-box prompt check: `wrapped_tasks 27`, `inner_tasks 0`, `mismatches 0`, `prompt_ok`. Index 0 is `Move the left slider to position 1`.
- 2026-09-02 12:35 UTC — `nohup bash vast/train.sh` started (pid 12782). Computing fresh norm stats (not reusing shuffled-language assets).
- 2026-09-02 12:40 UTC — train process up. JAX `device_count=4`. Fresh assets written under `/root/openpi_runs/pi05_busybox_multitask/...`. W&B `4ym0qegc`.
- 2026-09-02 12:42 UTC — step 0: `loss=0.2669`, `grad_norm=2.5032`, `param_norm=1802.3865`.
- 2026-09-02 13:12 UTC — ~2.0 it/s at step 3400, `loss=0.0214`. Remaining ~3h45m.
- 2026-09-02 13:34–17:10 UTC — published Hub 5k / 10k / 15k / 20k / 25k / 29999.
- 2026-09-02 17:10 UTC — `train_done`. GPUs idle. Instance still billed ~$13.53/hr.
- 2026-09-02 18:50 UTC — uploaded prompt-fix model card to Hub `README.md` (W&B `4ym0qegc`, code `84a93ad`). Destroyed Vast `49643702`.

## Results
- runtime: ~4h 30m (train start `2026-09-02T12:40:12+00:00`, Hub 29999 / `train_done` `2026-09-02T17:10:03+00:00`)
- final step: 29999
- start_train_loss: `0.2669` (step 0)
- end_train_loss: `0.0030` (step 29900)
- local checkpoint: `.../pi05_busybox_multitask/pi05_busybox_multitask/29999/` (`keep_period=None`)
- assets: `norm_stats.json`, `norm_stats_actions_per_timestep.json`, `valid_indices.json`

## W&B
- project: `busybox_multitask_pi05`
- run: https://wandb.ai/pravsels/busybox_multitask_pi05/runs/4ym0qegc
- run id: `4ym0qegc`
- Hub: `pravsels/pi05_busybox_multitask`
- synced: online
- local: `/workspace/openpi/wandb/run-20260902_124025-4ym0qegc`

## HuggingFace
- Hub: https://huggingface.co/pravsels/pi05_busybox_multitask
- published: repo root replaced at 5k, 10k, 15k, 20k, 25k, and 29999 (`train_state` excluded)
- uploaded: 2026-09-02 17:10 UTC (final); model card 2026-09-02 18:50 UTC

## Next
- Vast `49643702` destroyed
- eval / RLT later; do not start in this slice
