# Train — π0 BusyBox multitask (Vast)

## Mode
- run_type: full-component fine-tune
- objective: fine-tune π0 on `armnet/busybox_multitask` for 20,000 steps (direct run, no smoke)
- status: completed

## Config
- config: `pi0_busybox_multitask`
- exp_name: `pi0_busybox_multitask`
- dataset: `armnet/busybox_multitask` (LeRobot v3, 132 episodes, 25,734 frames, 20 fps, 27 tasks, no single instruction)
- prompt: `prompt_from_task` (RCW `597aa9ad` remaps `task_index` onto sorted `meta.tasks`)
- cameras: `top` → `base_0_rgb`, `wrist` → `left_wrist_0_rgb`, `front` → `base_1_rgb`
- key settings: π0 (`pi05=False`), full-component fine-tune, action horizon 30, batch 8, 20k steps, save every 5k, `keep_period=None` (local `max_to_keep=1`), cosine LR 2.5e-5 → 2.5e-6 over 20k with 1k warmup, EMA 0.999, delta actions
- parallelism: 1 GPU (`fsdp_devices=1`, `REQUIRE_JAX_DEVICES=1`)
- input pipeline: TorchCodec, 8 workers
- memory: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`
- publishing: `HF_CHECKPOINT_LAYOUT=subfolder` uploads `checkpoint-<step>/` to `armnet/pi0_busybox_multitask`; `train_state` excluded; prior Hub checkpoints kept
- init: `weights/pi0_base/params`
- code: `7679e16c4870701cea3c528252a36f4db7c50b28` on `task/multitask_busybox_pi0`
- RCW: git `main` lock `597aa9ad21176e7f7dcee4aede5dc1ffc07eacee` (v0.2.18)
- plan: `alpha-robotics/docs/plans/2026-10-24-busybox-multitask-finetunes-design.md`

## Infrastructure
- provider: Vast
- instance: `57658382` (`pi0-busybox-multitask`)
- offer: `57657671` 1× H100 NVL 94 GB, Virginia, whole machine, billed ~$3.11/hr with 296 GB disk
- image: `nvcr.io/nvidia/pytorch:25.03-py3`
- clone: `/workspace/openpi` at `task/multitask_busybox_pi0`
- launch: `EXPECTED_GPUS=1 REPO_REF=task/multitask_busybox_pi0 bash vast/bootstrap.sh` then `nohup bash vast/train.sh` (no `SMOKE=1`)
- train log: `/workspace/vast_runs/openpi/logs/train.log`
- bootstrap log: `/workspace/vast_runs/openpi/logs/bootstrap.log`
- checkpoints: `/workspace/openpi/checkpoints/pi0_busybox_multitask/pi0_busybox_multitask/`
- assets: `/root/openpi_runs/pi0_busybox_multitask/pi0_busybox_multitask/assets`
- SSH: `ssh -i ~/.ssh/id_ed25519 -p 27404 root@ssh5.vast.ai`
- do not `uv lock --upgrade` (that would drop the RCW pin; `pyproject.toml` says `rev = "main"`)

## Job
- execution_id: Vast `57658382`, train PID `13445`, publisher PID `13444`
- submitted/start: `2026-10-25T11:57:22Z` (train section; step 0 logged `2026-10-25T11:59:02Z`)
- start_human: Sunday, Oct 25th, 2026
- end: 2026-10-25T14:54:40Z
- runtime: 2 h 55 m (step 0 at 11:59 UTC, final publish 14:54 UTC)
- state: COMPLETED
- W&B: https://wandb.ai/pravsels/busybox_multitask_pi0/runs/3yuw4r3l
- Hugging Face: https://huggingface.co/armnet/pi0_busybox_multitask (first upload at step 5000)

## Status
- 2026-10-25 11:39 UTC — rented `57658382` from offer `57657671`. No whole-machine H100 was under the stricter disk filter; this Virginia NVL (94 GB, `gpu_frac=1`) was the next whole machine.
- 2026-10-25 11:47 UTC — SSH up on `ssh5.vast.ai:27404`. 1× H100 NVL, 95830 MiB. Secrets at `/workspace/secrets/{hf_token,wandb_token,github_token}`. Clone `7679e16` on `task/multitask_busybox_pi0`.
- 2026-10-25 11:47 UTC — bootstrap started (`EXPECTED_GPUS=1`, `REPO_REF=task/multitask_busybox_pi0`).
- 2026-10-25 11:48 UTC — `uv sync` installed RCW `0.2.18` from git `597aa9ad21176e7f7dcee4aede5dc1ffc07eacee`. `branch_assets_ok`.
- 2026-10-25 11:50 UTC — `setup_done`. Staged `pi0_base` and `pi05_base`.
- 2026-10-25 11:50 UTC — `nohup bash vast/train.sh` (pid 11616). Fresh norm stats, not a reused asset dir.
- 2026-10-25 11:57 UTC — train process up. JAX `device_count=1`, `local_batch_size=8`. W&B `3yuw4r3l`. Publisher running with `--layout=subfolder`.
- 2026-10-25 11:58 UTC — XLA rematerialization warning: could not shrink the step below ~62 GiB (down from 62.3 GiB). It still fits.
- 2026-10-25 11:59 UTC — step 0: `loss=0.4257`, `grad_norm=11.9115`, `param_norm=1377.8652`.
- 2026-10-25 12:02 UTC — step **400**, `loss=0.3608`, `grad_norm=6.0072`. Progress **461 / 20000** at **1.9 it/s**. GPU **91 GB / 94 GB**, utilization 100%.
- 2026-10-25 14:54 UTC — **completed** 20,000 steps in 2 h 54 m. Last logged step 19,900: loss **0.0967**, grad_norm 1.983. Local checkpoint `19999` saved. GPU idle.
- 2026-10-25 14:54 UTC — published `checkpoint-5000/` `checkpoint-10000/` `checkpoint-15000/` `checkpoint-19999/` to [armnet/pi0_busybox_multitask](https://huggingface.co/armnet/pi0_busybox_multitask). Final step index is 19999, not 20000. `train_state` excluded. Instance later destroyed.

## W&B
- project: `busybox_multitask_pi0`
- run: https://wandb.ai/pravsels/busybox_multitask_pi0/runs/3yuw4r3l
- run id: `3yuw4r3l`
- Hub: `armnet/pi0_busybox_multitask`
- synced: online

## HuggingFace
- Hub: https://huggingface.co/armnet/pi0_busybox_multitask
- publish cadence: `checkpoint-5000/` `checkpoint-10000/` `checkpoint-15000/` `checkpoint-19999/` (`train_state` excluded)
- uploaded: all four. `checkpoint-19999/` is the final step.

## Next
1. None. Instance `57658382` is destroyed and the final checkpoint is on the Hub.
