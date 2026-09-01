# Train — π0.5 BusyBox multitask absolute actions

## Mode
- run_type: full-component fine-tune
- objective: train π0.5 on `villekuosmanen/busybox_multitask` with absolute 6D actions, true min/max normalization, and per-episode task prompts
- status: not launched

## Config
- config: `pi05_busybox_multitask_abs`
- exp_name: `pi05_busybox_multitask_abs`
- dataset: `villekuosmanen/busybox_multitask` (LeRobot v3, 66 episodes, 12141 frames, 20 fps, 27 tasks)
- prompt: `prompt_from_task` (no single default)
- cameras: `top` → `base_0_rgb`, `wrist` → `left_wrist_0_rgb`, `front` → `base_1_rgb`
- key settings: π0.5, full-component fine-tuning, action horizon 30, global batch 32, 30k steps, save every 5k and retain only the latest checkpoint, cosine LR 2.5e-5 → 2.5e-6, EMA 0.999, absolute 6D actions, per-timestep min/max bounds mapped to `[-1, 1]`
- parallelism: `fsdp_devices=1`
- input pipeline: TorchCodec, 8 workers
- init: `weights/pi05_base/params`
- plan: `docs/plans/2026-09-01-pi05-busybox-multitask-abs.md`

## Infrastructure
- provider: not selected
- instance: none
- launch: not launched
- checkpoints: not created
- assets: separate config-scoped directory required; do not reuse `pi05_busybox_multitask` assets

## Status
- 2026-09-01 — Variant 2 config and true min/max normalization wiring prepared locally.
- 2026-09-01 — Training not launched.

## W&B
- project: `busybox_multitask_pi05_abs`
- run: none
- run id: none

## HuggingFace
- Hub (later): https://huggingface.co/pravsels/pi05_busybox_multitask_abs
- publish status: not launched

## Next
- review and choose infrastructure before computing assets or launching training
- run a smoke gate before any full 30k launch
- keep the relative run `pi05_busybox_multitask` and its assets unchanged
