# π0.5 BusyBox Multitask — Variant 1 (relative)

**Goal:** Rewrite `pi05_busybox_multitask` so it trains π0.5 on the current single-arm [`villekuosmanen/busybox_multitask`](https://huggingface.co/datasets/villekuosmanen/busybox_multitask) (66 eps, 12 141 frames, 27 tasks, 6D, `top`/`wrist`/`front`) using the green-button relative-action recipe.

**Where:** This repo. Config work first. Launch later on Vast (not GMAN/Isambard); Vast scripts are not part of this slice.

**Out of scope for this slice:** Variant 2 (absolute + min/max). `pi05_rlt_busybox_multitask`. 10-step smoke. Hub publish. Changing green-button configs.

---

## Dataset vs existing config

Hub `info.json` is single-arm SO101, 6D, cameras `front`/`top`/`wrist`. The current `pi05_busybox_multitask` still uses `LeRobotSO101BimanualDataConfig` (12D, 10k steps). That is wrong for this dataset. Rewrite in place; keep the name and W&B project `busybox_multitask_pi05`.

Isambard/Modal/slurm still mention this config as bimanual 10k. Comment-only so nobody sbatches the old recipe by accident. Do not retarget those launchers.

---

## Variant 1 knobs (now)

Clone `pi05_busybox_push_green_button`, except prompts come from the dataset.

| Knob | Value |
|---|---|
| Config | `pi05_busybox_multitask` |
| Data | `LeRobotSO101ThreeCamDataConfig` |
| Repo | `villekuosmanen/busybox_multitask` |
| Prompt | `prompt_from_task=True` (27 instructions; no single `default_prompt`) |
| Actions | relative: `use_delta_actions=True` (5 joints delta, gripper absolute), per-timestep quantile |
| Model | `Pi0Config(pi05=True, action_horizon=30, image_keys=SO101_THREE_CAM_IMAGE_KEYS)` |
| Init | `weights/pi05_base/params` |
| Steps | 30 000, save every 5 000, `keep_period=None` |
| Batch | 32 global, `fsdp_devices=1`, 8 workers, TorchCodec |
| LR | cosine 1k warmup, 2.5e-5 → 2.5e-6, `decay_steps=30_000` |
| Hub (later) | `pravsels/pi05_busybox_multitask` |

Cameras: `top` → `base_0_rgb`, `wrist` → `left_wrist_0_rgb`, `front` → `base_1_rgb`.

---

## Prompt passthrough (required)

`PromptFromLeRobotTask` injects `prompt` before repack. Bimanual `create()` already copies `"prompt"` into `RepackTransform` when `prompt_from_task` is set. Three-cam does not — every sample would fall back to `"push the green button"`.

Port that passthrough into `LeRobotSO101ThreeCamDataConfig.create()`. Stop using `"push the green button"` as the ThreeCam fallback when `default_prompt` is None (use `"complete the task"`). Green-button configs keep their explicit default.

---

## Tests (CPU)

Add `test_busybox_multitask_pi05_config` next to the green-button config test: three-cam factory, repo id, `use_delta_actions=True`, `prompt_from_task`, 30k / batch 32 / 8 workers / TorchCodec / `pi05_base`.

Add a unit test that ThreeCam `create(..., prompt_from_task=True)` keeps `prompt` in the repack structure.

---

## Implementation order

1. Failing config + repack tests.
2. ThreeCam prompt passthrough + None fallback.
3. Rewrite `pi05_busybox_multitask` TrainConfig + comment. Leave `pi05_rlt_busybox_multitask` as-is (still bimanual; do not train it on this dataset).
4. Comment-only slurm/modal notes.
5. Run log stub `run_logs/2026-09-01_train_pi05_busybox_multitask_vast.md` (status: not launched).

Vast launch (image, 4×H100, bootstrap, 10-step gate) comes after this lands.

---

## Variant 2 (later)

Implemented separately as `pi05_busybox_multitask_abs` with true min/max bounds; see `docs/plans/2026-09-01-pi05-busybox-multitask-abs.md`.
