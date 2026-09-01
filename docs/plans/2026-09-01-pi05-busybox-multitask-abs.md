# π0.5 BusyBox Multitask — Variant 2 (absolute)

## Goal

Add `pi05_busybox_multitask_abs`, a second π0.5 full-component fine-tune on
[`villekuosmanen/busybox_multitask`](https://huggingface.co/datasets/villekuosmanen/busybox_multitask).
It keeps the Variant 1 dataset, cameras, task prompts, and train schedule, but
trains on absolute 6D SO101 actions normalized from their observed minima and
maxima to `[-1, 1]`.

The dataset is LeRobot v3: 66 episodes, 12,141 frames at 20 fps, 27 tasks,
single-arm 6D joints, with `top`, `wrist`, and `front` cameras.

## Configuration

| Knob | Value |
|---|---|
| Config | `pi05_busybox_multitask_abs` |
| W&B project | `busybox_multitask_pi05_abs` |
| Hub target (later) | `pravsels/pi05_busybox_multitask_abs` |
| Data factory | `LeRobotSO101ThreeCamDataConfig` |
| Prompt | `prompt_from_task=True`, `default_prompt=None` |
| Actions | absolute 6D (`use_delta_actions=False`) |
| Model | `Pi0Config(pi05=True, action_horizon=30, image_keys=SO101_THREE_CAM_IMAGE_KEYS)` |
| Initialization | `weights/pi05_base/params` |
| Schedule | 30,000 steps; save every 5,000; `keep_period=None` |
| Batch / devices | batch 32; `fsdp_devices=1` |
| Input pipeline | TorchCodec; 8 workers |
| Learning rate | cosine, 1,000 warmup, 2.5e-5 to 2.5e-6 over 30,000 steps |
| EMA | 0.999 |

Cameras map as follows: `top` → `base_0_rgb`, `wrist` →
`left_wrist_0_rgb`, and `front` → `base_1_rgb`.

## Min/max normalization

π0.5 uses the quantile normalization transform. That transform reads fields
named `q01` and `q99` and maps their values to `-1` and `1`; the names are
historical and do not require the stored bounds to be the 1st and 99th
percentiles.

`RunningStats.get_statistics()` keeps the repository-wide default of
histogram-based 1%/99% bounds. When `use_min_max=True`, it instead writes the
exact tracked `_min` and `_max` values into `q01` and `q99`. The absolute config
opts into this through `DataConfig.use_min_max_norm_stats=True`.
`compute_norm_stats.py` and `compute_norm_stats_per_timestep.py` propagate that
choice for global state/action stats and per-timestep action stats. Per-timestep
action normalization is explicitly enabled because the ThreeCam factory only
auto-enables it for delta actions.

## Asset isolation

No `asset_id` override is needed. Both compute scripts save at the config's
assets root, and both loaders read that same root when `asset_id=None`.
The distinct config name gives this variant its own default assets directory;
the GMAN/Vast wrapper convention further isolates it under
`${HOME}/openpi_runs/pi05_busybox_multitask_abs/<exp>/assets`. It therefore
cannot reuse the relative run's transformed-action statistics.

## Out of scope

- Launching or smoke-testing the absolute run on Vast or any other GPU provider.
- Changing `vast/train.sh` defaults or interrupting the live relative 30k run.
- Changing `pi05_busybox_multitask`, `pi05_rlt_busybox_multitask`, or any
  green-button config.
- Implementing an RLT absolute-action variant.
