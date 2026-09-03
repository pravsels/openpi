---
tags:
- pi05
- robotics
- openpi
- rlt
- rl-token
datasets:
- villekuosmanen/busybox_multitask
---

# pi05_rlt_busybox_multitask_singlearm_minmax

RL-token encoder/decoder (RLT Stage 1) on the frozen prompt-fix [π0.5](https://github.com/Physical-Intelligence/openpi) minmax multitask checkpoint. This is what `hw_control.pi0_rlt` loads to extract tokens.

Same relative 6D recipe as the VLA ([pravsels/pi05_busybox_multitask_minmax](https://huggingface.co/pravsels/pi05_busybox_multitask_minmax)): 5 joints delta, gripper absolute, per-timestep min/max in `q01`/`q99`. Prompts are remapped per frame (`prompt_from_task`, RCW git `main` `#597aa9ad`).

| | |
|---|---|
| **Policy** | π0.5 + RLT (`Pi0RLConfig`, `pi05=true`) |
| **Frozen VLA** | [`pravsels/pi05_busybox_multitask_minmax`](https://huggingface.co/pravsels/pi05_busybox_multitask_minmax) (W&B `swjv9hbs`) |
| **Dataset** | [villekuosmanen/busybox_multitask](https://huggingface.co/datasets/villekuosmanen/busybox_multitask) (66 episodes, 27 tasks) |
| **Prompt** | `prompt_from_task` (27 remapped instructions; no single default) |
| **Action dim** | 6 (single-arm; 5 joints delta, gripper absolute) |
| **Cameras** | `top` → `base_0_rgb`, `wrist` → `left_wrist_0_rgb`, `front` → `base_1_rgb` |
| **Training** | GCloud 1× A100 80GB, VLA frozen (`rl_vla_loss_weight=0.0`), 20k steps, global batch 16, action horizon 30 |
| **Norm** | per-timestep min/max in `q01`/`q99` (Hub assets copied, not recomputed) |
| **W&B project** | [busybox_multitask_rlt_singlearm_minmax](https://wandb.ai/pravsels/busybox_multitask_rlt_singlearm_minmax) |
| **W&B run** | [93keszgb](https://wandb.ai/pravsels/busybox_multitask_rlt_singlearm_minmax/runs/93keszgb) |

## Checkpoints

`main` is the finalized 20k checkpoint (`params/` + `assets/` at the repo root). `train_state/` is not published.

| Field | Value |
|------|--------|
| Latest published | `main` (`params/`, step 19999) |
| Config | `pi05_rlt_busybox_multitask_singlearm_minmax` |
| Code | `90490b9` on `task/rlt_busybox_multitask` |
| Runtime | 4h 13m (2026-09-03 12:16 UTC → 16:28 UTC) |
| Train loss | 10755.68 (step 0) → 293.81 (step 19900) |

## Usage

```python
from huggingface_hub import snapshot_download
from openpi.policies import policy_config
from openpi.training.config import get_config

ckpt = snapshot_download("pravsels/pi05_rlt_busybox_multitask_singlearm_minmax")
policy = policy_config.create_trained_policy(
    get_config("pi05_rlt_busybox_multitask_singlearm_minmax"),
    ckpt,
)
```

```bash
uv run scripts/serve_policy.py \
  --policy.config=pi05_rlt_busybox_multitask_singlearm_minmax \
  --policy.dir=/path/to/pravsels/pi05_rlt_busybox_multitask_singlearm_minmax
```
