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

# pi05_rlt_busybox_multitask_singlearm

RL-token encoder/decoder (RLT Stage 1) on the frozen prompt-fix [π0.5](https://github.com/Physical-Intelligence/openpi) BusyBox multitask checkpoint. This is what `hw_control.pi0_rlt` loads to extract tokens.

Same relative 6D recipe as the VLA ([pravsels/pi05_busybox_multitask](https://huggingface.co/pravsels/pi05_busybox_multitask)): 5 joints delta, gripper absolute, per-timestep 1%/99%. Prompts are remapped per frame (`prompt_from_task`, RCW git `main` `#597aa9ad`).

| | |
|---|---|
| **Policy** | π0.5 + RLT (`Pi0RLConfig`, `pi05=true`) |
| **Frozen VLA** | [`pravsels/pi05_busybox_multitask`](https://huggingface.co/pravsels/pi05_busybox_multitask) (W&B `4ym0qegc`) |
| **Dataset** | [villekuosmanen/busybox_multitask](https://huggingface.co/datasets/villekuosmanen/busybox_multitask) (66 episodes, 27 tasks) |
| **Prompt** | `prompt_from_task` (27 remapped instructions; no single default) |
| **Action dim** | 6 (single-arm; 5 joints delta, gripper absolute) |
| **Cameras** | `top` → `base_0_rgb`, `wrist` → `left_wrist_0_rgb`, `front` → `base_1_rgb` |
| **Training** | GCloud 1× A100 80GB, VLA frozen (`rl_vla_loss_weight=0.0`), 20k steps, global batch 16, action horizon 30 |
| **Norm** | per-timestep 1%/99% (Hub assets copied, not recomputed) |
| **W&B project** | [busybox_multitask_rlt_singlearm](https://wandb.ai/pravsels/busybox_multitask_rlt_singlearm) |
| **W&B run** | [xmkdxvrl](https://wandb.ai/pravsels/busybox_multitask_rlt_singlearm/runs/xmkdxvrl) |

## Checkpoints

`main` is the finalized 20k checkpoint (`params/` + `assets/` at the repo root). `train_state/` is not published.

| Field | Value |
|------|--------|
| Latest published | `main` (`params/`, step 19999) |
| Config | `pi05_rlt_busybox_multitask_singlearm` |
| Code | `29fb3dc` on `task/rlt_busybox_multitask` |
| Runtime | 4h 10m (2026-09-03 10:46 UTC → 14:56 UTC) |
| Train loss | 12117.80 (step 0) → 246.70 (step 19900) |

## Usage

```python
from huggingface_hub import snapshot_download
from openpi.policies import policy_config
from openpi.training.config import get_config

ckpt = snapshot_download("pravsels/pi05_rlt_busybox_multitask_singlearm")
policy = policy_config.create_trained_policy(
    get_config("pi05_rlt_busybox_multitask_singlearm"),
    ckpt,
)
```

```bash
uv run scripts/serve_policy.py \
  --policy.config=pi05_rlt_busybox_multitask_singlearm \
  --policy.dir=/path/to/pravsels/pi05_rlt_busybox_multitask_singlearm
```
