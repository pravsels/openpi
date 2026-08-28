---
tags:
- pi05
- robotics
- openpi
- rlt
- rl-token
datasets:
- villekuosmanen/busybox_push_green_button
---

# pi05_rlt_busybox_push_green_button

RL-token encoder/decoder (RLT Stage 1) trained on the frozen [π0.5](https://github.com/Physical-Intelligence/openpi) green-button checkpoint. This is the checkpoint `hw_control.pi0_rlt` loads to extract tokens for the demo cache and online RL.

| | |
|---|---|
| **Policy** | π0.5 + RLT (`Pi0RLConfig`, `pi05=true`) |
| **Frozen VLA** | [`pravsels/pi05_busybox_push_green_button`](https://huggingface.co/pravsels/pi05_busybox_push_green_button) |
| **Dataset** | [villekuosmanen/busybox_push_green_button](https://huggingface.co/datasets/villekuosmanen/busybox_push_green_button) |
| **Task** | `busybox_push_green_button` |
| **Prompt** | `push the green button` |
| **Action dim** | 6 (single-arm; 5 joints delta, gripper absolute) |
| **Cameras** | `top` → `base_0_rgb`, `wrist` → `left_wrist_0_rgb`, `front` → `base_1_rgb` |
| **Training** | GCloud 1× A100 80GB, VLA frozen (`rl_vla_loss_weight=0.0`), 20k steps, global batch 16, action horizon 30 |
| **W&B project** | [busybox_push_green_button_rlt](https://wandb.ai/pravsels/busybox_push_green_button_rlt) |
| **W&B run** | [wnk0bxds](https://wandb.ai/pravsels/busybox_push_green_button_rlt/runs/wnk0bxds) |

## Checkpoints

`main` is the finalized 20k checkpoint (`params/` + `assets/` at the repo root). `train_state/` is not published.

| Field | Value |
|------|--------|
| Latest published | `main` (`params/`, step 19999) |
| Config | `pi05_rlt_busybox_push_green_button` |
| Code | `4354d7d` on `task/rlt_busybox_green_button` |
| Runtime | 4h 15m (2026-08-27 20:04 UTC → 2026-08-28 00:19 UTC) |
| Train loss | 9935.29 (step 0) → 189.18 (step 19900) |

## Usage

```python
from huggingface_hub import snapshot_download
from openpi.policies import policy_config
from openpi.training.config import get_config

ckpt = snapshot_download("pravsels/pi05_rlt_busybox_push_green_button")
policy = policy_config.create_trained_policy(
    get_config("pi05_rlt_busybox_push_green_button"),
    ckpt,
)
```

```bash
uv run scripts/serve_policy.py \
  --policy.config=pi05_rlt_busybox_push_green_button \
  --policy.dir=/path/to/pravsels/pi05_rlt_busybox_push_green_button
```
