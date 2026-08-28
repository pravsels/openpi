---
tags:
- pi05
- robotics
- openpi
- vla
datasets:
- villekuosmanen/busybox_push_green_button
---

# pi05_busybox_push_green_button

Full-component fine-tune of [π0.5](https://github.com/Physical-Intelligence/openpi) for `busybox_push_green_button` on SO101 data.

| | |
|---|---|
| **Policy** | π0.5 (`pi05=true`, full-component) |
| **Init checkpoint** | `pi05_base` (`gs://openpi-assets/checkpoints/pi05_base`) |
| **Dataset** | [villekuosmanen/busybox_push_green_button](https://huggingface.co/datasets/villekuosmanen/busybox_push_green_button) |
| **Task** | `busybox_push_green_button` |
| **Prompt** | `push the green button` |
| **Action dim** | 6 (single-arm) |
| **Cameras** | `top` → `base_0_rgb`, `wrist` → `left_wrist_0_rgb`, `front` → `base_1_rgb` |
| **Training** | GMAN 4× H100 80GB data-parallel, full-component, 30k steps, global batch 32, action horizon 30, 720x1280 cameras |
| **W&B project** | [busybox_push_green_button_pi05](https://wandb.ai/pravsels/busybox_push_green_button_pi05) |
| **W&B run** | [h00ekmci](https://wandb.ai/pravsels/busybox_push_green_button_pi05/runs/h00ekmci) |

## Checkpoints

`main` is the latest finalized checkpoint (`params/` + `assets/` at the repo root). The publisher overwrites the root at each 5k save. Training target is step 30,000.

| Field | Value |
|------|--------|
| Latest published | `main` (`params/`) |
| Config | `pi05_busybox_push_green_button` |

## Usage

```python
from huggingface_hub import snapshot_download
from openpi.policies import policy_config
from openpi.training.config import get_config

ckpt = snapshot_download("pravsels/pi05_busybox_push_green_button")
policy = policy_config.create_trained_policy(
    get_config("pi05_busybox_push_green_button"),
    ckpt,
)
```

```bash
uv run scripts/serve_policy.py \
  --policy.config=pi05_busybox_push_green_button \
  --policy.dir=/path/to/pravsels/pi05_busybox_push_green_button
```
