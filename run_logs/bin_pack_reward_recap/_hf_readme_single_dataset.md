---
license: apache-2.0
tags:
  - robotics
  - pi0
  - bin-packing
  - openpi
---

# pi0.5 Bin Pack — Single Dataset Baseline

Fine-tuned [pi0.5](https://github.com/Physical-Intelligence/openpi) checkpoint for coffee capsule bin packing, trained on a single dataset of ~200 teleoperated episodes. This serves as the base checkpoint for the reward recap experiments.

## Config

- **Config name:** `pi05_bin_pack_coffee_capsules_delta_single_dataset`
- **Model:** pi0.5 (`pi05=True`, `action_horizon=50`)
- **Batch size:** 36
- **Learning rate:** 5e-5 cosine decay (10k warmup)
- **Optimizer:** AdamW (gradient clip norm 1.0)
- **EMA decay:** 0.999
- **Delta actions:** enabled
- **Weight init:** `weights/pi05_base/params` (pi0.5 base weights)
- **Training steps:** 30,000

## Dataset

- `villekuosmanen/bin_pick_pack_coffee_capsules` (~200 teleoperated episodes)

## Checkpoint Hash

Verify integrity with `tar cf - -C checkpoints/29999 params | sha256sum`.

| Step | SHA-256 |
|------|---------|
| 29,999 | `bb051b5a3ee10adae7ee5313102fd7157e49d77a12a3b9a48e0688617108f9b0` |

## Downstream

This checkpoint is the weight init for the reward recap experiments:

- [pi05-bin-pack-reward-recap-positive-only](https://huggingface.co/pravsels/pi05-bin-pack-reward-recap-positive-only)
- [pi05-bin-pack-reward-recap-mixed](https://huggingface.co/pravsels/pi05-bin-pack-reward-recap-mixed)

## Repo Structure

```
assets/                      # Norm stats + valid indices for inference
checkpoints/29999/params/    # Model weights (params only)
README.md                    # This file
```

## Usage

```python
from openpi.training.config import get_config
from openpi.serving.policy_server import PolicyServer

config = get_config("pi05_bin_pack_coffee_capsules_delta_single_dataset")
server = PolicyServer(config, checkpoint_path="checkpoints/29999/params")
```
