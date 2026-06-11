---
license: apache-2.0
tags:
  - robotics
  - LeRobot
  - pi0
  - so101
  - openpi
---

# pi0.5 SO101 Eye Drops Top Shelf (Isambard)

Fine-tuned pi0.5 checkpoint for SO101 eye-drops placing task. Trained on Isambard GH200 (4× GPU node). Final checkpoint at step 49,999.

## Published checkpoints

| Directory | Step   | Notes |
| --------- | ------ | ----- |
| params/   | 49,999 | Final |

## Experiment

- **Objective:** Fine-tune pi0.5 on `lorenzouttini/so101_eye_drops_top_shelf2_20260609_160053`.
- **Weight init:** pi0.5 base (`pi05_base`).
- **Training target:** 50,000 steps (`save_interval=5000`).

## Config

- **Config name:** `pi05_so101_eye_drops_top_shelf`
- **Experiment name:** `so101_eye_drops_top_shelf`
- **Model:** pi0.5 (`pi05=True`, `action_horizon=30`)
- **Batch size:** 32 · **LR:** 2.5e-5 cosine (1k warmup) · **EMA:** 0.999
- **Delta actions:** yes · **Default prompt:** `place the eye drops on the top shelf`

## Dataset

- [lorenzouttini/so101_eye_drops_top_shelf2_20260609_160053](https://huggingface.co/datasets/lorenzouttini/so101_eye_drops_top_shelf2_20260609_160053)

## W&B

- Training run (to be synced)

## Usage

```python
from openpi.training.config import get_config
from openpi.serving.policy_server import PolicyServer

config = get_config("pi05_so101_eye_drops_top_shelf")
server = PolicyServer(config, checkpoint_path="params")
```

Verify integrity before load:

```bash
validate-checkpoint . --require-signoff
```
