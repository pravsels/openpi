---
license: apache-2.0
tags:
  - robotics
  - pi0
  - arx5
  - multitask
  - openpi
---

# pi0.5 ARX5 Multitask Micro Baseline

Fine-tuned [pi0.5](https://github.com/Physical-Intelligence/openpi) checkpoint for multi-task manipulation with ARX5 arms, trained on a 14-dataset micro mix with valid-index filtering (human-controlled + successful episodes only).

## Experiment

- **Objective:** Fine-tune PI0.5 on the micro training mix with baseline valid indices; compare later to advantaged variant.
- **Weight init:** `weights/pi05_base/params` (pi0.5 base weights).
- **Total steps:** 30,000 (completed)
- **Final loss:** 0.0107 (step 29,900)

## Config

- **Config name:** `pi05_arx5_multitask_micro_baseline`
- **Model:** pi0.5 (`pi05=True`, `action_horizon=50`)
- **Batch size:** 36
- **Learning rate:** 5e-5 cosine decay (1k warmup, decay over 100k steps)
- **Optimizer:** AdamW (gradient clip norm 1.0)
- **EMA decay:** 0.999
- **Delta actions:** enabled (delta joints, absolute grippers)
- **Per-timestep action normalization:** enabled (auto from delta actions)
- **Action space:** 14D bimanual (single-arm 7D padded to 14D with loss masking)

## Dataset

14 LeRobot datasets from `training_mix_micro.json` (all `villekuosmanen/*` repos). Filtered by `valid_indices.txt` to include only human-controlled, successful episodes.

## Checkpoint Hashes

Verify integrity with:

```bash
cd checkpoints/<step> && find params -type f | sort | xargs sha256sum | sha256sum
```

| Step | Loss | SHA-256 |
|------|------|---------|
| 25,000 | 0.0119 | `69ee51b80032d3a4424bd3834167fdd4d839701ab3b267c73ae6b7386922f1f8` |
| 29,999 | 0.0107 | `450e1c86c1d95ccb7215cc3662b90c6b56fb483006b640dfa2bc70bfa2593c01` |

## W&B

- [Training dashboard](https://wandb.ai/pravsels/arx5_multitask/runs/gtk5f6zw)

## Repo Structure

```
assets/                      # Norm stats, valid_indices.txt, training_mix_micro.json
checkpoints/<step>/params/   # Model weights (params only)
README.md                    # This file
TRAINING_LOG.md              # Training log
```

## Usage

```python
from openpi.training.config import get_config
from openpi.serving.policy_server import PolicyServer

config = get_config("pi05_arx5_multitask_micro_baseline")
server = PolicyServer(config, checkpoint_path="checkpoints/<step>/params")
```
