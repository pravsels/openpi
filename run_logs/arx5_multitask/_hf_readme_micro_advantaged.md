---
license: apache-2.0
tags:
  - robotics
  - pi0
  - arx5
  - multitask
  - openpi
---

# pi0.5 ARX5 Multitask Micro Advantaged

Fine-tuned [pi0.5](https://github.com/Physical-Intelligence/openpi) checkpoint for multi-task manipulation with ARX5 arms, trained on a 14-dataset micro mix with advantaged valid-index filtering.

## Experiment

- **Objective:** Fine-tune PI0.5 on the micro training mix with advantaged valid indices; compare to baseline variant.
- **Weight init:** `weights/pi05_base/params` (pi0.5 base weights).
- **Total steps:** 30,000 (completed)
- **Final loss:** 0.0080 (step 29,900)

## Config

- **Config name:** `pi05_arx5_multitask_micro_advantaged`
- **Model:** pi0.5 (`pi05=True`, `action_horizon=50`)
- **Batch size:** 36
- **Learning rate:** 5e-5 cosine decay (1k warmup, decay over 100k steps)
- **Optimizer:** AdamW (gradient clip norm 1.0)
- **EMA decay:** 0.999
- **Delta actions:** enabled (delta joints, absolute grippers)
- **Per-timestep action normalization:** enabled (auto from delta actions)
- **Action space:** 14D bimanual (single-arm 7D padded to 14D with loss masking)

## Dataset

14 LeRobot datasets from `training_mix_micro.json` (all `villekuosmanen/*` repos). Filtered by `valid_indices.txt` (advantaged indices).

## Checkpoint Hashes

Verify integrity with:

```bash
cd checkpoints/<step> && find params -type f | sort | xargs sha256sum | sha256sum
```

| Step | Loss | SHA-256 |
|------|------|---------|
| 25,000 | 0.0089 | `1648c67a7ac44d377f28f316384bdcab72af4422237f9f9485e1e77a02c6a65c` |
| 29,999 | 0.0080 | `aff337d89dd426388303855ed8fca784f5b5615b33cbad14f26dfbe8688caa88` |

## W&B

- [Training dashboard](https://wandb.ai/pravsels/arx5_multitask/runs/jik4rmpl)

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

config = get_config("pi05_arx5_multitask_micro_advantaged")
server = PolicyServer(config, checkpoint_path="checkpoints/<step>/params")
```
