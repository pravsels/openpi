---
license: apache-2.0
tags:
  - robotics
  - pi0
  - block-tower
  - openpi
  - joints-only
---

# pi0.5 Build Block Tower 6-Mix — Joints-Only Ablation

Fine-tuned [pi0.5](https://github.com/Physical-Intelligence/openpi) checkpoint for the block-tower task, trained with loss restricted to the first 7 joint dimensions only. EEF channels of state/action are zeroed and excluded from the flow-matching loss via `action_dim_mask`.

## Experiment

- **Objective:** Test whether removing EEF supervision improves joint-space policy quality on the block-tower task.
- **Weight init:** `weights/pi05_base/params` (pi0.5 base weights).
- **Total steps:** 50,000 (completed)
- **Final loss:** 0.0112 (step 49,900)

## Config

- **Config name:** `pi05_build_block_tower_baseline_6mix_joints_only`
- **Model:** pi0.5 (`pi05=True`, `action_horizon=50`)
- **Batch size:** 36
- **Learning rate:** 5e-5 cosine decay (1k warmup)
- **Optimizer:** AdamW (gradient clip norm 1.0)
- **EMA decay:** 0.999
- **Delta actions:** enabled
- **Action space:** 17D canonical (first 7 joint dims active, remaining 10 EEF dims masked)
- **`joints_only`:** `True` — `action_dim_mask=[True]*7+[False]*10`
- **Norm stats:** reused from `pi05_build_block_tower_baseline_6mix/retain/step_49999/alpha_0.5/assets`

## Dataset

6 HuggingFace datasets: `villekuosmanen/build_block_tower` plus `dAgger_build_block_tower_1.0.0` through `1.4.0` (340 episodes total).

## Checkpoint Hashes

Verify integrity with:

```bash
cd checkpoints/<step> && find params -type f | sort | xargs sha256sum | sha256sum
```

| Step | Loss | SHA-256 |
|------|------|---------|
| 49,999 | 0.0112 | `7cd730580e57c6ae390ff2e0fc41491941b6b4d10e0e5cf959e6243034eabb45` |

## W&B

- [Training dashboard](https://wandb.ai/pravsels/build_block_tower_baseline_6mix_joints_only/runs/u6z1ph6k)

## Repo Structure

```
assets/                      # Norm stats, valid_indices.txt
checkpoints/49999/params/    # Model weights (params only)
README.md                    # This file
TRAINING_LOG.md              # Training log
```

## Usage

```python
from openpi.training.config import get_config
from openpi.serving.policy_server import PolicyServer

config = get_config("pi05_build_block_tower_baseline_6mix_joints_only")
server = PolicyServer(config, checkpoint_path="checkpoints/49999/params")
```
