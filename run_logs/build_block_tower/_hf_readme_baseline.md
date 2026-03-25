---
license: apache-2.0
tags:
  - robotics
  - pi0
  - block-tower
  - openpi
---

# pi0.5 Build Block Tower — Baseline

Fine-tuned [pi0.5](https://github.com/Physical-Intelligence/openpi) checkpoint for building a block tower, trained with **imitation learning only** (no advantage conditioning) on 200 human demonstrations.

## Experiment

- **Objective:** Train a block tower policy from pi0.5 base weights using human demos only (imitation learning baseline).
- **Weight init:** `weights/pi05_base/params` (pi0.5 base weights).
- **Advantage conditioning:** None (standard imitation learning).
- **Target steps:** 100,000

## Config

- **Config name:** `pi05_build_block_tower_baseline`
- **Model:** pi0.5 (`pi05=True`, `action_horizon=50`)
- **Batch size:** 36
- **Learning rate:** 5e-5 cosine decay (10k warmup)
- **Optimizer:** AdamW (gradient clip norm 1.0)
- **EMA decay:** 0.999
- **Delta actions:** enabled
- **State/action space:** 7D joint-space

## Dataset

- `villekuosmanen/build_block_tower` (200 episodes, LeRobot v2.1)

## Checkpoint Hashes

Verify integrity with `tar cf - -C checkpoints/<step> params | sha256sum`.

| Step | Loss | SHA-256 |
|------|------|---------|
| 25,000 | 0.0111 | `e137fa5a690e21f275e9dcbbffda6428c0eabd0100270c407ac366c225674ea2` |
| 50,000 | 0.0077 | `51e01160ab7119dbbf6b7c4515900cc794eb11773c47ced6f853e0c4f976c652` |
| 55,000 | 0.0068 | `158ff29591982484b6594afaabec078bdf8b59e7b72a4c886d4a542233581d11` |

## W&B

- [Training dashboard](https://wandb.ai/pravsels/block_tower/runs/6hoa4kt5)

## Repo Structure

```
assets/                      # Norm stats for inference
checkpoints/<step>/params/   # Model weights (params only)
README.md                    # This file
TRAINING_LOG.md              # Training log
```

## Usage

```python
from openpi.training.config import get_config
from openpi.serving.policy_server import PolicyServer

config = get_config("pi05_build_block_tower_baseline")
server = PolicyServer(config, checkpoint_path="checkpoints/<step>/params")
```
