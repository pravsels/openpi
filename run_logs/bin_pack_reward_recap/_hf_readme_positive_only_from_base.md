---
license: apache-2.0
tags:
  - robotics
  - pi0
  - reward-recap
  - bin-packing
  - openpi
---

# pi0.5 Bin Pack — Reward Recap (Positive Only, from Base)

Fine-tuned [pi0.5](https://github.com/Physical-Intelligence/openpi) checkpoint for coffee capsule bin packing, trained with **positive-only advantage conditioning** (reward recap) starting from **pi0.5 base weights** (no task-specific pre-training).

## Experiment

- **Objective:** Test whether positive-only advantage conditioning works when training from pi0.5 base weights directly (no task-specific pre-training).
- **Weight init:** `weights/pi05_base/params` (pi0.5 base weights).
- **Advantage mode:** `positive_only` — human demos are trained with prompt `"pack coffee capsules into the cardboard bin container. Advantage: positive"`, policy-collected frames are dropped.
- **Target steps:** 100,000

## Config

- **Config name:** `pi05_bin_pack_coffee_capsules_reward_recap_positive_only_from_base`
- **Model:** pi0.5 (`pi05=True`, `action_horizon=50`)
- **Batch size:** 36
- **Learning rate:** 5e-5 cosine decay (10k warmup)
- **Optimizer:** AdamW (gradient clip norm 1.0)
- **EMA decay:** 0.999
- **Delta actions:** enabled

## Dataset

9 LeRobot datasets (1 base + 8 dAgger rounds):

- `villekuosmanen/bin_pick_pack_coffee_capsules`
- `villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.0.0`
- `villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.1.0`
- `villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.2.0`
- `villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.3.1`
- `villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.4.0`
- `villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.5.0`
- `villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.5.1`
- `villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.7.0`

## Loss Progression

| Step | Loss |
|------|------|
| 1,100 | 0.0425 |
| 25,000 | 0.0097 |
| 50,000 | 0.0066 |

## Checkpoint Hashes

Verify integrity with `tar cf - -C checkpoints/<step> params | sha256sum`.

| Step | Loss | SHA-256 |
|------|------|---------|
| 25,000 | 0.0097 | `fe1e6b97b7dafa1ea6e74a35f698df798ba2b739ecef54c38a810beffb404e75` |
| 50,000 | 0.0066 | `27d2ceb579a0cdd7b34a06c2c6c52b3b8bcdbbdcd81b0ed2826ee62daaaa6603` |
| 73,000 | 0.0053 | `a43e1edbb84b6050588f7d1bec5158ba2c5b4649613586a6105cf9a8318ceb00` |

## Repo Structure

```
assets/                      # Norm stats for inference
checkpoints/<step>/params/   # Model weights (params only)
README.md                    # This file
TRAINING_LOG.md              # Training log
```

## W&B

- [Training dashboard](https://wandb.ai/pravsels/recap_plain/runs/l4kfxxqe)

## Usage

```python
from openpi.training.config import get_config
from openpi.serving.policy_server import PolicyServer

config = get_config("pi05_bin_pack_coffee_capsules_reward_recap_positive_only_from_base")
server = PolicyServer(config, checkpoint_path="checkpoints/<step>/params")
```
