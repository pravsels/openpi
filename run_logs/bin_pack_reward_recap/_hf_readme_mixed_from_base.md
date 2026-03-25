---
license: apache-2.0
tags:
  - robotics
  - pi0
  - reward-recap
  - bin-packing
  - openpi
---

# pi0.5 Bin Pack — Reward Recap (Mixed, from Base)

Fine-tuned [pi0.5](https://github.com/Physical-Intelligence/openpi) checkpoint for coffee capsule bin packing, trained with **mixed positive/negative advantage conditioning** (reward recap) starting from **pi0.5 base weights** (no task-specific pre-training).

## Experiment

- **Objective:** Test whether mixed positive/negative advantage conditioning works when training from pi0.5 base weights directly (no task-specific pre-training).
- **Weight init:** `weights/pi05_base/params` (pi0.5 base weights).
- **Advantage mode:** `mixed` — both successful and unsuccessful demonstrations receive advantage prompts (positive and negative).
- **Target steps:** 100,000

## Config

- **Config name:** `pi05_bin_pack_coffee_capsules_reward_recap_mixed_from_base`
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
| 0 | 0.3473 |
| 25,000 | 0.0125 |

Note: High initial loss (0.35) is expected — mixed mode introduces negative demonstrations the base model hasn't seen. Loss dropped rapidly in the first few thousand steps.

## Checkpoint Hashes

Verify integrity with `tar cf - -C checkpoints/<step> params | sha256sum`.

| Step | Loss | SHA-256 |
|------|------|---------|
| 25,000 | 0.0125 | `fda750265baa451291c6a0148ea405216e8319d86e1f6cbbf7d31103510e87e2` |

## Repo Structure

```
assets/                      # Norm stats for inference
checkpoints/<step>/params/   # Model weights (params only)
README.md                    # This file
TRAINING_LOG.md              # Training log
```

## W&B

- [Training dashboard](https://wandb.ai/pravsels/recap_plain/runs/kurb306v)

## Usage

```python
from openpi.training.config import get_config
from openpi.serving.policy_server import PolicyServer

config = get_config("pi05_bin_pack_coffee_capsules_reward_recap_mixed_from_base")
server = PolicyServer(config, checkpoint_path="checkpoints/<step>/params")
```
