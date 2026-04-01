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
- **Advantage mode:** `mixed` — human demos are trained with prompt `"pack coffee capsules into the cardboard bin container. Advantage: positive"`, policy-collected frames with `"... Advantage: negative"`.
- **Target steps:** 100,000 (stopped at 74,200)

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
| 50,000 | ~0.0075 |
| 74,000 | ~0.0070 |

Note: High initial loss (0.35) is expected — mixed mode introduces negative demonstrations the base model hasn't seen. Loss dropped rapidly in the first few thousand steps, then converged to ~0.007 by 74k.

## Checkpoint Hashes

Verify integrity with `find params -type f | sort | xargs cat | sha256sum`.

| Step | Loss | SHA-256 |
|------|------|---------|
| 25,000 | 0.0125 | `9e6d903b70a0159d6fb9979570556b031650f2e733e9b8a30c1d17b08f3307c2` |
| 50,000 | ~0.0075 | `f347d098e046f63ef65aa9c0c7a5614e0735667f1d03a5f8fb893e43698079c9` |
| 74,000 | ~0.0070 | `4fd1de8b341df95595b7691443e524638c80a0f1eb58c09d717a33741482d70e` |

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
