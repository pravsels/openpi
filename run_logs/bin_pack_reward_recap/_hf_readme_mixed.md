---
license: apache-2.0
tags:
  - robotics
  - pi0
  - reward-recap
  - bin-packing
  - openpi
---

# pi0.5 Bin Pack — Reward Recap (Mixed)

Fine-tuned [pi0.5](https://github.com/Physical-Intelligence/openpi) checkpoint for coffee capsule bin packing, trained with **mixed positive/negative advantage conditioning** (reward recap).

## Experiment

- **Objective:** Test whether mixed positive/negative advantage conditioning improves bin-pack policy when fine-tuning from a task-trained checkpoint.
- **Weight init:** Resumed from [pi05-bin-pack-single-dataset](https://huggingface.co/pravsels/pi05-bin-pack-single-dataset) checkpoint (step 29999).
- **Advantage mode:** `mixed` — human demos are trained with prompt `"pack coffee capsules into the cardboard bin container. Advantage: positive"`, policy-collected frames with `"... Advantage: negative"`.
- **Target steps:** 100,000

## Config

- **Config name:** `pi05_bin_pack_coffee_capsules_reward_recap_mixed`
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
| 0 | 0.5005 |
| 25,000 | 0.0098 |
| 50,000 | 0.0075 |

Note: High initial loss (0.50) is expected — mixed mode introduces negative demonstrations the model hasn't seen. Loss dropped rapidly in the first few thousand steps.

## Checkpoint Hashes

Verify integrity with `tar cf - -C checkpoints/<step> params | sha256sum`.

| Step | Loss | SHA-256 |
|------|------|---------|
| 25,000 | 0.0098 | `626c9cbce476d5e90abfefa57dda4322777240314630eb79abc5f37ff8f75ffb` |
| 50,000 | 0.0075 | `bec9d174f325623bc1c677e139001212c5a2c915810113be3872b45956fe609a` |
| 72,000 | 0.0061 | `2cdb1acbf6bdad15a681b219af63468d7c0af0f707764c433209f1d3b435a69c` |

## Repo Structure

```
assets/                      # Norm stats for inference
checkpoints/<step>/params/   # Model weights (params only)
README.md                    # This file
TRAINING_LOG.md              # Training log
```

## W&B

- [Training dashboard](https://wandb.ai/pravsels/recap_plain/runs/vvd1y2sk)

## Usage

```python
from openpi.training.config import get_config
from openpi.serving.policy_server import PolicyServer

config = get_config("pi05_bin_pack_coffee_capsules_reward_recap_mixed")
server = PolicyServer(config, checkpoint_path="checkpoints/<step>/params")
```
