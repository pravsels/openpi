---
license: apache-2.0
tags:
  - robotics
  - pi0
  - reward-recap
  - bin-packing
  - openpi
---

# pi0.5 Bin Pack — Reward Recap (Positive Only)

Fine-tuned [pi0.5](https://github.com/Physical-Intelligence/openpi) checkpoint for coffee capsule bin packing, trained with **positive-only advantage conditioning** (reward recap).

## Experiment

- **Objective:** Test whether positive-only advantage conditioning improves bin-pack policy when fine-tuning from a task-trained checkpoint.
- **Weight init:** Resumed from [pi05-bin-pack-single-dataset](https://huggingface.co/pravsels/pi05-bin-pack-single-dataset) checkpoint (step 29999).
- **Advantage mode:** `positive_only` — human demos are trained with prompt `"pack coffee capsules into the cardboard bin container. Advantage: positive"`, policy-collected frames are dropped.
- **Target steps:** 100,000

## Config

- **Config name:** `pi05_bin_pack_coffee_capsules_reward_recap_positive_only`
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
| 3,100 | 0.0148 |
| 25,000 | 0.0074 |
| 50,000 | 0.0058 |

## Checkpoint Hashes

Verify integrity with `tar cf - -C checkpoints/<step> params | sha256sum`.

| Step | Loss | SHA-256 |
|------|------|---------|
| 25,000 | 0.0074 | `23558b103ffeccb94dead23db3adf0c9119f38338d7a7ddc171db579a83bf6b1` |
| 50,000 | 0.0058 | `7b705f798619f1ecf4d5e8773896684ac735844265bcd0649cdd9d6dc18b5207` |
| 80,000 | 0.0050 | `f0cbcbf79a6072e33696e83d20540b9b9367d0c48bb1ace0d5d48b8d17981ccc` |

## Repo Structure

```
assets/                      # Norm stats for inference
checkpoints/<step>/params/   # Model weights (params only)
README.md                    # This file
TRAINING_LOG.md              # Training log
```

## W&B

- [Training dashboard](https://wandb.ai/pravsels/recap_plain/runs/9cfn5pz0)

## Usage

```python
from openpi.training.config import get_config
from openpi.serving.policy_server import PolicyServer

config = get_config("pi05_bin_pack_coffee_capsules_reward_recap_positive_only")
server = PolicyServer(config, checkpoint_path="checkpoints/<step>/params")
```
