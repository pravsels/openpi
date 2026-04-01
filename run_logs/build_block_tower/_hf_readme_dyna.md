---
license: apache-2.0
tags:
  - robotics
  - pi0
  - reward-recap
  - block-tower
  - openpi
---

# pi0.5 Build Block Tower — Dyna (Positive-Only Conditioning)

Fine-tuned [pi0.5](https://github.com/Physical-Intelligence/openpi) checkpoint for building a block tower, trained with **positive-only advantage conditioning** (Dyna-style) using human + dAgger demonstrations.

## Experiment

- **Objective:** Train a block tower policy using all human + dAgger data with Dyna-style conditioning (positive-only: human frames get "Advantage: positive", policy frames dropped).
- **Weight init:** `weights/pi05_base/params` (pi0.5 base weights).
- **Advantage mode:** `positive_only` — human demos are trained with prompt `"build a block tower. Advantage: positive"`, policy-collected frames are dropped.
- **Target steps:** 100,000
- **Latest observed step before stop:** 81,900 (loss 0.0111)

## Config

- **Config name:** `pi05_build_block_tower_dyna`
- **Model:** pi0.5 (`pi05=True`, `action_horizon=50`)
- **Batch size:** 36
- **Learning rate:** 5e-5 cosine decay (10k warmup)
- **Optimizer:** AdamW (gradient clip norm 1.0)
- **EMA decay:** 0.999
- **Delta actions:** enabled
- **State/action space:** 7D joint-space

## Dataset

6 LeRobot datasets (1 base + 5 dAgger rounds, v2.1):

- `villekuosmanen/build_block_tower`
- `villekuosmanen/dAgger_build_block_tower_1.0.0`
- `villekuosmanen/dAgger_build_block_tower_1.1.0`
- `villekuosmanen/dAgger_build_block_tower_1.2.0`
- `villekuosmanen/dAgger_build_block_tower_1.3.0`
- `villekuosmanen/dAgger_build_block_tower_1.4.0`

## Checkpoint Hashes

Verify integrity with `tar cf - -C checkpoints/<step> params | sha256sum`.

| Step | Loss | SHA-256 |
|------|------|---------|
| 25,000 | 0.0212 | `9bb8e728529b7d4461467b45ec1c7b9fed972ab03bf74feec85e1021047f0545` |
| 50,000 | 0.0145 | `a6673bee89023ba6726fa5c2c7d9957c289b8cdd5427be0815720a31eb3f0164` |
| 80,000 | ~0.0115 | `564caf6cfdfe57d80733e8806f28005d6077189d7dfdcb8f2b4f5e6857dab827` |

Note: checkpoint `48,000` was removed from the HuggingFace repo.

## W&B

- [Training dashboard](https://wandb.ai/pravsels/block_tower/runs/bcqxnzhs)
- Synced coverage currently reaches `_step=81600` (final local logs reached step 81,900 before manual stop).

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

config = get_config("pi05_build_block_tower_dyna")
server = PolicyServer(config, checkpoint_path="checkpoints/<step>/params")
```
