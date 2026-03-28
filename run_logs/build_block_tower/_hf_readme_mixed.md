---
license: apache-2.0
tags:
  - robotics
  - pi0
  - reward-recap
  - block-tower
  - openpi
---

# pi0.5 Build Block Tower — Mixed (Advantage-Conditioned with Negative Labels)

Fine-tuned [pi0.5](https://github.com/Physical-Intelligence/openpi) checkpoint for building a block tower, trained with **mixed advantage conditioning** (human frames → "Advantage: positive", policy frames → "Advantage: negative") using human + dAgger demonstrations.

## Experiment

- **Objective:** Train on all 6 block tower datasets with mixed advantage prompts. Compare against dyna (positive-only, drops policy frames) and baseline (no advantage prompts).
- **Weight init:** `weights/pi05_base/params` (pi0.5 base weights).
- **Advantage mode:** `mixed` (human → positive, policy → negative)
- **Total steps:** 100,000 (completed)
- **Final loss:** 0.0097 (step 99,900)

## Config

- **Config name:** `pi05_build_block_tower_mixed`
- **Model:** pi0.5 (`pi05=True`, `action_horizon=50`)
- **Batch size:** 36
- **Learning rate:** 5e-5 cosine decay (10k warmup, decay to 5e-5 over 1M steps)
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

Verify integrity with:

```bash
cd checkpoints/<step> && find params -type f | sort | xargs sha256sum | sha256sum
```

| Step | Loss | SHA-256 |
|------|------|---------|
| 25,000 | 0.0207 | `6d44e6b2aec69b964e974ffcf551834ac443196a36d85e891d9f246859a1afb1` |
| 30,000 | 0.0179 | `d9544183f5a70f044f044c103c551c57588aaeaf3ff1b46b4159cd10fef6f528` |
| 50,000 | 0.0144 | `3f289b60f8f4d9676ff3250aae41919355045d72b92cd7d4e0a21ea9071dea91` |
| 75,000 | 0.0112 | `4a2b9d394f89bdb4ba2e3620eb774bba5ba9a8c829e85e2319d1dd1adb2bd03d` |

## W&B

- [Training dashboard](https://wandb.ai/pravsels/block_tower/runs/jhyepslj)

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

config = get_config("pi05_build_block_tower_mixed")
server = PolicyServer(config, checkpoint_path="checkpoints/<step>/params")
```
