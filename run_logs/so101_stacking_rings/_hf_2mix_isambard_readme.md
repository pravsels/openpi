---
license: apache-2.0
tags:
  - robotics
  - pi0
  - so101
  - openpi
  - lerobot
---

# pi0.5 SO101 Stacking Rings — 2-mix (Isambard)

Fine-tuned [pi0.5](https://github.com/Physical-Intelligence/openpi) checkpoint for SO101 ring stacking, trained on a **two-dataset mix** (original teleop + rollout corrections). Trained on Isambard GH200 (4× GPU node).

## Experiment

- **Objective:** Improve stacking success with rollout-correction data mixed into teleop.
- **Weight init:** pi0.5 base weights (`pi05_base`).
- **Published step:** **49,999** (50k training run, `save_interval=5000`).
- **Loss at step ~49,900:** ~0.0074

## Config

- **Config name:** `pi05_so101_stacking_rings`
- **Experiment name:** `so101_stacking_rings_2mix`
- **Model:** pi0.5 (`pi05=True`, `action_horizon=30`)
- **Batch size:** 32 · **Steps:** 50,000 · **Save interval:** 5,000
- **LR:** 2.5e-5 cosine (1k warmup) · **EMA:** 0.999
- **Delta actions:** yes · **Default prompt:** `stack the rings`

## Datasets (2-mix)

- [lorenzouttini/so101_stacking_rings](https://huggingface.co/datasets/lorenzouttini/so101_stacking_rings) — 101 episodes, ~34k frames (teleop)
- [lorenzouttini/rollout_so101_stacking_rings_20260603_154953](https://huggingface.co/datasets/lorenzouttini/rollout_so101_stacking_rings_20260603_154953) — 100 episodes, ~28k frames (rollout corrections)

## Checkpoint Hashes

Verify integrity (params only at published step):

```bash
cd checkpoints/49999/params && find . -type f | sort | xargs sha256sum | sha256sum
```

| Step | Loss (approx) | SHA-256 (params tree) |
|------|---------------|------------------------|
| **49,999** | 0.0074 | `e93a3d4f2a372c72d33f600c36326d5cd148ea9682f4c7bfb8153fe2796d7fcb` |

## W&B

- [2-mix Isambard training run](https://wandb.ai/pravsels/so101_stacking_rings_2mix_isambard/runs/y4z7w346)
- [Single-dataset Isambard baseline](https://wandb.ai/pravsels/so101_stacking_rings_isambard/runs/ox7qwnsz) — for comparison

## Artifacts

Signed `MODEL_PASSPORT.json` + `SIGNOFF.json` at repo root (checkpoint-passport v0.2, verdict `soft_signal`).

## Repo structure

```
README.md
TRAINING_LOG.md
MODEL_PASSPORT.json
SIGNOFF.json
assets/                              # norm stats, valid_indices, reference_test_vector
checkpoints/49999/params/            # <-- published training step (params only, no train_state)
```

## Usage

```python
from openpi.training.config import get_config
from openpi.serving.policy_server import PolicyServer

config = get_config("pi05_so101_stacking_rings")
server = PolicyServer(config, checkpoint_path="checkpoints/49999/params")
```
