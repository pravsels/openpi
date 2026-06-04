---
license: apache-2.0
tags:
  - robotics
  - pi0
  - so101
  - openpi
  - lerobot
---

# pi0.5 SO101 Stacking Magnetic Cubes (Isambard)

Fine-tuned [pi0.5](https://github.com/Physical-Intelligence/openpi) checkpoint for SO101 magnetic-cube stacking. Trained on Isambard GH200 (4× GPU node). **Training is in progress** — this repo publishes an interim **10,000-step** checkpoint; later steps (e.g. 50k) may be added under separate `step_<N>/` directories.

## Published checkpoints

| Directory | Step | Loss (approx) | Notes |
|-----------|------|---------------|--------|
| `step_10000/params/` | 10,000 | 0.0179 | Interim publish while full 50k run continues |

Future checkpoints will use the same layout, e.g. `step_49999/params/`.

## Experiment

- **Objective:** Fine-tune pi0.5 on `lorenzouttini/so101_stacking_magnetic_cubes`.
- **Weight init:** pi0.5 base (`pi05_base`).
- **Training target:** 50,000 steps (`save_interval=5000`).

## Config

- **Config name:** `pi05_so101_stacking_magnetic_cubes`
- **Experiment name:** `so101_stacking_magnetic_cubes`
- **Model:** pi0.5 (`pi05=True`, `action_horizon=30`)
- **Batch size:** 32 · **LR:** 2.5e-5 cosine (1k warmup) · **EMA:** 0.999
- **Delta actions:** yes · **Default prompt:** `stack the magnetic cubes`

## Dataset

- [lorenzouttini/so101_stacking_magnetic_cubes](https://huggingface.co/datasets/lorenzouttini/so101_stacking_magnetic_cubes)

## W&B

- [Training run](https://wandb.ai/pravsels/so101_stacking_magnetic_cubes/runs/7h1jiwva) (synced partial; full 50k curves after final sync)

## Repo structure

```
README.md
TRAINING_LOG.md
MODEL_PASSPORT.json          # signed for step_10000 package layout
SIGNOFF.json
assets/                      # norm stats, valid_indices, reference_test_vector
step_10000/params/           # interim checkpoint (params only)
```

## Usage

Point `checkpoint_path` at the published step directory:

```python
from openpi.training.config import get_config
from openpi.serving.policy_server import PolicyServer

config = get_config("pi05_so101_stacking_magnetic_cubes")
server = PolicyServer(config, checkpoint_path="step_10000/params")
```

Verify integrity before load:

```bash
validate-checkpoint . --require-signoff
```
