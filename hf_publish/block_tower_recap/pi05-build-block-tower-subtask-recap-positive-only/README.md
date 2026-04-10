---
license: apache-2.0
tags:
  - robotics
  - pi0
  - pi05
  - openpi
  - block-tower
  - reward-recap
  - subtask
---

# pi0.5 Build Block Tower Subtask Reward Recap - Positive Only

Fine-tuned pi0.5 checkpoint for block tower building using positive-only reward recap semantics with advantage conditioning, dropout, and hierarchical subtask loss.

## Experiment

- **Config name:** `pi05_build_block_tower_subtask_recap_positive_only`
- **Run type:** experiment
- **Objective:** train block tower recap with positive-only advantage conditioning and hierarchical subtask prompting (subtask_loss_weight=1.0) to compare against flat-prompt and mixed variants

## Dataset

- 6 HuggingFace datasets: `villekuosmanen/build_block_tower` plus 5 DAgger rounds (1.0.0 through 1.4.0)

## Uploaded Checkpoints

- `22000`: final checkpoint (training stopped at 22k/50k steps), SHA-256 `0680c2a5db6bac2771b4bf39cd2c9769aa5575905b45c775f88db77083120a10`

Checkpoints are stored as params-only artifacts under `checkpoints/<step>/params/`.

To verify integrity after download:

```bash
cd checkpoints/22000 && find params -type f | sort | xargs sha256sum | sha256sum
```

## Assets

- `assets/` contains normalization stats and dataset metadata used by this run.

## W&B

- [Project dashboard](https://wandb.ai/pravsels/openpi_block_tower_recap)
- [Run j20yc105](https://wandb.ai/pravsels/openpi_block_tower_recap/runs/j20yc105)

## Repo Structure

```
checkpoints/22000/params/
checkpoints/22000/assets/
README.md
TRAINING_LOG.md
```
