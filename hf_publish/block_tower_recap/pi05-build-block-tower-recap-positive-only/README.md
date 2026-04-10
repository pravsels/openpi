---
license: apache-2.0
tags:
  - robotics
  - pi0
  - openpi
  - block-tower
  - reward-recap
---

# pi0.5 Build Block Tower Reward Recap - Positive Only

Fine-tuned pi0.5 checkpoint for block tower building using positive-only reward recap semantics with advantage conditioning and dropout.

## Experiment

- **Config name:** `pi05_build_block_tower_recap_positive_only`
- **Run type:** experiment
- **Objective:** train block tower recap with positive-only advantage conditioning (flat prompt, no subtask hierarchy) to compare against mixed and subtask variants

## Dataset

- 6 HuggingFace datasets: `villekuosmanen/build_block_tower` plus 5 DAgger rounds (1.0.0 through 1.4.0)

## Uploaded Checkpoints

- `30000`: intermediate checkpoint, SHA-256 `cb05d0f00920563db4bb556001b28c739cdf19c4b327ffb7fb7e4f558e58c076`
- `40000`: intermediate checkpoint, SHA-256 `cb1cd2e9198c2090d01c7c0c671001a3e553db1d4c43f297bf37f10764285912`
- `49999`: final checkpoint, SHA-256 `92a5dcfcac63b986815f06b3f4c1d34e048b0f30771fb67e4a5906e1c7bf3130`

Checkpoints are stored as params-only artifacts under `checkpoints/<step>/params/`.

## Assets

- `assets/` contains normalization stats and dataset metadata used by this run.

## W&B

- [Project dashboard](https://wandb.ai/pravsels/openpi_block_tower_recap)

## Repo Structure

```
checkpoints/30000/params/
checkpoints/30000/assets/
checkpoints/40000/params/
checkpoints/40000/assets/
checkpoints/49999/params/
checkpoints/49999/assets/
README.md
TRAINING_LOG.md
```
