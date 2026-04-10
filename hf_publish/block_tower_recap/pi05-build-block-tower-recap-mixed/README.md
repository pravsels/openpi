---
license: apache-2.0
tags:
  - robotics
  - pi0
  - openpi
  - block-tower
  - reward-recap
---

# pi0.5 Build Block Tower Reward Recap - Mixed

Fine-tuned pi0.5 checkpoint for block tower building using mixed reward recap semantics with advantage conditioning and dropout.

## Experiment

- **Config name:** `pi05_build_block_tower_recap_mixed`
- **Run type:** experiment
- **Objective:** train block tower recap with mixed advantage conditioning (flat prompt, no subtask hierarchy) to compare against positive-only and subtask variants

## Dataset

- 6 HuggingFace datasets: `villekuosmanen/build_block_tower` plus 5 DAgger rounds (1.0.0 through 1.4.0)

## Uploaded Checkpoints

- `30000`: intermediate checkpoint, SHA-256 `341308cd5f8be0236d26cde04a1a7494c9428da7e5559996578b7ed55fb6dabf`
- `40000`: intermediate checkpoint, SHA-256 `ec86899d8566a7c0e4f90839616f113fb0401d4288180d020d9d0f15f5972ef2`
- `49999`: final checkpoint, SHA-256 `8fbeda24b6593304cb0a6b42a5ec9dd342cdb5f0e1b25a074687785761eccf9f`

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
