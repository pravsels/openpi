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

# pi0.5 Build Block Tower Subtask Reward Recap - Mixed

Fine-tuned pi0.5 checkpoint for block tower building using mixed reward recap semantics with advantage conditioning, dropout, and hierarchical subtask loss.

## Experiment

- **Config name:** `pi05_build_block_tower_subtask_recap_mixed`
- **Run type:** experiment
- **Objective:** train block tower recap with mixed advantage conditioning and hierarchical subtask prompting (subtask_loss_weight=1.0) to compare against flat-prompt and positive-only variants

## Dataset

- 6 HuggingFace datasets: `villekuosmanen/build_block_tower` plus 5 DAgger rounds (1.0.0 through 1.4.0)

## Uploaded Checkpoints

- `22000`: final checkpoint (training stopped at 22k/50k steps), SHA-256 `7addc02dc6542c61b08b8f608d5784d7aa1d5bbfc47d85a4c3ba9c8bf2af29f6`

Checkpoints are stored as params-only artifacts under `checkpoints/<step>/params/`.

To verify integrity after download:

```bash
cd checkpoints/22000 && find params -type f | sort | xargs sha256sum | sha256sum
```

## Assets

- `assets/` contains normalization stats and dataset metadata used by this run.

## W&B

- [Project dashboard](https://wandb.ai/pravsels/openpi_block_tower_recap)
- [Run jrtdxg3h](https://wandb.ai/pravsels/openpi_block_tower_recap/runs/jrtdxg3h)

## Repo Structure

```
checkpoints/22000/params/
checkpoints/22000/assets/
README.md
TRAINING_LOG.md
```
