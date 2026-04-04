---
license: apache-2.0
tags:
  - robotics
  - pi0
  - openpi
  - bin-packing
  - reward-recap
---

# pi0.5 Bin Pack Reward Recap - Positive Only Fix

Fine-tuned pi0.5 checkpoint for coffee capsule bin packing, rerun after the advantage-token placement and valid-index persistence fixes using positive-only reward recap semantics.

## Experiment

- **Config name:** `pi05_bin_pack_coffee_capsules_recap_positive_only`
- **Run type:** experiment
- **Objective:** rerun bin-pack positive-only recap after fixing advantage-token placement, config-aware valid-index persistence, and shortening the run to 50k steps

## Dataset

- 9 HuggingFace datasets: `bin_pick_pack_coffee_capsules` plus 8 DAgger rounds

## Uploaded Checkpoints

- `30000`: intermediate checkpoint, SHA-256 `489b7aead63b6bd715aaa55305eeeabef4dc70634d79eeca9d2d384e1d416052`
- `40000`: intermediate checkpoint, SHA-256 `d53517fa406be906f64e26a9e7eb3c15f84e1809f10f46074c7dce6b43b9ab3a`
- `49999`: final checkpoint, SHA-256 `a880a381e90a16706a24699f5a95c6554e7bb10f6445c5fdbccc169846205f52`

Checkpoints are stored as params-only artifacts under `checkpoints/<step>/params/`.

## Assets

- `assets/` contains normalization stats and dataset metadata used by this run.

## W&B

- [Project dashboard](https://wandb.ai/pravsels/openpi_recap_fix)

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
