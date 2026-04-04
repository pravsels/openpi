---
license: apache-2.0
tags:
  - robotics
  - pi0
  - openpi
  - bin-packing
  - reward-recap
---

# pi0.5 Bin Pack Reward Recap - Mixed Fix

Fine-tuned pi0.5 checkpoint for coffee capsule bin packing, rerun after the advantage-token placement and valid-index persistence fixes using mixed reward recap semantics.

## Experiment

- **Config name:** `pi05_bin_pack_coffee_capsules_recap_mixed`
- **Run type:** experiment
- **Objective:** rerun bin-pack mixed recap after fixing advantage-token placement, config-aware valid-index persistence, and shortening the run to 50k steps

## Dataset

- 9 HuggingFace datasets: `bin_pick_pack_coffee_capsules` plus 8 DAgger rounds

## Uploaded Checkpoints

- `30000`: intermediate checkpoint, SHA-256 `29fb62934e85c7fd18a3c10cd6b80b816eb323b8b4a7bdf0dab3a1847cdb95ed`
- `40000`: intermediate checkpoint, SHA-256 `d1b7d7c4b465662ad32dd091d56163be49c79b7259261457c136cfe414fa46ab`
- `49999`: final checkpoint, SHA-256 `1dbc8980ce3f860b05a34b28afd795596b82016ad458d1348b3ad476516b66c8`

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
