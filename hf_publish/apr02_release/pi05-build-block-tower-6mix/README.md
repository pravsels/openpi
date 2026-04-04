---
license: apache-2.0
tags:
  - robotics
  - pi0
  - openpi
  - block-tower
---

# pi0.5 Build Block Tower - 6mix

Fine-tuned pi0.5 checkpoint for build-block-tower, trained on the base dataset plus five DAgger rounds (6 datasets total) with imitation learning only.

## Experiment

- **Config name:** `pi05_build_block_tower_baseline`
- **Run type:** replication
- **Objective:** train the build-block-tower baseline on the base dataset plus five DAgger rounds using the synced `baseline` checkpoint/output path
- **Weight init:** `weights/pi05_base/params`

## Dataset

- 6 HuggingFace datasets: `villekuosmanen/build_block_tower` plus `dAgger_build_block_tower_1.0.0` through `1.4.0`

## Uploaded Checkpoints

- `30000`: intermediate checkpoint, SHA-256 `3f0e7a56c29623df26809b19f698e7ee60232a5f203c820bc9d8b248413e2119`
- `40000`: intermediate checkpoint, SHA-256 `1022fe15232cc345a4034172e626cfd22f80529096618b3ccf81f02f89572075`
- `49999`: final checkpoint, SHA-256 `8c2267b86dbda5e8987452f1717da0d1fb00e581fb7e21e2da4ec88de3746ecc`

Checkpoints are stored as params-only artifacts under `checkpoints/<step>/params/`.

## Assets

- `assets/` contains normalization stats and dataset metadata used by this run.

## W&B

- [Project dashboard](https://wandb.ai/pravsels/openpi_block_tower_6mix)

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
