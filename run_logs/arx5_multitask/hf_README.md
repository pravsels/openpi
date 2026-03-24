---
license: apache-2.0
tags:
  - robotics
  - pi0
  - openpi
  - arx5
  - manipulation
  - imitation-learning
---

# PI0.5 ARX5 Multitask v1

PI0.5 fine-tuned on 186 ARX5 single-arm and bimanual manipulation datasets. Supports mixed 7D (single-arm) and 14D (bimanual) action spaces with per-dimension loss masking.

## Model Details

- **Base model:** PI0.5 (pretrained `pi05_base`)
- **Architecture:** `Pi0Config(pi05=True, action_horizon=50)`
- **Action space:** 14D joint-space (bimanual). Single-arm datasets are zero-padded with `action_dim_mask` to ignore unused dimensions during training.
- **Action type:** Absolute (not delta)
- **Training framework:** [openpi](https://github.com/Physical-Intelligence/openpi) (JAX/Flax, Orbax checkpoints)

## Training

- **Datasets:** 186 LeRobot v2.1 datasets (see `assets/training_mix_v1.json` for full list)
- **Prompts:** Subtask descriptions extracted from dataset metadata
- **Steps:** 100,000 (training still in progress; checkpoints uploaded at milestones)
- **Batch size:** 36
- **Learning rate:** Cosine decay, 10k warmup, peak 5e-5
- **Optimizer:** AdamW, gradient clip norm 1.0
- **EMA decay:** 0.999
- **Hardware:** 4x GH200 GPU on Isambard AI cluster

### Loss Progression

| Step | Train Loss | Grad Norm |
|------|-----------|-----------|
| 0 | 0.1667 | 1.665 |
| 5,000 | 0.0226 | 0.131 |
| 10,000 | 0.0219 | 0.108 |
| 15,000 | 0.0204 | 0.098 |
| 20,000 | 0.0187 | 0.086 |
| 25,000 | 0.0160 | 0.079 |
| 30,000 | 0.0160 | 0.081 |
| 35,000 | 0.0151 | 0.076 |
| 40,000 | 0.0140 | 0.074 |
| 45,000 | 0.0129 | 0.068 |
| 50,000 | 0.0126 | 0.069 |
| 55,000 | 0.0116 | 0.065 |

Full training curves: [W&B dashboard](https://wandb.ai/pravsels/pi05-arx5-multitask/runs/1vj1v4h4)

## Available Checkpoints

| Checkpoint | Train Loss | SHA-256 (tar of params dir) |
|-----------|-----------|------|
| `checkpoints/25000/params` | 0.016 | `b856b219...c86f6218` |
| `checkpoints/40000/params` | 0.014 | `ad865b75...b81c9341` |
| `checkpoints/55000/params` | 0.012 | `7ee69681...82fe5b2f` |

Checkpoints contain model parameters only (Orbax format). Optimizer/training state is not included.

<details>
<summary>Full SHA-256 hashes</summary>

```
25000: b856b2198f0f04791b52257eb20a78072aa6612970a35f9864d45cefc86f6218
40000: ad865b75714a0d8d057074b45639cc18718dcdfed3b2ec2479cded32b81c9341
55000: 7ee69681991cdc5e04b4759d3bf93bca5dac6bc98639ec7b00202d2f82fe5b2f
```

Verify with: `tar cf - -C checkpoints/<step> params | sha256sum`
</details>

## Assets

- `assets/norm_stats.json` — 14-dim per-dimension quantile normalisation statistics
- `assets/training_mix_v1.json` — list of all 186 dataset repo IDs
- `assets/valid_indices.txt` — ~1.4M filtered frame indices

## Usage

Load a checkpoint in openpi:

```python
from openpi.training import weight_loaders

loader = weight_loaders.CheckpointWeightLoader("path/to/checkpoints/55000/params")
```

Or reference it in a training config for further fine-tuning:

```python
TrainConfig(
    ...
    weight_loader=weight_loaders.CheckpointWeightLoader("path/to/checkpoints/55000/params"),
)
```

## Citation

If you use this model, please cite [openpi](https://github.com/Physical-Intelligence/openpi).
