# Training Log — ARX5 Multitask Micro Baseline

## Mode
- run_type: experiment
- objective: Fine-tune PI0.5 on the micro training mix (14 datasets) with valid-index filtering (human-controlled + successful episodes).

## Config
- config: `pi05_arx5_multitask_micro_baseline`
- exp_name: `micro_baseline_v1`
- dataset: `training_mix_micro.json` — 14 `villekuosmanen/*` LeRobot repos
- key settings: 14D bimanual action space (7D padded), delta actions (delta joints, absolute grippers), per-timestep action normalization, 30k steps, batch_size=36, lr=5e-5 cosine (1k warmup), from pi0.5 base weights

## Training Dynamics

| Step | Loss | Grad Norm |
|------|------|-----------|
| 0 | 0.1664 | 2.3741 |
| 5,000 | 0.0274 | 0.1175 |
| 10,000 | 0.0197 | 0.0886 |
| 15,000 | 0.0159 | 0.0789 |
| 20,000 | 0.0133 | 0.0677 |
| 25,000 | 0.0119 | 0.0655 |
| 29,900 | 0.0107 | 0.0592 |

- loss_one_liner: Steep drop from 0.17 to ~0.03 in the first 5k, then steady decline to 0.011 by 30k; no plateau or overfitting.

## W&B
- synced: https://wandb.ai/pravsels/arx5_multitask/runs/gtk5f6zw

## Checkpoint Hashes

Verify with:

```bash
cd checkpoints/<step> && find params -type f | sort | xargs sha256sum | sha256sum
```

| Step | SHA-256 |
|------|---------|
| 25,000 | `69ee51b80032d3a4424bd3834167fdd4d839701ab3b267c73ae6b7386922f1f8` |
| 29,999 | `450e1c86c1d95ccb7215cc3662b90c6b56fb483006b640dfa2bc70bfa2593c01` |

## Status
- Started: Friday, Mar 27th, 2026
- Completed: Friday, Mar 27th, 2026
- Runtime: 04:54:46
- Published checkpoints: 25k, 29999 (params-only)
