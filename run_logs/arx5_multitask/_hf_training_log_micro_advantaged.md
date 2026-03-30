# Training Log — ARX5 Multitask Micro Advantaged

## Mode
- run_type: experiment
- objective: Fine-tune PI0.5 on the micro training mix (14 datasets) with advantaged valid-index filtering; compare to baseline variant.

## Config
- config: `pi05_arx5_multitask_micro_advantaged`
- exp_name: `micro_advantaged_v1`
- dataset: `training_mix_micro.json` — 14 `villekuosmanen/*` LeRobot repos
- key settings: 14D bimanual action space (7D padded), delta actions (delta joints, absolute grippers), per-timestep action normalization, 30k steps, batch_size=36, lr=5e-5 cosine (1k warmup), from pi0.5 base weights

## Training Dynamics

| Step | Loss | Grad Norm |
|------|------|-----------|
| 0 | 0.1770 | 2.3837 |
| 5,000 | 0.0215 | 0.1070 |
| 10,000 | 0.0146 | 0.0805 |
| 15,000 | 0.0118 | 0.0665 |
| 20,000 | 0.0101 | 0.0620 |
| 25,000 | 0.0089 | 0.0551 |
| 29,900 | 0.0080 | 0.0540 |

- loss_one_liner: Steep drop from 0.18 to ~0.02 in the first 5k, then steady decline to 0.008 by 30k; lower final loss than baseline (0.0080 vs 0.0107).

## W&B
- synced: https://wandb.ai/pravsels/arx5_multitask/runs/jik4rmpl

## Checkpoint Hashes

Verify with:

```bash
cd checkpoints/<step> && find params -type f | sort | xargs sha256sum | sha256sum
```

| Step | SHA-256 |
|------|---------|
| 25,000 | `1648c67a7ac44d377f28f316384bdcab72af4422237f9f9485e1e77a02c6a65c` |
| 29,999 | `aff337d89dd426388303855ed8fca784f5b5615b33cbad14f26dfbe8688caa88` |

## Status
- Started: Sunday, Mar 29th, 2026
- Completed: Sunday, Mar 29th, 2026
- Runtime: 04:52:03
- Published checkpoints: 25k, 29999 (params-only)
