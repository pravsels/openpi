# Training Log — PI0.5 ARX5 Multitask v1

## Objective

Fine-tune PI0.5 on 186 ARX5 single-arm and bimanual datasets with subtask descriptions as prompts, absolute actions, and per-dimension loss masking for mixed 7/14-dim action spaces.

## Config

- **Config name:** `pi05_arx5_multitask_v1`
- **Experiment name:** `arx5_abs_v1`
- **Dataset:** `training_mix_v1.json` — 186 `villekuosmanen/*` LeRobot repos
- **Base weights:** PI0.5 pretrained (`pi05_base`)
- **Key settings:**
  - Model: `Pi0Config(pi05=True, action_horizon=50)`
  - Action dim: 14 (bimanual), single-arm padded with `action_dim_mask`
  - Batch size: 36
  - LR: cosine decay, warmup 10k steps, peak 5e-5, decay to 5e-5 over 1M steps
  - Optimizer: AdamW, gradient clip norm 1.0
  - EMA decay: 0.999
  - Num train steps: 100,000
  - use_delta_actions: False (absolute actions)
  - W&B: enabled (offline mode)
- **Precomputed assets:**
  - `norm_stats.json` — 14-dim per-dimension quantile stats
  - `valid_indices.txt` — ~1.4M filtered frame indices
  - `training_mix_v1.json` — dataset list

## Training History

### Attempt 1 (failed)

- Started: Sunday, Mar 22nd, 2026
- Reached step 2,000 (loss dropped from 0.167 to 0.025) before deadlocking on checkpoint finalisation. Project scratch quota was full (5TB/5TB), preventing checkpoint I/O. Training hung for the remaining ~23.5 hours and was killed by the walltime limit.
- No usable checkpoints from this attempt.
- Scratch cleaned from 4.4TB to 1.3TB before resubmission.

### Attempt 2 (successful)

- Started: Tuesday, Mar 24th, 2026
- Fresh start from base weights (no checkpoint to resume from).
- Training progressed smoothly at ~1.7 it/s on 4x GH200 GPU (Isambard AI cluster).

#### Status updates

- Step 28,500 (elapsed ~5h) — loss 0.0159, checkpoints at 5k–25k.
- Step 56,400 (elapsed ~9.5h) — loss 0.0118, checkpoints at 5k–55k (every 5k). Loss still declining but rate slowing.

## Loss Progression

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

Loss dropped steeply in the first 5k steps (0.167 to 0.023), then continued a slow steady decline. By 55k steps the rate of improvement is diminishing but not fully plateaued.

## W&B

Full training curves: [W&B dashboard](https://wandb.ai/pravsels/pi05-arx5-multitask/runs/1vj1v4h4)

## Next Steps

- If walltime-interrupted, resume with same script (`--resume` flag is set)
- Future experiment: enable delta actions (`use_delta_actions=True` with `DeltaActionsFromState`) and recompute norm stats
