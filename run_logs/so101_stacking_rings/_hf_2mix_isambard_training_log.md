# Training Log — pi0.5 SO101 Stacking Rings 2-mix (Isambard)

## Mode
- run_type: experiment
- objective: Fine-tune pi0.5 on teleop + rollout-correction mix to improve ring stacking success rate

## Config
- config: `pi05_so101_stacking_rings`
- exp_name: `so101_stacking_rings_2mix`
- script: `slurm/train_so101_stacking_rings_slurm.sh`
- datasets:
  - [lorenzouttini/so101_stacking_rings](https://huggingface.co/datasets/lorenzouttini/so101_stacking_rings) (101 episodes, ~34k frames)
  - [lorenzouttini/rollout_so101_stacking_rings_20260603_154953](https://huggingface.co/datasets/lorenzouttini/rollout_so101_stacking_rings_20260603_154953) (100 episodes, ~28k frames)
- key settings: pi0.5, action_horizon=30, batch_size=32, 50k steps, save_interval=5000, lr=2.5e-5 cosine (1k warmup), delta actions, init from pi05_base

## Job
- cluster: Isambard-AI u6kr
- execution_id: 5028123
- runtime: 07:43:01
- started: 2026-06-03 23:16 UTC
- finished: 2026-06-04 06:59 UTC

## Training dynamics

| Step | Loss (approx) |
|------|---------------|
| 0 | 0.2561 |
| 49,900 | 0.0074 |

- loss_one_liner: Loss fell from ~0.26 at step 0 to ~0.007 by 50k; stable low single-digit ×10⁻² range after ~20k.

## W&B
- synced: https://wandb.ai/pravsels/so101_stacking_rings_2mix_isambard/runs/y4z7w346 (2026-06-04)

## Checkpoint hashes

```bash
cd checkpoints/49999/params && find . -type f | sort | xargs sha256sum | sha256sum
```

| Step | SHA-256 (params tree) |
|------|------------------------|
| 49,999 | `e93a3d4f2a372c72d33f600c36326d5cd148ea9682f4c7bfb8153fe2796d7fcb` |

## HF publish
- repo: https://huggingface.co/pravsels/pi05-so101-stacking-rings-2mix-isambard
- revision: `main`
- published step: **49999** under `checkpoints/49999/params/` (params-only)
- also includes: `assets/`, `MODEL_PASSPORT.json`, `SIGNOFF.json`, `README.md`, `TRAINING_LOG.md`

## Status
- 2026-06-04 — training completed (50k steps)
- 2026-06-04 — W&B synced; passport + signoff on step 49999
