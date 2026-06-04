# Training Log — pi0.5 SO101 Stacking Magnetic Cubes (Isambard)

## Mode
- run_type: replication
- objective: First pi0.5 fine-tune on magnetic-cube stacking (Isambard)

## Config
- config: `pi05_so101_stacking_magnetic_cubes`
- exp_name: `so101_stacking_magnetic_cubes`
- script: `slurm/train_so101_stacking_magnetic_cubes_slurm.sh`
- dataset: [lorenzouttini/so101_stacking_magnetic_cubes](https://huggingface.co/datasets/lorenzouttini/so101_stacking_magnetic_cubes)
- key settings: pi0.5, action_horizon=30, batch_size=32, 50k steps (target), save_interval=5000, lr=2.5e-5 cosine, delta actions, init from pi05_base

## Job
- cluster: Isambard-AI u6kr
- execution_id: 5038584
- started: 2026-06-04 11:50 UTC
- status at 10k publish: training still running toward 50k

## Training dynamics (partial)

| Step | Loss (approx) |
|------|---------------|
| 0 | 0.4056 |
| 5,000 | 0.0248 |
| 10,000 | 0.0179 |

## W&B
- synced: https://wandb.ai/pravsels/so101_stacking_magnetic_cubes/runs/7h1jiwva (2026-06-04, partial while training continues)

## HF publish (interim)
- repo: https://huggingface.co/pravsels/pi05-so101-stacking-magnetic-cubes-isambard
- published step: **10000** under `step_10000/params/` (params-only)
- also includes: `assets/`, `MODEL_PASSPORT.json`, `SIGNOFF.json`, `README.md`, `TRAINING_LOG.md`

## Next
- complete 50k training, then optional `step_49999/params/` upload to same repo
