# Training Log — Build Block Tower Mixed

## Mode
- run_type: experiment
- objective: Train on all 6 block tower datasets with mixed advantage prompts (human → positive, policy → negative). Compare against dyna (positive_only, drops policy frames) and baseline (no advantage prompts).

## Config
- config: `pi05_build_block_tower_mixed`
- exp_name: `mixed_v1`
- dataset: 6 datasets — `villekuosmanen/build_block_tower` + 5 dAgger iterations (LeRobot v2.1)
- key settings: 7D joint state/action, delta actions, mixed advantage conditioning, 100k steps, batch_size=36, lr=5e-5 cosine, from pi0.5 base weights

## Training Dynamics

| Step | Loss | Grad Norm |
|------|------|-----------|
| 0 | 0.1413 | 1.0680 |
| 25,000 | 0.0207 | 0.0903 |
| 30,000 | 0.0179 | 0.0825 |
| 50,000 | 0.0144 | 0.0693 |
| 75,000 | 0.0112 | 0.0627 |
| 99,900 | 0.0097 | 0.0552 |

- loss_one_liner: Steady decline from 0.14 to ~0.01 over 100k steps; grad norm stabilized early, no sign of plateau or overfitting.

## W&B
- synced: https://wandb.ai/pravsels/block_tower/runs/jhyepslj

## Checkpoint Hashes

Verify with:

```bash
cd checkpoints/<step> && find params -type f | sort | xargs sha256sum | sha256sum
```

| Step | SHA-256 |
|------|---------|
| 25,000 | `6d44e6b2aec69b964e974ffcf551834ac443196a36d85e891d9f246859a1afb1` |
| 30,000 | `d9544183f5a70f044f044c103c551c57588aaeaf3ff1b46b4159cd10fef6f528` |
| 50,000 | `3f289b60f8f4d9676ff3250aae41919355045d72b92cd7d4e0a21ea9071dea91` |
| 75,000 | `4a2b9d394f89bdb4ba2e3620eb774bba5ba9a8c829e85e2319d1dd1adb2bd03d` |

## Status
- Started: Friday, Mar 27th, 2026
- Completed: Saturday, Mar 28th, 2026
- Runtime: 17:12:33
- Published checkpoints: 25k, 30k, 50k, 75k (params-only)
