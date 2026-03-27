# Training Log — Build Block Tower Dyna

## Mode
- run_type: experiment
- objective: train a block tower policy using all human + dAgger data with Dyna-style conditioning (positive-only advantage)

## Config
- config: `pi05_build_block_tower_dyna`
- dataset: 6 datasets — `villekuosmanen/build_block_tower` + 5 dAgger iterations (LeRobot v2.1)
- key settings: 7D joint state/action, delta actions, positive_only advantage conditioning, 100k steps, batch_size=36, lr=5e-5 cosine, from pi0.5 base weights

## Training Dynamics

| Step | Loss | Grad Norm |
|------|------|-----------|
| 0 | 0.1351 | 1.0301 |
| 25,000 | 0.0212 | 0.0945 |
| 50,000 | 0.0145 | 0.0701 |
| 80,000 | ~0.0115 | ~0.0604 |
| 81,900 | 0.0111 | 0.0593 |

- loss_one_liner: Loss dropped from 0.135 to around 0.011 by 81.9k, with stable checkpoint finalization through 81k and continued gradual improvement before manual stop.

## W&B
- synced: https://wandb.ai/pravsels/block_tower/runs/bcqxnzhs
- synced_step: 81600 (after force-resync of both offline chunks)

## Checkpoint Hashes (params tar)

Verify with `tar cf - -C checkpoints/<step> params | sha256sum`.

| Step | SHA-256 |
|------|---------|
| 50,000 | `a6673bee89023ba6726fa5c2c7d9957c289b8cdd5427be0815720a31eb3f0164` |
| 80,000 | `564caf6cfdfe57d80733e8806f28005d6077189d7dfdcb8f2b4f5e6857dab827` |

Note: checkpoint `48,000` was deleted from HuggingFace.

## Status
- Started: Tuesday, Mar 24th, 2026
- Current published checkpoints: 25k, 50k, 80k (params-only)
