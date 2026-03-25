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

- loss_one_liner: Loss dropped from 0.14 to 0.015; higher plateau than baseline likely due to larger and more diverse dataset with dAgger data. Still decreasing.

## W&B
- synced: https://wandb.ai/pravsels/block_tower/runs/bcqxnzhs

## Status
- Started: Tuesday, Mar 24th, 2026
- Training in progress
