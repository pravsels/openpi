# Training Log — Build Block Tower Baseline

## Mode
- run_type: experiment
- objective: train a block tower policy from pi0.5 base weights using 200 human demos (imitation learning only, no advantage conditioning)

## Config
- config: `pi05_build_block_tower_baseline`
- dataset: `villekuosmanen/build_block_tower` (200 episodes, LeRobot v2.1)
- key settings: 7D joint state/action, delta actions, 100k steps, batch_size=36, lr=5e-5 cosine, from pi0.5 base weights

## Training Dynamics

| Step | Loss | Grad Norm |
|------|------|-----------|
| 0 | 0.1514 | 1.1447 |
| 25,000 | 0.0111 | 0.0731 |
| 50,000 | 0.0077 | 0.0600 |

- loss_one_liner: Loss dropped steadily from 0.15 to 0.007 with no sign of plateau; healthy training throughout.

## W&B
- synced: https://wandb.ai/pravsels/block_tower/runs/6hoa4kt5

## Status
- Started: Tuesday, Mar 24th, 2026
- Training in progress
