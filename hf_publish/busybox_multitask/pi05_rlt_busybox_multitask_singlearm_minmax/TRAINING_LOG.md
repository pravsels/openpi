# Training log — pi05_rlt_busybox_multitask_singlearm_minmax

RLT Stage 1 on frozen `pravsels/pi05_busybox_multitask_minmax`. VLA frozen (`rl_vla_loss_weight=0.0`). Per-timestep min/max norm. 20k steps, batch 16, 1× A100 80GB.

| Step | Loss | Grad norm |
|------|------|-----------|
| 0 | 10755.68 | 33158.02 |
| 12900 | 358.34 | 1229.78 |
| 17200 | 315.43 | 1028.22 |
| 19900 | 293.81 | 999.75 |

Runtime 4h 13m (2026-09-03 12:16 UTC → 16:28 UTC). Exit 0. Checkpoint step 19999.

W&B: https://wandb.ai/pravsels/busybox_multitask_rlt_singlearm_minmax/runs/93keszgb
