# Training log — pi05_rlt_busybox_multitask_singlearm

RLT Stage 1 on frozen `pravsels/pi05_busybox_multitask`. VLA frozen (`rl_vla_loss_weight=0.0`). 20k steps, batch 16, 1× A100 80GB.

| Step | Loss | Grad norm |
|------|------|-----------|
| 0 | 12117.80 | 38526.98 |
| 7500 | 386.30 | 1141.76 |
| 19000 | 249.69 | 727.33 |
| 19900 | 246.70 | 738.00 |

Runtime 4h 10m (2026-09-03 10:46 UTC → 14:56 UTC). Exit 0. Checkpoint step 19999.

W&B: https://wandb.ai/pravsels/busybox_multitask_rlt_singlearm/runs/xmkdxvrl
