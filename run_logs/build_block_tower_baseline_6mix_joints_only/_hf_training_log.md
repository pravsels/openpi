# Training Log — Build Block Tower 6-Mix Joints-Only Ablation

## Mode
- run_type: experiment
- objective: Train block-tower policy restricted to the first 7 joint dims; EEF channels zeroed and excluded from loss via `action_dim_mask`.

## Config
- config: `pi05_build_block_tower_baseline_6mix_joints_only`
- exp_name: `joints_only`
- dataset: 6 HuggingFace datasets (`villekuosmanen/build_block_tower` + DAgger rounds 1.0.0–1.4.0, 340 episodes)
- key settings: 17D canonical action space, `joints_only=True` (`action_dim_mask=[True]*7+[False]*10`), delta actions, per-timestep action normalization, 50k steps, batch_size=36, lr=5e-5 cosine (1k warmup), from pi0.5 base weights

## Training Dynamics

| Step | Loss | Grad Norm |
|------|------|-----------|
| 0 | 0.1980 | 4.6369 |
| 5,000 | 0.0341 | 0.2118 |
| 10,000 | 0.0306 | 0.1508 |
| 15,000 | 0.0231 | 0.1136 |
| 20,000 | 0.0197 | 0.1023 |
| 25,000 | 0.0163 | 0.0808 |
| 30,000 | 0.0144 | 0.0772 |
| 35,000 | 0.0139 | 0.0740 |
| 40,000 | 0.0129 | 0.0735 |
| 45,000 | 0.0121 | 0.0680 |
| 49,900 | 0.0112 | 0.0666 |

- loss_one_liner: Steep drop from 0.20 to ~0.03 in the first 5k, then steady decline to 0.011 by 50k; no plateau or instability.

## W&B
- synced: https://wandb.ai/pravsels/build_block_tower_baseline_6mix_joints_only/runs/u6z1ph6k

## Checkpoint Hashes

Verify with:

```bash
cd checkpoints/<step> && find params -type f | sort | xargs sha256sum | sha256sum
```

| Step | SHA-256 |
|------|---------|
| 49,999 | `7cd730580e57c6ae390ff2e0fc41491941b6b4d10e0e5cf959e6243034eabb45` |

## Status
- Started: Tuesday, Apr 21st, 2026 (00:08 UTC)
- Completed: Tuesday, Apr 21st, 2026 (11:34 UTC)
- Runtime: 11:26:05
- Published checkpoints: 49999 (params-only)
