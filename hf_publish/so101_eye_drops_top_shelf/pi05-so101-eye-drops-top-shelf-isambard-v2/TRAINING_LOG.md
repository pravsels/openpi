# Training Log — pi0.5 SO101 Eye Drops Top Shelf v2 (Isambard)

## Summary

- config: `pi05_so101_eye_drops_top_shelf_v2`
- exp_name: `so101_eye_drops_top_shelf_v2`
- objective: Fine-tune pi0.5 on SO101 eye-drops top-shelf placing task with delta actions (101 episodes, +50 vs v1)
- dataset: lorenzouttini/so101_eye_drops_top_shelf2_20260609_160053
- key settings: pi0.5, action_horizon=30, batch_size=32, 50k steps, save_interval=5000, lr=2.5e-5 cosine (1k warmup), delta actions, per-timestep action norm, ema_decay=0.999, base weights pi05_base
- hardware: Isambard GH200 (4× GPU node, arm64)
- container: pravsels/openpi-isambard — openpi_arm64.sif
- wandb_project: so101_eye_drops_top_shelf (offline; to be synced)
- slurm_job: 5145319

## Uploaded Checkpoints

- step_25000: halfway checkpoint
- step_49999: final checkpoint

## Notes

- Sanitized for Hugging Face publication.
- Cluster-specific job IDs, node names, and scratch paths omitted.
