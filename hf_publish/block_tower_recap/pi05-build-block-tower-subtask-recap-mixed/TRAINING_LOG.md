# Training Log

## Summary

- config: `pi05_build_block_tower_subtask_recap_mixed`
- objective: train block tower recap with mixed advantage conditioning and hierarchical subtask prompting
- dataset: 6 HuggingFace datasets: `villekuosmanen/build_block_tower` plus 5 DAgger rounds (1.0.0 through 1.4.0)
- key settings: 17D action space, delta actions, per-timestep action normalization, advantage dropout 0.3, subtask_loss_weight 1.0, batch_size 128, FSDP across 4 GPUs, target 50k steps (stopped at 22k)
- synced_wandb_project: `https://wandb.ai/pravsels/openpi_block_tower_recap`
- wandb_run: `https://wandb.ai/pravsels/openpi_block_tower_recap/runs/jrtdxg3h`

## Uploaded Checkpoints

- `22000`: final checkpoint (22k/50k steps), SHA-256 `7addc02dc6542c61b08b8f608d5784d7aa1d5bbfc47d85a4c3ba9c8bf2af29f6`

## Notes

- Sanitized for Hugging Face publication.
- Cluster-specific job IDs, node names, and scratch paths omitted.
- Training stopped at 22k steps due to walltime limits; subtask variants required OOM mitigations (platform allocator, FSDP) and batch-size ramp (32 → 64 → 128).
