# Training Log

## Summary

- config: `pi05_build_block_tower_recap_positive_only`
- objective: train block tower recap with positive-only advantage conditioning (flat prompt, no subtask hierarchy)
- dataset: 6 HuggingFace datasets: `villekuosmanen/build_block_tower` plus 5 DAgger rounds (1.0.0 through 1.4.0)
- key settings: 17D action space, delta actions, per-timestep action normalization, advantage dropout 0.3, batch_size 36, 4 GPUs, 50k steps
- synced_wandb_project: `https://wandb.ai/pravsels/openpi_block_tower_recap`

## Uploaded Checkpoints

- `30000`: intermediate checkpoint, SHA-256 `cb05d0f00920563db4bb556001b28c739cdf19c4b327ffb7fb7e4f558e58c076`
- `40000`: intermediate checkpoint, SHA-256 `cb1cd2e9198c2090d01c7c0c671001a3e553db1d4c43f297bf37f10764285912`
- `49999`: final checkpoint, SHA-256 `92a5dcfcac63b986815f06b3f4c1d34e048b0f30771fb67e4a5906e1c7bf3130`

## Notes

- Sanitized for Hugging Face publication.
- Cluster-specific job IDs, node names, and scratch paths omitted.
