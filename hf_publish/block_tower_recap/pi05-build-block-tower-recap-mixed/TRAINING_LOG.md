# Training Log

## Summary

- config: `pi05_build_block_tower_recap_mixed`
- objective: train block tower recap with mixed advantage conditioning (flat prompt, no subtask hierarchy)
- dataset: 6 HuggingFace datasets: `villekuosmanen/build_block_tower` plus 5 DAgger rounds (1.0.0 through 1.4.0)
- key settings: 17D action space, delta actions, per-timestep action normalization, advantage dropout 0.3, batch_size 36, 4 GPUs, 50k steps
- synced_wandb_project: `https://wandb.ai/pravsels/openpi_block_tower_recap`

## Uploaded Checkpoints

- `30000`: intermediate checkpoint, SHA-256 `341308cd5f8be0236d26cde04a1a7494c9428da7e5559996578b7ed55fb6dabf`
- `40000`: intermediate checkpoint, SHA-256 `ec86899d8566a7c0e4f90839616f113fb0401d4288180d020d9d0f15f5972ef2`
- `49999`: final checkpoint, SHA-256 `8fbeda24b6593304cb0a6b42a5ec9dd342cdb5f0e1b25a074687785761eccf9f`

## Notes

- Sanitized for Hugging Face publication.
- Cluster-specific job IDs, node names, and scratch paths omitted.
