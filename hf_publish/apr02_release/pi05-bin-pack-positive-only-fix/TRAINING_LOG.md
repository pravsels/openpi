# Training Log

## Summary

- config: `pi05_bin_pack_coffee_capsules_recap_positive_only`
- objective: rerun bin-pack positive-only recap after fixing advantage-token placement, config-aware valid-index persistence, and shortening the run to 50k steps
- dataset: 9 HuggingFace datasets: `bin_pick_pack_coffee_capsules` plus 8 DAgger rounds
- synced_wandb_project: `https://wandb.ai/pravsels/openpi_recap_fix`

## Uploaded Checkpoints

- `30000`: intermediate checkpoint, SHA-256 `489b7aead63b6bd715aaa55305eeeabef4dc70634d79eeca9d2d384e1d416052`
- `40000`: intermediate checkpoint, SHA-256 `d53517fa406be906f64e26a9e7eb3c15f84e1809f10f46074c7dce6b43b9ab3a`
- `49999`: final checkpoint, SHA-256 `a880a381e90a16706a24699f5a95c6554e7bb10f6445c5fdbccc169846205f52`

## Notes

- Sanitized for Hugging Face publication.
- Cluster-specific job IDs, node names, and scratch paths omitted.
