# Training Log

## Summary

- config: `pi05_bin_pack_coffee_capsules_recap_mixed`
- objective: rerun bin-pack mixed recap after fixing advantage-token placement, config-aware valid-index persistence, and shortening the run to 50k steps
- dataset: 9 HuggingFace datasets: `bin_pick_pack_coffee_capsules` plus 8 DAgger rounds
- synced_wandb_project: `https://wandb.ai/pravsels/openpi_recap_fix`

## Uploaded Checkpoints

- `30000`: intermediate checkpoint, SHA-256 `29fb62934e85c7fd18a3c10cd6b80b816eb323b8b4a7bdf0dab3a1847cdb95ed`
- `40000`: intermediate checkpoint, SHA-256 `d1b7d7c4b465662ad32dd091d56163be49c79b7259261457c136cfe414fa46ab`
- `49999`: final checkpoint, SHA-256 `1dbc8980ce3f860b05a34b28afd795596b82016ad458d1348b3ad476516b66c8`

## Notes

- Sanitized for Hugging Face publication.
- Cluster-specific job IDs, node names, and scratch paths omitted.
