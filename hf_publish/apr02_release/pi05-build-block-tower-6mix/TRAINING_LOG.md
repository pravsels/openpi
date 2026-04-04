# Training Log

## Summary

- config: `pi05_build_block_tower_baseline`
- objective: train the build-block-tower baseline on the base dataset plus five DAgger rounds using the synced `baseline` checkpoint/output path
- dataset: 6 HuggingFace datasets: `villekuosmanen/build_block_tower` plus `dAgger_build_block_tower_1.0.0` through `1.4.0`
- synced_wandb_project: `https://wandb.ai/pravsels/openpi_block_tower_6mix`

## Uploaded Checkpoints

- `30000`: intermediate checkpoint, SHA-256 `3f0e7a56c29623df26809b19f698e7ee60232a5f203c820bc9d8b248413e2119`
- `40000`: intermediate checkpoint, SHA-256 `1022fe15232cc345a4034172e626cfd22f80529096618b3ccf81f02f89572075`
- `49999`: final checkpoint, SHA-256 `8c2267b86dbda5e8987452f1717da0d1fb00e581fb7e21e2da4ec88de3746ecc`

## Notes

- Sanitized for Hugging Face publication.
- Cluster-specific job IDs, node names, and scratch paths omitted.
