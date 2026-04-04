# reward_recap fix positive_only — post-placement + valid-indices update

## Mode
- run_type: experiment
- objective: rerun bin-pack positive-only recap after fixing advantage-token placement, config-aware valid-index persistence, and shortening the run to 50k steps

## Config
- script: `slurm/train_bin_pack_reward_recap_slurm.sh positive_only`
- config: `pi05_bin_pack_coffee_capsules_recap_positive_only` (`src/openpi/training/config.py`)
- dataset: 9 datasets (bin_pick_pack + 8 dAgger rounds) on HuggingFace
- key settings: 17D action space, delta actions enabled, per-timestep action normalization, advantage dropout `0.3`, positive-only recap semantics, `num_train_steps=50_000`, code pulled on HPC at commit `e26281c`

## Job
- job_id: 3586066
- submitted/start: `2026-04-02T16:37:23Z`
- start_human: Thursday, Apr 2nd, 2026
- end: `2026-04-02T16:38:08Z`
- end_human: Thursday, Apr 2nd, 2026
- runtime: `00:00:43`
- node: `nid010225`

## Job (resubmitted)
- job_id: 3586381
- submitted/start: `2026-04-02T17:36:12Z`
- start_human: Thursday, Apr 2nd, 2026
- resumed from: fresh restart after fixing wrapped valid-indices auto-generation in commit `cacf4a1`
- node: `nid010617`

## Job (resubmitted again)
- job_id: 3586895
- submitted/start: `2026-04-02T18:38:26Z`
- start_human: Thursday, Apr 2nd, 2026
- resumed from: fresh resubmission after syncing commit `e8ca9da`
- node: `nid010380`
- synced: `https://wandb.ai/pravsels/openpi_recap_fix/runs/gzi3ryxp`

## Status
- 2026-04-02 17:24 UTC — failed after normalization precompute; training hit `AttributeError: 'TransformedDataset' object has no attribute '_datasets'` in `src/openpi/training/valid_indices.py`
- 2026-04-02 17:36 UTC — resubmitted as job `3586381` after unwrapping wrapped datasets in the valid-indices auto-generation path
- 2026-04-02 17:36 UTC — running on `nid010617`; skips normalization precompute using existing assets and reaches `scripts/train.py` startup with no traceback so far
- 2026-04-02 18:38 UTC — previous running jobs were manually cancelled for a clean restart; resubmitted as job `3586895` after syncing commit `e8ca9da`
- 2026-04-02 18:44 UTC — loaded norm stats, per-timestep action stats, and printed `data_config`; still running on `nid010380` but has not yet emitted `Step 0`
- 2026-04-03 03:07 UTC — completed successfully as job `3586895` (exit code `0`); synced W&B run to `pravsels/openpi_recap_fix`
- 2026-04-04 — uploaded self-contained Hugging Face checkpoints for `30000`, `40000`, and `49999` to `pravsels/pi05-bin-pack-positive-only-fix`

## Results
- runtime: `08:28:24` (start `2026-04-02T18:38:26Z`, end `2026-04-03T03:06:50Z`)
- final step: `49999`
- start_train_loss: `0.3194` (step `0`)
- end_train_loss: `0.0065` (step `49900`)
- loss_one_liner: Positive-only recap also converged cleanly, dropping from ~`0.32` to ~`0.0065` over 50k steps after the recap fix rerun.
- checkpoints_uploaded:
  - `30000` — SHA-256 `489b7aead63b6bd715aaa55305eeeabef4dc70634d79eeca9d2d384e1d416052`
  - `40000` — SHA-256 `d53517fa406be906f64e26a9e7eb3c15f84e1809f10f46074c7dce6b43b9ab3a`
  - `49999` — SHA-256 `a880a381e90a16706a24699f5a95c6554e7bb10f6445c5fdbccc169846205f52`

## W&B
- local: `/scratch/u6cr/pravsels.u6cr/openpi/wandb/offline-run-20260402_184300-gzi3ryxp`
- synced: `https://wandb.ai/pravsels/openpi_recap_fix/runs/gzi3ryxp`
- notes:

## HuggingFace
- repo: https://huggingface.co/pravsels/pi05-bin-pack-positive-only-fix
- uploaded checkpoints: `30000`, `40000`, `49999` (self-contained `params` + `assets`)
- includes: `README.md`, `TRAINING_LOG.md`, `checkpoints/<step>/params`, `checkpoints/<step>/assets`

## Next
- ready to use as the published positive-only recap fix checkpoint set
