# Train — pi0.5 SO101 stacking rings (Isambard u6kr)

## Mode
- run_type: replication (parallel to GCloud run)
- objective: same pi0.5 fine-tuning on Isambard GH200 cluster

## Config
- script: `slurm/train_so101_stacking_rings_slurm.sh`
- config: `pi05_so101_stacking_rings`
- dataset: `lorenzouttini/so101_stacking_rings` (101 episodes, 34k frames)
- exp_name: `so101_stacking_rings`
- same hyperparams as GCloud (see `2026-06-02_train.md`)

## Infrastructure
- cluster: Isambard-AI, project u6kr
- hardware: 4× GH200 (exclusive node, 1-day walltime max)
- worktree: `/home/u6kr/pravsels.u6kr/openpi_so101_stacking_rings` @ `task/stack_rings`
- SIF: `/scratch/u6kr/pravsels.u6kr/openpi/container/openpi_arm64.sif` (14G, HF bridge from u6cr)
- base weights: `/scratch/u6kr/pravsels.u6kr/openpi/weights/pi05_base` → absolute symlink to `openpi-assets/checkpoints/pi05_base` (12G, GCS)
- assets: `/scratch/u6kr/pravsels.u6kr/openpi/assets/pi05_so101_stacking_rings/so101_stacking_rings/assets/`
- checkpoints: `/scratch/u6kr/pravsels.u6kr/openpi/checkpoints/pi05_so101_stacking_rings/so101_stacking_rings/`
- venv: `/scratch/u6kr/pravsels.u6kr/openpi/.venv` (6.3G)
- wandb: offline at `/scratch/u6kr/pravsels.u6kr/openpi/wandb/`
- secrets: `/scratch/u6kr/pravsels.u6kr/.secrets/{.hf_token,.wandb_token}`

## Status
- 2026-06-03 00:11 UTC — job `4997330` COMPLETED (exit 0), 50k steps, ~7h38m on nid010516; final loss ~0.0058 @ step 49900
- 2026-06-02 16:33 UTC — job `4997330` running on nid010516; training started (weights loading)
- earlier failed attempts:
  - `4996181` — assets wiped by `--overwrite` (norm stats lost)
  - `4996918` — `--assets-dir` path didn't end with `/assets`
  - `4996973` — bare `python` instead of `uv run python`
  - `4997108` — relative weights symlink broken inside container bind mount

## Fixes applied (commits on `task/stack_rings`)
- `75e7e0b` — move assets outside checkpoint tree, generate valid_indices
- `d93684d` — assets path must end with `/assets`
- `f939aee` — use `uv run python` for valid_indices generation
- weights symlink made absolute on cluster (not in script, manual fix)

## Results
- checkpoints: `/scratch/u6kr/pravsels.u6kr/openpi/checkpoints/pi05_so101_stacking_rings/so101_stacking_rings/` (5k–45k, 49999; ~417G)
- slurm logs: `openpi_so101_stacking_rings/slurm-4997330.{out,err}`

## W&B
- local: `/scratch/u6kr/pravsels.u6kr/openpi/wandb/offline-run-20260602_163334-ox7qwnsz` (run id `ox7qwnsz`)
- project: `so101_stacking_rings_isambard`
- synced: https://wandb.ai/pravsels/so101_stacking_rings_isambard/runs/ox7qwnsz (2026-06-03)

## Passport
- 2026-06-03 — `PASSPORT_SEED.json` + `MODEL_PASSPORT.json` + `SIGNOFF.json` on step `49999` (soft signals only; signed)
- publish package: `autohpc_runs/so101_isambard_49999/hf_publish` (~12G, params-only)

## HF
- repo: https://huggingface.co/pravsels/pi05-so101-stacking-rings-isambard
- published: `assets/`, `params/`, passport/signoff, `README.md`, `TRAINING_LOG.md` (2026-06-03)

## Next
- Compare loss curve to GCloud run (`so101_stacking_rings` / `tmrsnyct`)
- Eval on robot/sim (`eval-tracking`) when ready
