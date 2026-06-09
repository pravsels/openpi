# Train 2-mix — pi0.5 SO101 stacking rings (Isambard u6kr)

## Mode
- run_type: experiment
- objective: fresh pi0.5 fine-tune on 2-dataset mix (original teleop + rollout corrections) to improve ring stacking success rate

## Config
- script: `slurm/train_so101_stacking_rings_slurm.sh`
- config: `pi05_so101_stacking_rings`
- exp_name: `so101_stacking_rings_2mix`
- datasets:
  - `lorenzouttini/so101_stacking_rings` (101 episodes, ~34k frames, original teleop)
  - `lorenzouttini/rollout_so101_stacking_rings_20260603_154953` (100 episodes, ~28k frames, rollout corrections)
- key settings: lr 2.5e-5, batch 32, 50k steps, action_horizon 30, delta actions, init from pi05_base
- worktree: `/home/u6kr/pravsels.u6kr/openpi_so101_stacking_rings` @ `task/stack_rings`

## Job
- execution_id: 5028123
- submitted: 2026-06-03 23:16 UTC
- start_human: Wednesday, Jun 3rd, 2026
- end: 2026-06-04 06:59 UTC
- end_human: Thursday, Jun 4th, 2026
- runtime: 07:43:01
- exit_code: 0

## Status
- 2026-06-04 00:14 UTC — submitted, job 5028123 queued
- 2026-06-04 07:27 UTC — COMPLETED (exit 0), 50k steps in 7h43m; final loss ~0.0074 @ step 49900; checkpoints at 5k–49999
- 2026-06-04 09:20 UTC — W&B synced to `so101_stacking_rings_2mix_isambard` (run `y4z7w346`)
- 2026-06-04 08:23 UTC — passport + signoff on step 49999 (`soft_signal`, 6 soft signals pre-sign)

## Results
- runtime: 07:43:01 (start 2026-06-03T23:16:35 UTC, end 2026-06-04T06:59:29 UTC)
- final step: 49999
- start_train_loss: 0.2561 (step 0)
- end_train_loss: 0.0074 (step 49900)
- loss_one_liner: loss fell from ~0.26 at step 0 to ~0.007 by 50k; stable single-digit ×10⁻² range after ~20k
- checkpoint: `/scratch/u6kr/pravsels.u6kr/openpi/checkpoints/pi05_so101_stacking_rings/so101_stacking_rings_2mix/` (5k–49999)
- slurm logs: `openpi_so101_stacking_rings/slurm-5028123.{out,err}`

## W&B
- local: `/scratch/u6kr/pravsels.u6kr/openpi/wandb/offline-run-20260603_232650-y4z7w346` (run id `y4z7w346`)
- project: `so101_stacking_rings_2mix_isambard`
- synced: https://wandb.ai/pravsels/so101_stacking_rings_2mix_isambard/runs/y4z7w346 (2026-06-04)
- notes: pending — review curves on dashboard

## Passport
- 2026-06-04 — `PASSPORT_SEED.json` + `MODEL_PASSPORT.json` + `SIGNOFF.json` on step `49999` (verdict `soft_signal`; unpinned 2mix dataset commits, same class of soft signals as baseline)
- logs: `autohpc_runs/so101_2mix_49999/{extract,assemble,validate,sign}.log`
- HF publish signoff: `soft_signal` on staged package (41 artifacts, paths under `checkpoints/49999/params/`)

## HuggingFace
- repo: https://huggingface.co/pravsels/pi05-so101-stacking-rings-2mix-isambard
- published step: **49999** at `checkpoints/49999/params/` (params-only, not top-level `params/`)
- includes: `assets/`, `MODEL_PASSPORT.json`, `SIGNOFF.json`, `README.md`, `TRAINING_LOG.md`
- package: `autohpc_runs/so101_2mix_49999/hf_publish` (~12G)
- uploaded: 2026-06-04

## Next
- eval on robot/sim vs `2026-06-02_train_isambard` baseline (`eval-tracking/SKILL.md`)
