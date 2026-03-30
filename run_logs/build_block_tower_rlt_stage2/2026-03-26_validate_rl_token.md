# build_block_tower RLT Stage 2 — RL token validation probe suite

## Mode
- run_type: experiment
- objective: validate that the frozen Stage 1 RL token is informative enough for downstream actor-critic work by training the full Stage 2 probe suite on held-out episodes

## Config
- script: `scripts/validate_rl_token.py`
- slurm: `slurm/validate_build_block_tower_rlt_stage2_slurm.sh`
- config: `pi05_rl_token_build_block_tower` (in `src/openpi/training/config.py`)
- checkpoint: `/scratch/u6cr/pravsels.u6cr/openpi/checkpoints/pi05_rl_token_build_block_tower/rlt_v1/9999/params`
- assets: `/scratch/u6cr/pravsels.u6cr/openpi/checkpoints/pi05_rl_token_build_block_tower/rlt_v1/9999/assets`
- output_dir: `/scratch/u6cr/pravsels.u6cr/openpi/eval_outputs/build_block_tower_rlt_stage2/validate_rl_token_9999`
- dataset: `villekuosmanen/build_block_tower` (200 episodes, LeRobot v2.1)
- split: deterministic episode-level `90/10` split from `assets/episode_split.json`
- key settings:
  - frozen model: `Pi0RL` loaded from Stage 1 checkpoint
  - probe suite:
    - action probe: `concat(rl_token, state) -> vla_action_chunk`
    - action comparison baseline: evaluate action probe output against both VLA actions and ground-truth demo actions
    - linear state probe: `rl_token -> normalized state`
    - random-feature baseline: `random_vector -> normalized state`
    - subtask classifier: `rl_token -> subtask logits` when labels are present
  - device: `cuda` (explicitly passed by the Slurm launcher)
  - probe model sizes: 2-layer MLP with hidden dim 256 for action probe and classifier; linear head for state probe and random-feature baseline
  - default probe hyperparams in script: `probe_epochs=40`, `lr=1e-3`, `probe_batch_size=256`, `num_denoising_steps=10`

## Probe Matrix
- `action_probe`
  - purpose: test whether `rl_token + state` preserves enough information to reconstruct what the frozen VLA would do
  - train target: sampled VLA action chunk
  - eval targets: VLA action chunk and ground-truth demo action chunk
- `linear_probe`
  - purpose: test whether low-dimensional control/state information is linearly recoverable from the RL token
  - train/eval target: normalized state
- `random_baseline`
  - purpose: test whether the linear state probe is learning real structure rather than exploiting target dimensionality alone
  - train/eval target: normalized state
- `subtask_classifier`
  - purpose: test whether the RL token separates semantic phases/subtasks
  - train/eval target: `subtask` labels from `SubtaskPlugin`, if populated

## Pre-Run Expectations
- The script should emit `metrics.json` and `features_and_predictions.npz` under a run-specific output directory.
- The action probe should fit the train split quickly and show a clearly lower validation error to VLA actions than to ground-truth demo actions.
- The linear state probe should outperform the random-feature baseline by a visible margin on validation MSE.
- The subtask classifier should only be treated as binding if the dataset actually contains non-empty `subtask` labels on both train and val splits.
- All probe variants should be reported separately in the final results rather than rolled into a single verdict.

## Expected Results
- `metrics.json` contains separate sections for `action_probe`, `linear_probe`, and `subtask_classifier`, plus the random-feature baseline nested under `linear_probe`.
- `action_probe.train_loss` decreases smoothly over epochs with no flatline at initialization scale.
- `action_probe.val_mse_to_vla` and `action_probe.val_l2_to_vla` decrease during training and settle at a stable value rather than diverging.
- `action_probe.val_l2_to_vla < action_probe.val_l2_to_ground_truth`.
- `linear_probe.val_mse < linear_probe.random_baseline_val_mse`.
- If subtask labels are populated:
  - `subtask_classifier.enabled == true`
  - `subtask_classifier.val_accuracy` is meaningfully above chance
- If subtask labels are not populated:
  - `subtask_classifier.enabled == false`
  - the run is still usable for Stage 2 decision-making based on the action probe + linear probe

## Interpretation Guide
- Strong pass:
  - action probe converges on train and val
  - VLA-target error is materially lower than demo-target error
  - linear probe beats random baseline
  - optional subtask classifier is above chance if labels exist
- Weak pass:
  - action probe learns but the generalization gap is large
  - linear probe only slightly beats random baseline
  - subtask labels missing or too sparse to judge
- Fail / investigate:
  - action probe does not reduce loss on train
  - validation metrics are unstable or worse than initialization
  - VLA-target error is not better than demo-target error
  - linear probe is indistinguishable from the random baseline
  - subtask classifier is at chance despite apparently valid labels

## Job (failed)
- job_id: 3378883
- submitted: `2026-03-26T17:21:56+00:00`
- start: `2026-03-26T20:14:58+00:00`
- end: `2026-03-26T20:14:59+00:00`
- runtime: 00:00:01
- node: nid010956
- failure: checkpoint `rlt_v1/9999/params` did not exist on scratch — the training run saved at steps 5000/15000/19999, not 9999. Downloaded step 9999 params from [HuggingFace](https://huggingface.co/pravsels/pi05-build-block-tower-rlt-v1/tree/main/checkpoints/9999) and copied assets from the 19999 checkpoint (identical across all steps).

## Job (resubmit 1 — failed)
- job_id: 3380023
- submitted: `2026-03-26T20:22:00+00:00`
- start: `2026-03-26T20:27:05+00:00`
- end: `2026-03-26T21:09:06+00:00`
- runtime: 00:42:01
- node: nid010854
- failure: `jaxtyping.TypeCheckError` — `tokenized_prompt` was a numpy array (`i64[32,200]`) but `gemma.Module.embed` expects `Int[Array, 'b t']` or `Int[Tensor, 'b t']`. Fix: convert batch to JAX arrays via `jnp.asarray` before `Observation.from_dict` (`2ac69d8`).

## Job (resubmit 2 — cancelled)
- job_id: 3386267 → cancelled before start
- note: replaced by resubmit 3 to pick up cleanup commit `1a88f37`.

## Job (resubmit 3 — failed)
- job_id: 3386482
- submitted: `2026-03-26T21:20:00+00:00`
- start: `2026-03-26T22:22:53+00:00`
- end: `2026-03-26T23:19:42+00:00`
- runtime: 00:56:49
- node: nid010651
- failure: `RESOURCE_EXHAUSTED` OOM in RL encoder FFN layer (`pi0_rl.py:113`) — batch_size=32 exceeds GPU memory for the full PaliGemma + RL encoder forward pass. Fix: reduce batch_size to 8 (`1c3095f`).

## Job (resubmit 4 — timed out)
- job_id: 3391484
- submitted: `2026-03-27T09:00:00+00:00`
- start_human: Friday, Mar 27th, 2026
- end: cancelled by Slurm (walltime)
- runtime: 12:00:00 (hit 12h limit)
- node: unknown
- failure: feature extraction too slow — processed ~6,357 of ~8,750 batches (batch_size=8, ~70k total samples) in 12h. No errors, no OOM; the script simply ran out of time during `_extract_split_features`. Root cause: full VLA forward pass + 10-step denoising per batch at batch_size=8 is ~530 batches/hr, needing ~16.5h total. Compounded by zero progress logging — appeared hung but was silently working.

## Job (resubmit 5)
- job_id: 3460713
- submitted: `2026-03-30`
- start: pending
- fixes applied:
  - added progress logging every 50 batches in `_extract_split_features` (`e822cca`)
  - capped samples: `--max-train-samples 5000 --max-val-samples 1000` (750 total batches, ~1.5h estimated extraction time)
  - added state-only action baseline (Probe 1b) and chance accuracy for subtask classifier (`e6c9752`)
- runtime: pending
- node: pending

## Status
- 2026-03-26 — created pre-submit Stage 2 validation log and recorded expected probe behavior before HPC launch.
- 2026-03-26 17:21 UTC — pulled `task/rlt_block_tower` on `openpi_rlt_block_tower` and submitted `slurm/validate_build_block_tower_rlt_stage2_slurm.sh` as job `3378883`.
- 2026-03-26 17:22 UTC — job `3378883` is pending in Slurm.
- 2026-03-26 20:14 UTC — job `3378883` failed after 1s: checkpoint `rlt_v1/9999/params` not found on scratch.
- 2026-03-26 20:22 UTC — downloaded step 9999 params from HuggingFace, copied assets from step 19999. Resubmitted as job `3380023`.
- 2026-03-26 21:09 UTC — job `3380023` failed after 42min: `tokenized_prompt` passed as numpy array, `gemma.Module.embed` requires JAX array. Fixed in `2ac69d8` by converting batch via `jnp.asarray`. Merged `main` into worktree, resubmitted as job `3386267`.
- 2026-03-26 21:20 UTC — cancelled pending job `3386267` to pick up cleanup commit `1a88f37` (avoid redundant GPU roundtrip, free JAX memory before probe training). Resubmitted as job `3386482`.
- 2026-03-26 23:19 UTC — job `3386482` OOM after 57min: XLA allocator exhausted during RL encoder FFN with batch_size=32. Reduced to batch_size=8 in `1c3095f`. Resubmitted as job `3391484`.
- 2026-03-30 — job `3391484` timed out after 12h. Diagnosed: extraction was working but too slow for the full dataset. Added progress logging and sample caps. Merged `origin/main` into `task/rlt_block_tower` worktree. Resubmitted as job `3460713`.

## Results
- runtime: pending
- output_dir: pending
- num_train_samples: pending (capped at 5,000)
- num_val_samples: pending (capped at 1,000)
- probe_suite_status: pending
- action_probe.val_mse_to_vla: pending
- action_probe.val_l2_to_vla: pending
- action_probe.val_mse_to_ground_truth: pending
- action_probe.val_l2_to_ground_truth: pending
- state_only_baseline.val_mse_to_vla: pending
- state_only_baseline.val_l2_to_vla: pending
- linear_probe.val_mse: pending
- linear_probe.random_baseline_val_mse: pending
- subtask_classifier.enabled: pending
- subtask_classifier.val_accuracy: pending
- subtask_classifier.chance_accuracy: pending
- verdict: pending

## W&B
- local: n/a
- synced: n/a
- notes: This script writes JSON/NPZ artifacts rather than using the training W&B flow by default.

## Next
- monitor job `3460713` — progress logs should appear every 50 batches
- after completion, copy the metrics for every probe variant into this file and decide whether Stage 3 critic training is justified
