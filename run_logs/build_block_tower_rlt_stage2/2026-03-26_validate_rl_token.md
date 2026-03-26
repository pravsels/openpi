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

## Job
- job_id: pending
- submitted: pending
- start_human: pending
- end: pending
- end_human: pending
- runtime: pending
- node: pending

## Status
- 2026-03-26 — created pre-submit Stage 2 validation log and recorded expected probe behavior before HPC launch.

## Results
- runtime: pending
- output_dir: pending
- num_train_samples: pending
- num_val_samples: pending
- probe_suite_status: pending
- action_probe.val_mse_to_vla: pending
- action_probe.val_l2_to_vla: pending
- action_probe.val_mse_to_ground_truth: pending
- action_probe.val_l2_to_ground_truth: pending
- linear_probe.val_mse: pending
- linear_probe.random_baseline_val_mse: pending
- subtask_classifier.enabled: pending
- subtask_classifier.val_accuracy: pending
- verdict: pending

## W&B
- local: n/a
- synced: n/a
- notes: This script writes JSON/NPZ artifacts rather than using the training W&B flow by default.

## Next
- submit `slurm/validate_build_block_tower_rlt_stage2_slurm.sh` on Isambard
- after completion, copy the metrics for every probe variant into this file and decide whether Stage 3 critic training is justified
