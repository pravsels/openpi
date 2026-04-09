# BEHAVIOR Challenge Solution Changes

This note summarizes the main changes in the `behavior-1k-solution` repo relative to baseline `openpi` / Pi0.5-style training, and classifies each change by when it applies.

Classification used here:

- `Training-stage`: requires changing training, the model definition, the data pipeline, or checkpoint contents. It may also affect inference indirectly because the trained model is different.
- `Inference-time only`: can be added on top of an already trained checkpoint without retraining.
- `Both`: has a training component and a separate inference-time component, so it does not fit cleanly into only one bucket.

## Changes By Stage

### Training-stage

- Task embeddings replace text prompts.
  Where: `src/b1k/models/pi_behavior.py`, `src/b1k/transforms.py`
  Notes: Replaces language-token conditioning with learned task ID embeddings, so model inputs and checkpoints change.

- Stage / "System 2" prediction auxiliary head.
  Where: `src/b1k/models/pi_behavior.py`, `src/b1k/transforms.py`
  Notes: Adds a stage prediction head trained from VLM features with an auxiliary loss.

- Stage-conditioned task fusion tokens.
  Where: `src/b1k/models/pi_behavior.py`
  Notes: Feeds stage information back into the prefix through learned embeddings and gating layers.

- Learnable mixed-layer KV cache transform.
  Where: `src/b1k/models/pi_behavior.py`
  Notes: Lets each action-expert layer attend to a learned mixture of VLM layers instead of fixed layer-to-layer pairing.

- Custom hierarchical attention layout.
  Where: `src/b1k/models/pi_behavior.py`
  Notes: Changes prefix token structure and attention relationships relative to stock Pi0.5.

- Delta action space.
  Where: `src/b1k/training/config.py`, `src/b1k/transforms.py`
  Notes: Trains on delta actions rather than absolute actions.

- Per-timestamp action normalization.
  Where: `src/b1k/shared/normalize.py`, `src/b1k/training/data_loader.py`
  Notes: Changes preprocessing and saved statistics used by training and serving.

- Correlated flow-matching noise.
  Where: `src/b1k/models/pi_behavior.py`, `src/b1k/shared/normalize.py`
  Notes: Replaces iid Gaussian flow noise with covariance-shaped noise loaded from norm stats.

- Multi-sample flow matching.
  Where: `src/b1k/models/pi_behavior.py`, `src/b1k/training/config.py`
  Notes: Reuses one prefix pass for several `(t, epsilon)` samples per example and averages their losses.

- FAST auxiliary loss / FAST token branch.
  Where: `src/b1k/models/pi_behavior.py`, `src/b1k/transforms.py`, `src/b1k/training/config.py`
  Notes: Adds an auxiliary action-token prediction objective and supporting data pipeline.

- Knowledge insulation toggle.
  Where: `src/b1k/models/pi_behavior.py`, `src/b1k/models/pi_behavior_config.py`
  Notes: Controls whether action-loss gradients can flow back into the VLM prefix cache.

- Multi-task training on all tasks.
  Where: training config / dataset setup
  Notes: This is part of the training recipe, not something added only at inference.

- Task-group-specific checkpoint fine-tuning.
  Where: training workflow, checkpoint mapping
  Notes: They later split tasks across multiple specialized checkpoints.

### Inference-time only

- Stage voting logic during rollout.
  Where: `src/b1k/shared/eval_b1k_wrapper.py`
  Notes: Uses recent predicted stages to advance, skip, or roll back stage state more smoothly.

- Rolling inpainting using saved future actions.
  Where: `src/b1k/shared/eval_b1k_wrapper.py`, `src/b1k/policies/pi_behavior_policy.py`
  Notes: Predict 30 actions, execute 26, keep 4 for the next call as initial conditions.

- Time-thresholded soft inpainting.
  Where: `src/b1k/models/pi_behavior.py`, `src/b1k/shared/eval_b1k_wrapper.py`
  Notes: Only enforces inpainting constraints during early denoising steps.

- Action compression via cubic spline interpolation.
  Where: `src/b1k/shared/eval_b1k_wrapper.py`
  Notes: Compresses executed actions, for example 26 predicted steps to 20 executed steps.

- Disable compression when grippers are moving a lot.
  Where: `src/b1k/shared/correction_rules.py`, `src/b1k/shared/eval_b1k_wrapper.py`
  Notes: Heuristic guard to avoid harming grasp events.

- General gripper correction rule.
  Where: `src/b1k/shared/correction_rules.py`
  Notes: Opens gripper when rollout state suggests an accidental failed grasp.

- Task-specific correction rules.
  Where: `src/b1k/shared/correction_rules.py`
  Notes: Hardcoded fixes for specific failure modes, such as the radio task.

- Task-to-checkpoint switching.
  Where: `src/b1k/policies/checkpoint_switcher.py`, wrapper logic
  Notes: Uses different fine-tuned checkpoints for different tasks at evaluation time.

### Both

- Correlation-aware soft inpainting.
  Where: `src/b1k/models/pi_behavior.py`
  Notes: The inference rule uses a correction matrix derived from the action covariance object computed and loaded through the training/statistics pipeline.

## Practical Grouping

If the goal is to borrow ideas into `openpi` with minimal disruption, the changes naturally split into three groups.

### Easiest to prototype

- Correlated flow-matching noise
- Multi-sample flow matching
- Inference-only correction rules
- Action compression
- Rolling inpainting wrapper logic

These are the most modular ideas and require the least redesign of the current `openpi` prompt / observation pipeline.

### Medium complexity

- Learnable mixed-layer KV transform
- Knowledge insulation toggle
- FAST auxiliary branch

These touch core model code and training, but do not require a full replacement of language conditioning.

### Largest fork from current `openpi`

- Task embeddings instead of text
- Stage prediction + stage-conditioned fusion
- Stage voting as part of the policy state machine
- Custom hierarchical prefix attention

These form a coherent alternate model family rather than a small patch on top of stock Pi0.5.

## Important Caveat

Some items are easiest to describe as "inference-time" or "training-stage", but in practice the line is not perfect:

- A `Training-stage` change often affects inference because it changes the checkpoint being served.
- `Correlation-aware soft inpainting` is marked `Both` because the inference rule depends on a covariance object computed and loaded through the training/statistics pipeline.

So if this note is used as a porting checklist, treat `Inference-time only` items as things that can be tested without retraining, and `Training-stage` items as things that require new experiments/checkpoints.
