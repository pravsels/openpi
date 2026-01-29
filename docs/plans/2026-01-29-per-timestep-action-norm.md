# Per-Timestep Action Normalization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a parallel per-timestep action normalization flow (actions only) with a separate stats file, auto-enabled for delta-action configs while preserving existing global normalization.

**Architecture:** Keep `norm_stats.json` unchanged for global stats; compute and store per-timestep action stats in `norm_stats_actions_per_timestep.json`. At runtime, merge per-timestep action stats into the normalization dict only when enabled; otherwise use existing global stats.

**Tech Stack:** Python, NumPy, dataclasses, JAX (existing), JSON I/O

---

### Task 1: Add per-timestep stats I/O + compute script

**Files:**
- Create: `scripts/compute_norm_stats_per_timestep.py`
- Modify: `src/openpi/shared/normalize.py`

**Step 1: Implement new stats file helpers**

- Add helpers in `normalize.py` to save/load `norm_stats_actions_per_timestep.json`.
- Use the existing `NormStats` schema for the actions-only stats file.

**Step 2: Implement compute script**

- Compute global stats for `state` and `actions` using existing `RunningStats` (same as current script).
- Compute per-timestep stats for `actions` by maintaining one `RunningStats` per timestep index and stacking into a single `NormStats` (mean/std/q01/q99 arrays shaped `[H, D]`).
- Write global stats to `norm_stats.json` and per-timestep action stats to `norm_stats_actions_per_timestep.json` in the same assets directory.

**Step 3: Quick verification**

- Run a dry invocation (no dataset) is not possible, so verify script imports and runs to argument parsing without errors:
  - `python scripts/compute_norm_stats_per_timestep.py --help`

---

### Task 2: Wire config + data loader + policy for per-timestep actions

**Files:**
- Modify: `src/openpi/training/config.py`
- Modify: `src/openpi/training/data_loader.py`
- Modify: `src/openpi/policies/policy_config.py`
- Modify: `src/openpi/training/checkpoints.py`

**Step 1: Add config fields**

- Extend `DataConfig` with:
  - `use_per_timestep_action_norm: bool | None = None`
  - `per_timestep_action_norm_stats: NormStats | None = None`
- In `DataConfigFactory.create_base_config`, load the per-timestep actions stats file into `per_timestep_action_norm_stats` (no change to existing `norm_stats`).

**Step 2: Auto-enable for delta action configs**

- In `LeRobotBinPackDataConfig.create`, if `use_delta_actions=True` and `use_per_timestep_action_norm is None`, set it to `True`.
- In `LeRobotAlohaDataConfig.create`, if `use_delta_joint_actions=True` and `use_per_timestep_action_norm is None`, set it to `True`.
- In `LeRobotLiberoDataConfig.create`, if `extra_delta_transform=True` and `use_per_timestep_action_norm is None`, set it to `True`.

**Step 3: Merge stats for normalization**

- Add a small helper (in `normalize.py` or `data_loader.py`) that merges `actions` from `per_timestep_action_norm_stats` into the `norm_stats` dict when enabled.
- Use that merged dict in `data_loader.transform_dataset` and `transform_iterable_dataset`.
- In `policy_config.create_trained_policy`, use merged stats for `Normalize`/`Unnormalize` so inference uses per-timestep actions when enabled.

**Step 4: Save per-timestep stats in checkpoints**

- In `checkpoints.save_state`, if `per_timestep_action_norm_stats` exists and `asset_id` is set, save it to the assets directory using the new helper.

---

### Task 3: Documentation update

**Files:**
- Modify: `docs/norm_stats.md`

**Step 1: Document new file and behavior**

- Describe `norm_stats_actions_per_timestep.json` as actions-only, per-timestep stats.
- Note auto-enabling when delta actions are used, and how to override via config.
- Mention that state normalization remains global.

**Step 2: Quick verification**

- Ensure docs build without errors (no formal command required).

---

## Execution Notes

- **TDD waived** per user permission; do not write failing tests first.
- **Commits** only if user explicitly requests them (system constraint).
