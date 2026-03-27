# RLT Episode Validation Split Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add deterministic episode-level train/validation splitting for RLT-style training, persist the split in assets, and log held-out validation loss during training.

**Architecture:** Extend the training/data config with an optional dataset split spec that defines an episode-level train/val partition. The data loader will filter samples to the requested split using persisted episode IDs, and the trainer will create a second loader for validation and periodically log held-out metrics without affecting the training path when no split is configured.

**Tech Stack:** Python, dataclasses, PyTorch DataLoader, JAX training loop, pytest

---

### Task 1: Define split schema and pure helper behavior

**Files:**
- Modify: `src/openpi/training/config.py`
- Modify: `src/openpi/training/data_loader.py`
- Test: `src/openpi/training/data_loader_test.py`

**Step 1: Write the failing tests**

- Add tests for deterministic episode partitioning:
  - same episode IDs + same seed => same train/val split
  - 90/10 split over 200 episodes => 180 train / 20 val
  - split persistence round-trip to assets file preserves exact episode IDs
  - sample filtering by split includes only requested episodes

**Step 2: Run tests to verify they fail**

Run: `pytest src/openpi/training/data_loader_test.py -k "episode_split or persisted_split" -v`

Expected: FAIL because split helpers and config do not exist yet.

**Step 3: Write minimal implementation**

- Add a small config dataclass for dataset splitting (enabled flag, ratio, seed, split names).
- Add pure helper functions in `data_loader.py` for:
  - extracting/canonicalizing episode IDs
  - computing deterministic episode splits
  - saving/loading split metadata under assets
  - filtering dataset indices by requested split

**Step 4: Run tests to verify they pass**

Run: `pytest src/openpi/training/data_loader_test.py -k "episode_split or persisted_split" -v`

Expected: PASS.

---

### Task 2: Wire split-aware data loading

**Files:**
- Modify: `src/openpi/training/config.py`
- Modify: `src/openpi/training/data_loader.py`
- Test: `src/openpi/training/data_loader_test.py`

**Step 1: Write the failing test**

- Add a test that exercises loader construction with a configured split and verifies:
  - train loader samples only train-split indices
  - val loader samples only val-split indices
  - behavior is unchanged when split config is absent

**Step 2: Run test to verify it fails**

Run: `pytest src/openpi/training/data_loader_test.py -k "create_data_loader.*split or split_loader" -v`

Expected: FAIL because `create_data_loader` has no split argument and does not consult persisted episode splits.

**Step 3: Write minimal implementation**

- Extend `create_data_loader` and torch loader plumbing with an optional split selector such as `dataset_split="train" | "val"`.
- Reuse persisted episode split metadata from assets, creating it deterministically on first use if missing.
- Keep existing `valid_indices.txt` behavior and intersect it with split-specific indices when both are present.

**Step 4: Run tests to verify they pass**

Run: `pytest src/openpi/training/data_loader_test.py -k "split_loader or filtered_sampler" -v`

Expected: PASS.

---

### Task 3: Add validation pass to trainer

**Files:**
- Modify: `scripts/train.py`
- Possibly modify: `src/openpi/training/config.py`
- Test: `src/openpi/training/data_loader_test.py` or a new focused trainer test if practical

**Step 1: Write the failing test**

- Prefer a small pure helper test around validation scheduling/metric naming if full trainer testing is too heavy.
- If a trainer test is impractical, document that the trainer integration will be verified by targeted execution instead.

**Step 2: Run test to verify it fails**

Run: targeted `pytest` command for the new helper test, if added.

Expected: FAIL because validation logging helper does not exist yet.

**Step 3: Write minimal implementation**

- Add config knobs for validation cadence and batch count only when split validation is enabled.
- In `scripts/train.py`, build:
  - train loader with `dataset_split="train"`
  - val loader with `dataset_split="val"` when split config is enabled
- Add a non-training eval step that computes held-out loss on a bounded number of validation batches and logs metrics like `val_loss`.

**Step 4: Verify the integration**

Run:
- `pytest src/openpi/training/data_loader_test.py -v`
- a lightweight train-script smoke check if feasible on fake data, otherwise rely on import-level verification plus tests

Expected: PASS for tests; trainer imports and validation path compile.

---

### Task 4: Update run-facing configuration/docs

**Files:**
- Modify: `src/openpi/training/config.py`
- Modify: `run_logs/build_block_tower_rlt/2026-03-25_rlt_v1.md` (if documenting the discovered gap is in scope)
- Optionally modify: `slurm/train_build_block_tower_rlt_slurm.sh`

**Step 1: Add RLT split defaults**

- Enable deterministic 90/10 episode split for `pi05_rl_token_build_block_tower`.
- Keep non-RLT configs unchanged unless explicitly opted in.

**Step 2: Document expected assets**

- Persist split metadata in assets so future reruns use the exact same episode partition.

**Step 3: Quick verification**

- Inspect generated config and confirm the split is opt-in and reproducible.

---

## Execution Notes

- Use TDD for helper and loader logic before production edits.
- Avoid changing dataset semantics for configs that do not opt into validation splitting.
- Commits only if the user explicitly requests them.
