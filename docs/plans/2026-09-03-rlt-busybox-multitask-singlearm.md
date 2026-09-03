# RLT BusyBox Multitask Single-Arm Implementation Plan

> **For Claude:** Implement on `task/rlt_busybox_multitask` (no worktree). Follow TDD. Design: `docs/plans/2026-09-03-rlt-busybox-multitask-singlearm-design.md`.

**Goal:** Add `pi05_rlt_busybox_multitask_singlearm` and a GCloud launcher that trains the RL-token bottleneck on frozen `pravsels/pi05_busybox_multitask`.

**Architecture:** Clone the green-button Stage 1 RLT config and GCloud script. Point data/prompts at `villekuosmanen/busybox_multitask` with `prompt_from_task`, and the weight loader at Hub-root `checkpoints/pi05_busybox_multitask/params`. Leave bimanual `pi05_rlt_busybox_multitask` unchanged.

**Tech Stack:** OpenPI `TrainConfig` / `Pi0RLConfig`, pytest, GCloud Docker launcher.

---

### Task 1: Config test + TrainConfig

**Files:**
- Modify: `src/openpi/training/config_test.py` (after `test_busybox_push_green_button_rlt_gcloud_script_references_config`)
- Modify: `src/openpi/training/config.py` (after `pi05_rlt_busybox_push_green_button`)

**Step 1: Write the failing tests** (clone green-button RLT tests; assert `prompt_from_task`, Hub path, project `busybox_multitask_rlt_singlearm`).

**Step 2:** `uv run pytest src/openpi/training/config_test.py::test_busybox_multitask_singlearm_rlt_config src/openpi/training/config_test.py::test_busybox_multitask_singlearm_rlt_gcloud_script_references_config -v`

Expected: FAIL — config not found / script missing.

**Step 3:** Add `TrainConfig(name="pi05_rlt_busybox_multitask_singlearm", ...)` matching green-button knobs with three-cam multitask data.

**Step 4:** Re-run config test. Expected: config test PASS; script test still FAIL.

**Step 5:** Commit when both tasks land.

### Task 2: GCloud launcher

**Files:**
- Create: `slurm/train_busybox_multitask_singlearm_rlt_gcloud.sh`

**Step 1:** Script test already written in Task 1.

**Step 2:** Clone `slurm/train_busybox_push_green_button_rlt_gcloud.sh`; swap config, exp, Hub repo, checkpoint dir.

**Step 3:** Re-run both tests. Expected: PASS.

**Step 4:** Confirm `get_config("pi05_rlt_busybox_multitask")` is still the bimanual 12D config.
