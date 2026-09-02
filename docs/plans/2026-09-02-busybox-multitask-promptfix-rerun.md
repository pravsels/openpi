# BusyBox multitask prompt-fix rerun

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Do **not** use git worktrees; branch as `task/busybox_*`. Read `../autohpc/hpc-run-tracking/SKILL.md` before writing run logs. Read GMAN `gman://docs/llms.txt` (or https://givemeanode.com/llms.txt) before `create_node`.

**Goal:** Retrain the two relative-action BusyBox π0.5 30k variants once, with honest per-frame language, keeping OpenPI on LeRobot `v0.4.3` and installing RoboCandyWrapper from git `main`.

**Architecture:** RCW `main` already remaps sample `task_index` to the sorted `meta.tasks` table (`de4e4eb`). OpenPI always installs RCW from that repo (`[tool.uv.sources] robocandywrapper` git `main`); do not go back to PyPI. Do **not** edit RCW. `main` imports `lerobot.datasets.feature_utils` at factory import time (v2.1 compat), which 0.4.3 does not have. BusyBox is LeRobot v3, so OpenPI aliases those 0.5 module names onto `lerobot.datasets.utils` before any RCW import (`openpi.training.lerobot_rcw_compat`). Do not bump OpenPI to LeRobot 0.5. Then run the two existing TrainConfigs from scratch.

**Tech Stack:** OpenPI branch from `task/busybox_multitask_minmax`, RCW git `main`, LeRobot git `v0.4.3`, GMAN 8×H100, `gman/bootstrap.sh` + `gman/train.sh`.

---

## Do not

- Rerun `pi05_busybox_multitask_abs` (wrong action space).
- Resume the parked GMAN node `pi05-busybox-minmax-8xh100-prav` or bootstrap `cmd-dcr8t`.
- Install RCW from PyPI. Source is always git `main`.
- Upgrade OpenPI to LeRobot 0.5.x (Python 3.12 + Transformers v5).
- Reuse old `assets/` from the shuffled-language runs.
- Skip the prompt check and go straight to 30k.
- Use git worktrees.
- Edit or push RoboCandyWrapper. The 0.4.3 import break is handled in OpenPI.

The published Hub repo `pravsels/pi05_busybox_multitask` (and minmax if created) learned shuffled prompts. Overwriting repo **root** on the new 30k is intended.

---

## Variants (already in `src/openpi/training/config.py`)

Both: dataset `villekuosmanen/busybox_multitask` (v3, 66 eps, 12141 frames, 27 tasks), three-cam, `prompt_from_task=True`, 5 joints delta + abs gripper, 30k, batch 32, cosine 2.5e-5→2.5e-6, `pi05_base`, save every 5k keep latest only.

| Config | Norm | W&B | Hub |
|---|---|---|---|
| `pi05_busybox_multitask` | per-timestep **1%/99%** | `busybox_multitask_pi05` | `pravsels/pi05_busybox_multitask` |
| `pi05_busybox_multitask_minmax` | per-timestep **min/max** stuffed into q01/q99 | `busybox_multitask_pi05_minmax` | `pravsels/pi05_busybox_multitask_minmax` |

Relative Vast 30k was 4 GPU (`412112c`). This rerun uses GMAN `gman/train.sh`, which **requires 8 JAX devices**. Same global batch; only throughput changes.

---

### Task 1: OpenPI import shim for RCW `main` on LeRobot 0.4.3

Do not change RCW. Alias LeRobot 0.5 dataset helper module names onto `lerobot.datasets.utils` before any `robocandywrapper` import (`rewact_tools` also imports RCW).

**Files:**
- Add: `src/openpi/training/lerobot_rcw_compat.py` (aliases + re-export factory/plugins)
- Add: `src/openpi/training/lerobot_rcw_compat_test.py`
- Modify: `src/openpi/training/data_loader.py`, `scripts/compute_valid_indices.py`, `src/openpi/conftest.py`

Import RCW symbols from `openpi.training.lerobot_rcw_compat` so isort cannot hoist `robocandywrapper` above the aliases.

Run: `GIT_LFS_SKIP_SMUDGE=1 uv run pytest src/openpi/training/lerobot_rcw_compat_test.py -v`  
Expected: PASS. `from robocandywrapper.factory import …` without the shim still fails on 0.4.3; that is OK.

---

### Task 2: Lock OpenPI onto RCW git `main`

**Files:**
- Modify: `pyproject.toml`, `uv.lock` (already has git `main`; re-lock after Task 1)
- Branch: `task/busybox_multitask_promptfix` from `origin/task/busybox_multitask_minmax`

Keep:

```toml
"robocandywrapper",
```

```toml
[tool.uv.sources]
openpi-client = { workspace = true }
lerobot = { git = "https://github.com/huggingface/lerobot", rev = "v0.4.3" }
robocandywrapper = { git = "https://github.com/villekuosmanen/RoboCandyWrapper.git", rev = "main" }
```

`uv.lock` will record the resolved `main` SHA. That is the install; do not add a PyPI specifier. Locked SHA during this slice: `597aa9ad21176e7f7dcee4aede5dc1ffc07eacee` (includes remap `de4e4eb`).

Run: `GIT_LFS_SKIP_SMUDGE=1 uv lock --upgrade-package robocandywrapper && GIT_LFS_SKIP_SMUDGE=1 uv sync --group dev`

Expected: lock `source = { git = "...RoboCandyWrapper.git?rev=main#<sha>" }`. `from openpi.training.lerobot_rcw_compat import make_dataset_without_config` succeeds on LeRobot 0.4.3. Wrapper `__getitem__` remaps `task_index` and sets `item["task"]` (no separate `unified_task_index` key).

Push the OpenPI branch. GMAN clones this ref.

---

### Task 3: Prove language matches inner LeRobot tasks

On the OpenPI venv, after Task 2:

Import factory through `lerobot_rcw_compat`, not raw `robocandywrapper.factory`. `load_videos=False` still tries to decode mp4s; stub `_query_videos` when videos are not on disk. On GMAN after bootstrap, videos are present so the stub is unnecessary.

```bash
GIT_LFS_SKIP_SMUDGE=1 uv run python - <<'PY'
from openpi.training.lerobot_rcw_compat import make_dataset_without_config
from openpi.training.data_loader import _coerce_task_mapping
from openpi.transforms import PromptFromLeRobotTask

ds = make_dataset_without_config("villekuosmanen/busybox_multitask", load_videos=False)
for inner_ds in ds._datasets:
    inner_ds._query_videos = lambda *a, **k: {}
inner = _coerce_task_mapping(ds._datasets[0].meta.tasks)
wrapped = _coerce_task_mapping(ds.meta.tasks)
xf = PromptFromLeRobotTask(wrapped)
mismatches = []
for i in range(0, len(ds), max(1, len(ds)//80)):
    item = ds[i]
    prompt = xf(dict(item))["prompt"]
    inner_idx = int(item["task_index"].item() if hasattr(item["task_index"], "item") else item["task_index"])
    name = item.get("task", wrapped[inner_idx])
    if isinstance(name, bytes):
        name = name.decode()
    if prompt != name:
        mismatches.append((i, inner_idx, prompt, name))
print("wrapped_tasks", len(wrapped), "inner_tasks", len(inner))
print("mismatches", len(mismatches))
assert not mismatches, mismatches[:5]
assert wrapped, wrapped
print("prompt_ok")
PY
```

Expected: `prompt_ok`. If this fails, do not launch GPUs.

Local result (2026-09-02): `wrapped_tasks 27`, `mismatches 0`, index 0 is `Move the left slider to position 1` (alphabetical first). A `"push the green button"` frame has wrapped `task_index` 23 and that prompt (inner LeRobot index 0 is the green button; without remap, OpenPI would have trained on the slider sentence). `_coerce_task_mapping` returns `{}` for the inner pandas DataFrame (`to_pandas` is HF-only); training uses wrapped `meta.tasks`, which is a dict.

---

### Task 4: GMAN 30k × 2 (fresh nodes)

`gman/bootstrap.sh` defaults `REPO_REF=task/train_pi_policies_green_button`. Override to `task/busybox_multitask_promptfix`.

Secrets: `github-repo`, `hf-token`, `wandb-api-key`. `WANDB_MODE=online`.

Two **new** 8×H100 80GB nodes (`cuda-12.9`, scratch ≥250 GiB). Name them so they are obviously yours. `list_nodes` first. `stop_node` when done.

| Node (suggested) | Config | After bootstrap |
|---|---|---|
| `pi05-busybox-rel-promptfix-8xh100-prav` | `pi05_busybox_multitask` | `CONFIG_NAME=pi05_busybox_multitask bash gman/train.sh` |
| `pi05-busybox-minmax-promptfix-8xh100-prav` | `pi05_busybox_multitask_minmax` | `CONFIG_NAME=pi05_busybox_multitask_minmax bash gman/train.sh` |

No `SMOKE=1`. Repeat Task 3 on the node after `uv sync` before `train.sh`. Distinct `ASSETS_DIR`s (config name already isolates them). `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`.

Parallel if quota allows; otherwise relative first, then minmax.

---

### Task 5: Run logs

New files (keep the cancelled minmax log as history):

- `run_logs/2026-09-02_train_pi05_busybox_multitask_promptfix_gman.md`
- `run_logs/2026-09-02_train_pi05_busybox_multitask_minmax_promptfix_gman.md`

Follow AutoHPC run-tracking. Record: OpenPI branch SHA, RCW `main` SHA from the lockfile, W&B id, Hub URL, JAX device count, step 0 / last loss, `train_done`.

---

## Done when

1. Both 30k jobs print `train_done`.
2. Hub roots are step **29999** (not the shuffled-language checkpoints).
3. W&B runs exist under the two project names above.
4. Run logs list the locked RCW `main` SHA.

Do not start eval/RLT in this slice.
