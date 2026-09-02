# BusyBox multitask prompt-fix rerun

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Do **not** use git worktrees; branch as `task/busybox_*`. Read `../autohpc/hpc-run-tracking/SKILL.md` before writing run logs. Do **not** use GMAN. Rent Vast 4× H100 SXM boxes and launch with `vast/bootstrap.sh` + `vast/train.sh`.

**Goal:** Retrain the two relative-action BusyBox π0.5 30k variants once, with honest per-frame language, keeping OpenPI on LeRobot `v0.4.3` and installing RoboCandyWrapper from git `main`.

**Architecture:** RCW `main` already remaps sample `task_index` to the sorted `meta.tasks` table (`de4e4eb`). OpenPI always installs RCW from that repo (`[tool.uv.sources] robocandywrapper` git `main`); do not go back to PyPI. Do **not** edit RCW. `main` imports `lerobot.datasets.feature_utils` at factory import time (v2.1 compat), which 0.4.3 does not have. BusyBox is LeRobot v3, so OpenPI aliases those 0.5 module names onto `lerobot.datasets.utils` before any RCW import (`openpi.training.lerobot_rcw_compat`). Do not bump OpenPI to LeRobot 0.5. Then run the two existing TrainConfigs from scratch on Vast.

**Tech Stack:** OpenPI `task/busybox_multitask_promptfix` (`84a93ad`), RCW git `main` lock `597aa9ad21176e7f7dcee4aede5dc1ffc07eacee`, LeRobot git `v0.4.3`, Vast 4× H100 SXM 80GB, image `nvcr.io/nvidia/pytorch:25.03-py3`, `vast/bootstrap.sh` + `vast/train.sh` (sets `REQUIRE_JAX_DEVICES=4`, then execs `gman/train.sh`).

**Progress:** Tasks 1–3 are done on `task/busybox_multitask_promptfix` (pushed). Local prompt check passed (`wrapped_tasks 27`, `mismatches 0`, `prompt_ok`). Skip to Task 4.

---

## Do not

- Use GMAN. Do not `create_node`, resume `pi05-busybox-minmax-8xh100-prav` / `cmd-dcr8t`, or continue `pi05-busybox-rel-promptfix-8xh100-prav`.
- Call `gman/train.sh` directly (`REQUIRE_JAX_DEVICES` defaults to 8). Always `bash vast/train.sh`.
- Rent H100 **PCIe** (discarded on the abs run). Require **SXM** 80GB HBM3.
- Rent a pricey US 4×H100 (~$21/hr) when a cheaper NL/TW SXM offer exists (~$11–14/hr).
- Rerun `pi05_busybox_multitask_abs` (wrong action space).
- Install RCW from PyPI. Source is always git `main`.
- Upgrade OpenPI to LeRobot 0.5.x (Python 3.12 + Transformers v5).
- Reuse old `assets/` from the shuffled-language runs.
- Skip the on-box prompt check and go straight to 30k.
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

## Proven Vast recipe (copy this)

From the completed 30k runs (`run_logs/2026-09-01_train_pi05_busybox_multitask_vast.md`, `..._abs.md`):

| Knob | Value |
|---|---|
| GPUs | **4× H100 SXM 80GB HBM3** (not PCIe, not 8×) |
| Image | `nvcr.io/nvidia/pytorch:25.03-py3` |
| Disk | ≥1500 GB |
| Price | NL ~$11.11/hr beat TW ~$13.54/hr and US ~$21.72/hr |
| Throughput | ~2.0 it/s after compile (~0.50 s/step); 30k ≈ **4h 25–28m** |
| JAX | `device_count=4`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` |
| SSH key | `~/.ssh/id_ed25519` |
| Secrets | `/workspace/secrets/{hf_token,wandb_token,github_token}` (no echo) |
| Clone | `/workspace/openpi` |
| First-step noise | XLA gemm autotune `Results do not match the reference` + rematerialization warnings; training continued |

`vast/train.sh` already defaults `CONFIG_NAME=pi05_busybox_multitask` and `WANDB_PROJECT=busybox_multitask_pi05`. Minmax **must** override both `CONFIG_NAME` and `WANDB_PROJECT`.

---

### Task 1: OpenPI import shim for RCW `main` on LeRobot 0.4.3 — DONE

Do not change RCW. Alias LeRobot 0.5 dataset helper module names onto `lerobot.datasets.utils` before any `robocandywrapper` import (`rewact_tools` also imports RCW).

Landed in `84a93ad`: `src/openpi/training/lerobot_rcw_compat.py`, `lerobot_rcw_compat_test.py`, plus imports in `data_loader.py`, `scripts/compute_valid_indices.py`, `src/openpi/conftest.py`.

---

### Task 2: Lock OpenPI onto RCW git `main` — DONE

Branch `task/busybox_multitask_promptfix` is pushed. Lock records git `main` SHA `597aa9ad21176e7f7dcee4aede5dc1ffc07eacee`.

Vast clones this ref. Override `vast/bootstrap.sh` default (`REPO_REF=task/busybox_multitask`) to `task/busybox_multitask_promptfix`.

---

### Task 3: Prove language matches inner LeRobot tasks — DONE locally; repeat on each Vast box

Import factory through `lerobot_rcw_compat`, not raw `robocandywrapper.factory`. `load_videos=False` still tries to decode mp4s; stub `_query_videos` when videos are not on disk. After Vast bootstrap, videos are present so the stub is unnecessary.

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

### Task 4: Vast 30k × 2 (fresh 4× H100 SXM)

Rent **two** 4× H100 SXM 80GB instances (or one, relative first, if only one cheap SXM offer). Image `nvcr.io/nvidia/pytorch:25.03-py3`. Disk ≥1500 GB. Prefer cheapest verified-SXM offer (NL ~$11/hr was the cheap completed box).

`vast/bootstrap.sh` defaults `REPO_REF=task/busybox_multitask`. Override:

```bash
export REPO_REF=task/busybox_multitask_promptfix
bash vast/bootstrap.sh
```

Secrets must exist at `/workspace/secrets/{github_token,hf_token,wandb_token}` before bootstrap finishes `setup_done`. `WANDB_MODE=online`.

| Instance label | Config | After bootstrap + Task 3 `prompt_ok` |
|---|---|---|
| `pi05-busybox-rel-promptfix` | `pi05_busybox_multitask` | `nohup bash vast/train.sh` (defaults already match) |
| `pi05-busybox-minmax-promptfix` | `pi05_busybox_multitask_minmax` | `CONFIG_NAME=pi05_busybox_multitask_minmax WANDB_PROJECT=busybox_multitask_pi05_minmax nohup bash vast/train.sh` |

No `SMOKE=1`. Distinct `ASSETS_DIR`s (config name already isolates them). Destroy each instance after Hub 29999 / `train_done`. Do not touch leftover CRA boxes.

Train logs: `/workspace/vast_runs/openpi/logs/{bootstrap.log,train_30k.log}`.

---

### Task 5: Run logs

Keep abandoned GMAN logs as history. New Vast files:

- `run_logs/2026-09-02_train_pi05_busybox_multitask_promptfix_vast.md`
- `run_logs/2026-09-02_train_pi05_busybox_multitask_minmax_promptfix_vast.md`

Follow AutoHPC run-tracking. Record: OpenPI branch SHA, RCW `main` SHA from the lockfile, Vast instance id + offer + $/hr + GPU interconnect (must say SXM), W&B id, Hub URL, JAX device count (4), step 0 / last loss, `train_done`.

---

## Done when

1. Both 30k jobs print `train_done`.
2. Hub roots are step **29999** (not the shuffled-language checkpoints).
3. W&B runs exist under the two project names above.
4. Run logs list the locked RCW `main` SHA and Vast 4× H100 SXM instance ids.

Do not start eval/RLT in this slice.
