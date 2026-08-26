# π0 / π0.5 BusyBox Green-Button Comparison Runs

**Goal:** Full-component finetune π0 and π0.5 on `villekuosmanen/busybox_push_green_button` so they sit next to the ACT / SmolVLA / MolmoAct2 30k green-button jobs.

**Where:** [`pravsels/openpi`](https://github.com/pravsels/openpi), new branch (same idea as `task/train_lerobot_policies_green_button`). Launch on GMAN following the **CRA** working example, not Vast and not the unused MolmoAct2 Docker GMAN path.

**Out of scope:** post-train Armnet eval; Vast / RTX 5090; `openpi_amd64` / `openpi_arm64` Hub images; changing existing two-cam ArmNetBench SO101 configs.

---

## Comparison to match

| Knob | Value |
|---|---|
| Dataset | [`villekuosmanen/busybox_push_green_button`](https://huggingface.co/datasets/villekuosmanen/busybox_push_green_button) (LeRobot v3, 20 eps, 2471 frames, 20 fps, task `push the green button`) |
| Embodiment | single-arm SO101, 6D joint-space |
| Cameras | `top`, `wrist`, `front` at 720×1280 |
| Steps | 30,000 |
| Global batch | 32 |
| Action horizon | 30 (ArmNetBench / ACT / MolmoAct2, not SmolVLA 50) |
| Finetune | full component (OpenPI default: `freeze_filter` freezes nothing; SigLIP trains) |
| W&B | online, entity `pravsels` |
| Hub | `pravsels/pi0_busybox_push_green_button`, `pravsels/pi05_busybox_push_green_button` |

Existing OpenPI BusyBox configs (`pi05_busybox_press_green_yellow_buttons`, bimanual buttons, …) are **different datasets** (bimanual, 10k steps). Do not reuse them.

---

## Cameras and model slots

π0 / π0.5 always pack three images. Canonical names from ALOHA / DROID / LIBERO:

| Slot | Pretrain meaning | BusyBox camera |
|---|---|---|
| `base_0_rgb` | third-person / overhead (`cam_high`) | `top` |
| `left_wrist_0_rgb` | left or only wrist | `wrist` |
| third slot | right wrist in pretrain | `front` as **`base_1_rgb`** |

`front` is a second **scene** camera, not a wrist. There is no separate wrist encoder — all views share SigLIP. The key name only gates augmentation (`"wrist" not in key` → crop + ±5° rotate). Naming the third view `base_1_rgb` gives `front` the same transforms as `top`.

Do **not** change `SO101Inputs` (front + wrist, right slot masked). Add a three-cam transform + data config so ArmNetBench stays two-cam.

All three masks stay True.

---

## Model-side image keys

`IMAGE_KEYS` and `Pi0Config.inputs_spec` / `Pi05Config.inputs_spec` hard-code `right_wrist_0_rgb`. π0.5 already preprocesses `list(observation.images.keys())`. π0 `compute_loss` / sample paths still call `preprocess_observation` with the default keys — that would drop `base_1_rgb`.

Changes (only what these runs need; default remains the canonical trio):

1. Optional `image_keys` on `Pi0Config` (and the π0.5 `Pi0Config(pi05=True)` path used by ArmNetBench), defaulting to `IMAGE_KEYS`.
2. `inputs_spec` builds the observation dict from that tuple.
3. π0 preprocess calls pass `image_keys=list(observation.images.keys())` like π0.5.

JAX pytrees include dict keys, so init `fake_obs` and the training batch must use the same three names.

---

## Train configs

Add `pi0_busybox_push_green_button` and `pi05_busybox_push_green_button`, cloned from ArmNetBench single-arm:

- `LeRobotSO101ThreeCamDataConfig` (name TBD) on repo `villekuosmanen/busybox_push_green_button`
- `default_prompt="push the green button"`
- delta actions: 5 joints delta, gripper absolute; per-timestep action norm on
- init: `weights/pi0_base/params` / `weights/pi05_base/params`
- `Pi0Config(action_horizon=30, image_keys=(base_0, left_wrist, base_1))` and `Pi0Config(pi05=True, …)` respectively
- `num_train_steps=30_000`, `save_interval=5_000`, `keep_period=5_000`
- cosine: warmup 1k, peak 2.5e-5, **decay_steps=30_000**, decay 2.5e-6
- `batch_size=32` (global), `fsdp_devices=1` → 8-way data parallel, 4 samples/GPU on an 8-GPU node
- `ema_decay=0.999`, `wandb_enabled=True`

`scripts/train.py` currently `setdefault("WANDB_MODE", "offline")`. GMAN jobs must export `WANDB_MODE=online` **before** import / process start.

---

## GMAN launch (CRA pattern)

GMAN offers 1-GPU or 8-GPU. Use **8× ≥80 GB H100**, not `h100-1`. Fresh node; do not restore a large parked snapshot.

Follow CRA (`experimental/cra` in alpha-robotics), not MolmoAct2 Docker:

1. `create_node` (8-GPU chip) + `hold_node` + mission.
2. Bootstrap command, then train command, then `stop_node`.
3. Secrets only as typed refs: `env: {HF_TOKEN: {secret: "hf-token"}, WANDB_API_KEY: {secret: "wandb-api-key"}, GITHUB_TOKEN: {secret: "github-repo"}}`.
4. **Never** `gman run -e WANDB_API_KEY=wandb-api-key` (injects the name; CRA already hit a 7-character key).
5. `detach: true`, `hold_on_failure_minutes: 30`.
6. No Docker. Host `uv` on `$HOME` (survives `stop_node`; `/scratch` does not).

**Bootstrap** (`gman/bootstrap.sh` in this repo):

- ffmpeg, git, curl, ca-certificates (LeRobot video decode).
- install uv; `GIT_ASKPASS` / `url.insteadOf` from `GITHUB_TOKEN`; no `set -x`.
- clone `https://github.com/pravsels/openpi.git` at the task branch into `$HOME/openpi`.
- `GIT_LFS_SKIP_SMUDGE=1 uv sync --group dev && uv pip install -e .`
- stage `gs://openpi-assets/checkpoints/pi0_base` and `pi05_base` into `$HOME/openpi/weights/.../params`.
- refuse unless `nvidia-smi` shows **8 GPUs, each ≥ 80 GB**.
- print `setup_done`.

**Train:** compute per-timestep norm stats into the run assets dir (skip if present), then `uv run scripts/train.py <config> --exp-name=... --assets-dir=... --overwrite` (or `--resume`). JAX uses all 8 devices. π0 then π0.5 on the same node (sequential) unless two 8-GPU nodes are held.

Laptop driver: copy CRA’s `GmanCli` (`api post /nodes/<node>/commands`, poll logs, stop). Scripts live in **this** repo (`gman/`), not alpha-robotics.

Hub images (`praveensels/openpi_amd64`, `openpi_arm64`) are ~7 months stale and are CPU-arch tags, not GPU tags. RTX 5090 is amd64 Blackwell 32 GB — not this plan.

---

## 10-step publish gate

Each policy must pass this **before** 30k. Train does not count as green unless W&B logged and Hub has weights (CRA 10-step check).

| Smoke knob | Value |
|---|---|
| `num_train_steps` | 10 |
| `save_interval` | 5 |
| `keep_period` | 5 |
| `log_interval` | **1** (default 100 would only log step 0) |
| W&B | online, `pravsels`, project = config `project_name` |

OpenPI saves when `step % save_interval == 0 and step > start_step` **or** `step == num_train_steps - 1`. Loop is `range(0, 10)` → checkpoints **5** and **9**, not LeRobot `000010`.

After train exit 0, upload `params/` + `assets/` (ignore `train_state/`) with `HfApi.create_repo(..., exist_ok=True)` + `upload_folder` to `step_5/` and `step_9/`. Same layout as ArmNetBench publish scripts; namespace **`pravsels`**.

**Pass only if:**

1. Train exit 0 and the log shows 8 JAX devices.
2. W&B run in the matching project has a loss curve after step 0 (not a dead offline run).
3. Hub repo exists; `step_5` and `step_9` contain `params/`.
4. Env used typed secret refs.

Fail the command (keep the node 30 min) if any check misses. Delete smoke Hub revisions and local ckpts before the 30k job.

---

## Tests (CPU)

- Three-cam transform: `top`/`wrist`/`front` → `base_0_rgb`/`left_wrist_0_rgb`/`base_1_rgb`, all masks True.
- Two-cam `SO101Inputs` unchanged (right slot still zeros + masked).
- `Pi0Config(image_keys=(..., "base_1_rgb")).inputs_spec()` pytree uses those keys.
- Publish helper skips missing steps and exits non-zero if nothing uploaded.
- GMAN command payload uses `{secret: ...}` not literal env names.

---

## Failure modes

| Risk | Handling |
|---|---|
| 1-GPU `h100-1` instead of 8-GPU | Bootstrap GPU count/VRAM check |
| Restored snapshot | Fresh node (CRA: restore was slow and then still broke) |
| Secret *name* as W&B key | Typed refs only; smoke asserts a real run |
| `gs://openpi-assets` blocked | Bootstrap must finish with both `params` trees on disk |
| `WANDB_MODE` offline from `train.py` | Export `online` in the train process environment |
| Batch 32 not divisible by 8 | Config is 32; launcher may assert `batch_size % device_count == 0` |
| Third camera named `right_wrist_0_rgb` | Would get wrist aug; we use `base_1_rgb` |

---

## Files (expected)

| Path | Role |
|---|---|
| `src/openpi/policies/so101_three_cam_policy.py` (or flag on `so101_policy.py`) | Transforms |
| `src/openpi/training/config.py` | Data factory + two `TrainConfig`s |
| `src/openpi/models/pi0.py`, `pi0_config.py` | Observation keys / preprocess |
| `gman/bootstrap.sh`, `gman/train.sh`, `gman/publish_smoke.py`, `gman/launch.py` | CRA-style GMAN |
| Tests next to the new modules | CPU gates |
| `run_logs/2026-08-26_train_*_busybox_push_green_button_gman.md` | After launch |

---

## Implementation order

1. Three-cam transform + unit test.
2. `image_keys` on π0 spec / preprocess + unit test.
3. Train configs (30k defaults).
4. Publish helper + GMAN payload tests.
5. Bootstrap / train / 10-step gate scripts.
6. 10-step GMAN smoke (π0, then π0.5).
7. 30k production on a clean checkpoint dir.
