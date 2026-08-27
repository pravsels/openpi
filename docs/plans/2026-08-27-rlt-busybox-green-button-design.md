# RLT Stage 1 — BusyBox Push Green Button

**Goal:** Train the RL-token encoder/decoder on top of the frozen [`pravsels/pi05_busybox_push_green_button`](https://huggingface.co/pravsels/pi05_busybox_push_green_button) VLA so `hw_control.pi0_rlt` can extract tokens for the demo cache and online RL.

**Where:** Branch `task/rlt_busybox_green_button`. Launch on GCloud (Docker on an A100 80GB VM), not GMAN and not Isambard.

**Out of scope:** Stage 2+ RL; changing existing two-cam / bimanual RLT configs; waiting on a new VLA train; worktrees.

---

## Frozen VLA

Hub repo [`pravsels/pi05_busybox_push_green_button`](https://huggingface.co/pravsels/pi05_busybox_push_green_button) publishes `params/` + `assets/` at the **repo root** (not `step_N/`). Download into `checkpoints/pi05_busybox_push_green_button/` so the loader sees:

```
checkpoints/pi05_busybox_push_green_button/params
```

Reuse the published `assets/` (norm stats + valid indices) so the frozen VLA sees the same normalized inputs it was trained on. Do not recompute unless those files are missing.

Do **not** reuse `pi05_rlt_busybox_press_green_yellow_buttons` or the other bimanual busybox RLT configs. Those are 12D, different cameras, different datasets.

---

## Image-key fix

`Pi0RL` inherits `self.image_keys` from `Pi0`, but `extract_rl_token`, `sample_actions_with_rl_token`, and `compute_loss` still call `_model.preprocess_observation(...)` with the default `IMAGE_KEYS` (`right_wrist_0_rgb`). That drops `base_1_rgb` (`front`).

Fix: route those three calls through `self._preprocess_observation`, which already passes `configured_image_keys(observation, self.image_keys)`. Default two-cam / `right_wrist` RLT configs stay unchanged.

---

## Train config

Add `pi05_rlt_busybox_push_green_button` next to the other Stage-1 RLT entries.

| Knob | Value |
|---|---|
| Model | `Pi0RLConfig(pi05=True, action_horizon=30, image_keys=SO101_THREE_CAM_IMAGE_KEYS, rl_vla_loss_weight=0.0)` |
| Encoder/decoder | 2 layers, 8 heads, dim 2048, mlp 8192 (existing defaults) |
| Data | `LeRobotSO101ThreeCamDataConfig` on `villekuosmanen/busybox_push_green_button` |
| Prompt | `push the green button` |
| Actions | 6D; 5 joints delta, gripper absolute; per-timestep action norm |
| Freeze | `get_rl_freeze_filter()` — only `rl_encoder` / `rl_decoder` |
| Weight loader | `RLTokenCheckpointWeightLoader("checkpoints/pi05_busybox_push_green_button/params")` |
| Steps | 20,000 |
| Batch | 16 (global), `fsdp_devices=1`, `num_workers=8` |
| LR | cosine, warmup 1k, peak 5e-5, `decay_steps=20_000`, decay 5e-5 |
| EMA | 0.999 |
| Save | once at the end (`save_interval=20_000`), `keep_period=None` |
| W&B | online, entity `pravsels`, project `busybox_push_green_button_rlt` |

No episode val split. The 20-episode set is too small for a useful 90/10, and the other SO101 RLT configs omit it.

---

## GCloud launch

Follow `slurm/train_so101_stacking_rings_gcloud.sh`: run on the VM (or via `gcloud compute ssh`), Docker `openpi:latest`, mounts for `src`, `weights`, `assets`, `checkpoints`.

Script: `slurm/train_busybox_push_green_button_rlt_gcloud.sh`.

1. Refuse unless `nvidia-smi` shows at least one GPU with ≥80 GB.
2. `hf download pravsels/pi05_busybox_push_green_button` into `checkpoints/pi05_busybox_push_green_button` if `params/` is missing.
3. Copy Hub `assets/` into the run assets dir if norm-stat files are missing.
4. `uv run scripts/train.py pi05_rlt_busybox_push_green_button --exp-name=... --assets-dir=... --overwrite` (or `--resume`).
5. Export `WANDB_MODE=online` **before** process start (`scripts/train.py` defaults offline).
6. `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`.

One A100 80GB is enough: Stage 1 freezes the VLA and only the small bottleneck trains. Prefer a new `a2-ultragpu-1g` / `a2-highgpu-1g` (or reuse `openpi-so101-80g-2x` if it is already up). Disk must fit the ~75 GB Hub checkpoint plus the RLT run.

---

## Tests (CPU)

- `Pi0RLConfig(image_keys=SO101_THREE_CAM_IMAGE_KEYS).inputs_spec()` uses those three keys.
- `Pi0RL._preprocess_observation` keeps `base_1_rgb` and does not require `right_wrist_0_rgb`.
- Existing dummy `debug_pi0_rl` / default-key RLT path still works.

---

## Failure modes

| Risk | Handling |
|---|---|
| Default `IMAGE_KEYS` drops `front` | Use `self._preprocess_observation` |
| Weight loader pointed at `step_N/` | Hub root is `params/`; script checks that path |
| Recomputed norm stats drift from the VLA | Copy Hub `assets/` |
| `WANDB_MODE` offline from `train.py` | Export `online` in the Docker env |
| Disk fills on the ~75 GB download | Require ≥200 GB free before download |

---

## Files

| Path | Role |
|---|---|
| `src/openpi/models/pi0_rl.py` | Preprocess via `self._preprocess_observation` |
| `src/openpi/training/config.py` | New `TrainConfig` |
| `src/openpi/models/pi0_rl_test.py` (or next to existing model tests) | CPU gates |
| `slurm/train_busybox_push_green_button_rlt_gcloud.sh` | GCloud Docker launcher |
| `run_logs/2026-08-27_train_rlt_busybox_push_green_button_gcloud.md` | After launch |

---

## Implementation order

1. CPU test that three-cam preprocess fails on current `Pi0RL`.
2. Switch the three preprocess calls to `self._preprocess_observation`.
3. Add `pi05_rlt_busybox_push_green_button`.
4. Add the GCloud launcher.
5. Download the Hub VLA on the VM and start the 20k run.
