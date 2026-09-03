# RLT Stage 1 — BusyBox Multitask Single-Arm

**Goal:** Train the RL-token encoder/decoder on top of the frozen [`pravsels/pi05_busybox_multitask`](https://huggingface.co/pravsels/pi05_busybox_multitask) VLA so `hw_control.pi0_rlt` can extract tokens for the demo cache and online RL.

**Where:** Launch on GCloud (Docker on an A100 80GB VM), not GMAN, Vast, or Isambard. Same venue as the green-button RLT.

**Out of scope:** Stage 2+ RL; minmax VLA (`pi05_busybox_multitask_minmax`); rewriting the old bimanual `pi05_rlt_busybox_multitask`; changing other two-cam / bimanual RLT configs; new `Pi0RL` preprocess work (that landed with green button).

---

## Frozen VLA

Hub repo [`pravsels/pi05_busybox_multitask`](https://huggingface.co/pravsels/pi05_busybox_multitask) is the 2026-09-02 prompt-fix relative 30k (Vast 4× H100, W&B `4ym0qegc`, code `84a93ad`, RCW `597aa9ad`). It **replaces** the 2026-09-01 shuffled-language Hub root. `main` publishes `params/` + `assets/` at the **repo root** (step 29999; not `step_N/`). Download into `checkpoints/pi05_busybox_multitask/` so the loader sees:

```
checkpoints/pi05_busybox_multitask/params
```

Reuse the published `assets/` (norm stats + valid indices) so the frozen VLA sees the same per-timestep 1%/99% relative-action normalization it was trained on. Do not recompute.

Do **not** train the existing `pi05_rlt_busybox_multitask` config. That is still the old bimanual 12D / 10k Isambard recipe. Leave it untouched. The new config is `pi05_rlt_busybox_multitask_singlearm`.

---

## Train config

Add `pi05_rlt_busybox_multitask_singlearm` next to `pi05_rlt_busybox_push_green_button`.

| Knob | Value |
|---|---|
| Model | `Pi0RLConfig(pi05=True, action_horizon=30, image_keys=SO101_THREE_CAM_IMAGE_KEYS, rl_vla_loss_weight=0.0)` |
| Encoder/decoder | 2 layers, 8 heads, dim 2048, mlp 8192 (existing defaults) |
| Data | `LeRobotSO101ThreeCamDataConfig` on `villekuosmanen/busybox_multitask` |
| Prompt | `prompt_from_task=True` (RCW remaps `task_index`; 27 instructions; no single default) |
| Actions | 6D; 5 joints delta, gripper absolute; per-timestep 1%/99% |
| Freeze | `get_rl_freeze_filter()` — only `rl_encoder` / `rl_decoder` |
| Weight loader | `RLTokenCheckpointWeightLoader("checkpoints/pi05_busybox_multitask/params")` |
| Steps | 20,000 |
| Batch | 16 (global), `fsdp_devices=1`, `num_workers=8` |
| LR | cosine, warmup 1k, peak 5e-5, `decay_steps=20_000`, decay 5e-5 |
| EMA | 0.999 |
| Save | once at the end (`save_interval=20_000`), `keep_period=None` |
| W&B | online, entity `pravsels`, project `busybox_multitask_rlt_singlearm` |

W&B project is distinct from the old bimanual `busybox_multitask_rlt`. No episode val split.

---

## GCloud launch

Clone `slurm/train_busybox_push_green_button_rlt_gcloud.sh` as `slurm/train_busybox_multitask_singlearm_rlt_gcloud.sh`.

1. Refuse unless `nvidia-smi` shows at least one GPU with ≥80 GB. Do not use `a2-highgpu-1g` (40 GB).
2. Require ≥200 GB free under the repo dir (Hub VLA is ~75 GB).
3. `snapshot_download pravsels/pi05_busybox_multitask` into `checkpoints/pi05_busybox_multitask` if `params/_METADATA` + `manifest.ocdbt` are missing. Refuse unless the Hub README contains W&B `4ym0qegc` (prompt-fix 30k, not the shuffled Sep 1 root).
4. Copy Hub `assets/` into the run assets dir if norm-stat files are missing. Exit if those files are absent — never recompute.
5. Mount the host checkout (not just `src`). Set `UV_PROJECT_ENVIRONMENT=/workspace/repo/.venv` so `uv sync` persists on the mount (the image `/.venv` is discarded on `--rm`). Then `scripts/check_busybox_multitask_rcw_prompts.py` must print `rcw_sha_ok` and `prompt_ok` (27 remapped tasks; index 0 is the slider sentence). Train reuses that venv and re-checks the SHA. Do not start GPUs otherwise.
6. `uv run scripts/train.py pi05_rlt_busybox_multitask_singlearm --exp-name=busybox_multitask_rlt_singlearm --assets-dir=... --overwrite` (or `--resume` if a step dir already exists).
7. Export `WANDB_MODE=online` **before** process start (`scripts/train.py` defaults offline).
8. `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`.

One A100 80GB (`a2-ultragpu-1g`) is enough: Stage 1 freezes the VLA and only the small bottleneck trains. Docker `openpi:latest`, mounts for `src`, `weights`, `assets`, `checkpoints`. Secrets from `$HOME/.env` (`WANDB_API_KEY`, `HF_TOKEN`).

---

## Tests (CPU)

Same shape as the green-button RLT gates:

- Config: three-cam keys, `prompt_from_task`, freeze filter leaves encoder/decoder trainable, loader path `checkpoints/pi05_busybox_multitask/params`, 20k / batch 16 / save-once.
- Script: config name, Hub repo `pravsels/pi05_busybox_multitask`, 80 GB floor, “refusing to recompute”, `WANDB_MODE=online`, Orbax sentinels.

Existing dummy `debug_pi0_rl` / default-key RLT path and the old bimanual `pi05_rlt_busybox_multitask` stay unchanged.

---

## Failure modes

| Risk | Handling |
|---|---|
| Weight loader pointed at `step_N/` | Hub root is `params/`; script checks `_METADATA` + `manifest.ocdbt` |
| Recomputed norm stats drift from the VLA | Copy Hub `assets/`; refuse if missing |
| Image still has PyPI RCW (shuffled prompts) | `uv sync` from host lockfile; fail unless `rcw_sha_ok` + `prompt_ok` |
| Leftover shuffled Hub tree on a reused VM | Refuse unless README contains W&B `4ym0qegc` |
| Training the old bimanual config by name | New config is `*_singlearm`; leave `pi05_rlt_busybox_multitask` alone |
| `WANDB_MODE` offline from `train.py` | Export `online` in the Docker env |
| Disk fills on the ~75 GB download | Require ≥200 GB free before download |
| 40 GB A100 | Refuse unless GPU ≥80 GB |

---

## Files

| Path | Role |
|---|---|
| `src/openpi/training/config.py` | New `TrainConfig` `pi05_rlt_busybox_multitask_singlearm` |
| `src/openpi/training/config_test.py` | CPU config + script gates |
| `slurm/train_busybox_multitask_singlearm_rlt_gcloud.sh` | GCloud Docker launcher |
| `scripts/check_busybox_multitask_rcw_prompts.py` | RCW git SHA + remapped prompt gate |
| `run_logs/2026-09-03_train_rlt_busybox_multitask_singlearm_gcloud.md` | After launch |

Publish later (not this slice) to `pravsels/pi05_rlt_busybox_multitask_singlearm`: `params/` + `assets/` + README at repo root, no `train_state`.

---

## Implementation order

1. Add `pi05_rlt_busybox_multitask_singlearm` and the matching CPU config test.
2. Add the GCloud launcher and the script-string test.
3. Spin / reuse an `a2-ultragpu-1g` VM, pull the branch, download the Hub VLA, start the 20k run.
4. Write the run log at launch.
