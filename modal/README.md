# Training openpi on Modal

The Isambard allocation for `AIRR-RA - 2026-01-01 - Safe Robotics Round 3`
ended on 2026-08-05, so `slurm/` is no longer runnable. `train_modal.py` is the
replacement: same configs, same phases, same Hugging Face layout, different
scheduler.

| | SLURM (Isambard) | Modal |
|---|---|---|
| Launch | `bash slurm/submit_busybox_tasks.sh` | `modal run modal/train_modal.py::main` |
| Norm stats | inside the GPU job | separate CPU function |
| Checkpoints | `/scratch/.../checkpoints` | Volume `openpi-train-data` |
| Base weights | staged by hand on scratch | Volume `openpi-base-weights` |
| Interruption | `--requeue` + `--resume` | `modal.Retries` + `--resume` |
| Publish | `afterok` dependency job | `publish` function |

## One-time setup

Authenticate against the company workspace and confirm the two secrets exist:

```bash
modal profile current          # expect the safe-robotics workspace
modal secret list              # expect huggingface-secret and wandb-secret
```

Both of those are shared workspace secrets, so they do not necessarily belong
to you. That matters: an HF token owned by one account cannot push to another
account's namespace, and a W&B key cannot log runs to an entity it has no
access to. Modal never discloses secret values, so the only way to find out
what a shared secret can do is to use it:

```bash
modal run external/openpi/modal/train_modal.py::check
```

That prints the HF account, its orgs, the token's role, and the W&B account and
default entity. The token needs **write** scope on whichever namespace you are
publishing to, otherwise training runs for eight hours and then fails at the
upload.

As of 2026-08, the shared `huggingface-secret` holds a fine-grained token owned
by `villekuosmanen`, scoped to write in that account's namespace only. It cannot
publish to `lorenzouttini/*`. `wandb-secret` is likewise Ville's. Add your own
rather than editing the shared ones, which other apps depend on:

```bash
modal secret create huggingface-lorenzo HF_TOKEN=hf_...
OPENPI_HF_SECRET=huggingface-lorenzo modal run ...::check
```

Either token name works — Modal's Hugging Face template and hand-made secrets
disagree on whether it is `HF_TOKEN` or `HUGGING_FACE_ACCESS_TOKEN`, so the app
accepts both and normalises.

`OPENPI_HF_SECRET`, `OPENPI_WANDB_SECRET` and `OPENPI_WANDB_ENTITY` all
override at launch time. W&B entity defaults to whatever the API key's own
default is, which for a key belonging to a team account is the team, not your
personal namespace — set it explicitly if you want runs next to the Isambard
ones in `uttini-lorenzo`. Pass `--no-wandb` to skip logging entirely.

These overrides are read on your machine and passed to the remote functions as
arguments. They cannot be read inside the container: Modal re-imports this
module there without your local environment, so anything derived from
`os.environ` at module level silently falls back to its default.

## W&B permissions

`main` runs a CPU preflight before booking a GPU: it creates and deletes a
throwaway run to prove the entity/project is writable, and returns the entity
that worked. This exists because `wandb.init` failures are deterministic but
surface deep inside `scripts/train.py`, so on the GPU path each one costs a
container start times the retry count.

To debug entity permissions on their own:

```bash
modal run external/openpi/modal/train_modal.py::wandb_preflight \
    --project busybox_multitask_pi05
```

With no `--entity` it tries the key's default, then your username, then each
team, and reports the error for each. Note that a W&B account created inside an
organisation may have no personal entity at all, in which case only the team
entities work.

## Running

Always smoke test first. A hundred steps costs a couple of dollars and tells
you whether the image builds, the weights load, and the model fits in memory —
the three things that actually go wrong:

```bash
modal run external/openpi/modal/train_modal.py::main \
    --config-name pi05_busybox_multitask \
    --num-train-steps 100 --skip-publish
```

Then the real run. Use `--detach` so it survives your laptop closing:

```bash
modal run --detach external/openpi/modal/train_modal.py::main \
    --config-name pi05_busybox_multitask \
    --hf-repo-id lorenzouttini/pi05-so101-busybox-multitask-modal
```

The first run pays for the image build and the ~14GB `pi05_base` download.
Both are cached on Volumes afterwards, so later runs start in a couple of
minutes.

Phases can be skipped independently, which is what you want when a run dies
partway and the setup is already done:

```bash
modal run --detach external/openpi/modal/train_modal.py::main \
    --config-name pi05_busybox_multitask --skip-prepare
```

## Closing your laptop mid-run

`--detach` keeps the training container alive on Modal when the client
disconnects, so the run itself survives sleep. `main` is a *local* entrypoint
though: it orchestrates the phases from your machine, so if the process is
suspended before training returns, the publish step never fires. The
checkpoints are on a Volume, so just publish afterwards:

```bash
OPENPI_HF_SECRET=huggingface-lorenzo \
modal run external/openpi/modal/train_modal.py::publish \
    --config-name pi05_busybox_multitask \
    --exp-name pi05_busybox_multitask \
    --hf-repo-id lorenzouttini/pi05-so101-busybox-multitask-modal
```

## GPU and memory

Default is `H100:2` with `--fsdp-devices 2`.

openpi documents full fine-tuning as needing more than 70GB, so an 80GB H100 is
tight. FSDP is therefore on by default here and was not on Isambard's 96GB
GH200s — it shards parameters and optimizer state across both GPUs at some cost
in speed. Measured on 2×H100 with `pi05_busybox_multitask` at batch 32, this
configuration fits and runs at 1.2-1.3 it/s.

If a new config OOMs, go up in memory rather than fighting it:

```bash
OPENPI_MODAL_GPU=H200:2 modal run external/openpi/modal/train_modal.py::main ...
```

H200 is 141GB a card for about 15% more per hour. Other overrides:
`OPENPI_MODAL_FSDP_DEVICES`, `OPENPI_MODAL_SAVE_INTERVAL`.

`save_interval` defaults to 1000 here against 5000 in the TrainConfig, because
Modal reclaims containers and the interval sets how much work a preemption
costs. It is close to free: a save blocks training for ~22s and then finishes
the remaining ~105s on a background thread, so ten saves add about 4 minutes to
a run.

## Cost

About **$25** for a 10k-step baseline run on 2×H100 — roughly 2.5 hours at
$7.90/hour of GPU plus $1.26/hour of CPU and memory, plus about $1 for the CPU
prepare phase. Measured from the first `pi05_busybox_multitask` run rather than
projected.

Note for anyone comparing against the earlier estimate: 8 hours per run was
taken from the Isambard SLURM `--time` limit, which was a requested ceiling and
roughly 3x the real training time.

A100-80GB looks cheaper per hour at $2.50 but is around 1.7x slower, so it
costs more per run. Do not use it.

## Monitoring

```bash
modal app list
modal app logs openpi-train
```

Weights & Biases gets the training curves live, which is a straight improvement
on Isambard — there `WANDB_MODE=offline` meant syncing runs by hand afterwards.
