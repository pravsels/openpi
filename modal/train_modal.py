"""Modal launcher for openpi pi0.5 training.

Replacement for the Isambard SLURM path (``slurm/train_busybox_tasks_slurm.sh``
plus ``slurm/publish_busybox_tasks_slurm.sh``) after the Isambard allocation
expired on 2026-08-05. Same three phases, same config names, same Hugging Face
layout — only the scheduler changes.

    prepare  (CPU)  stage pi05_base weights, compute norm stats + valid indices
    train    (GPU)  uv run scripts/train.py <config>
    publish  (CPU)  upload params/ + assets/ to Hugging Face

Keeping ``prepare`` off the GPU matters: norm stats over a full LeRobot dataset
take the better part of an hour and cost ~$9/h on 2xH100 versus ~$0.40/h on CPU.

Why the phases are split rather than one long function: Modal reclaims
containers, and a preempted GPU function restarts from the top. Everything the
GPU phase needs is already on a Volume by the time it starts, so a restart
resumes from the last Orbax checkpoint instead of redoing an hour of setup.

Usage
-----
Smoke test first — validates the image, GPU memory and throughput for a couple
of dollars before committing to a full run::

    modal run external/openpi/modal/train_modal.py::main \
        --config-name pi05_busybox_multitask \
        --num-train-steps 100 --skip-publish

Full run, train then publish::

    modal run --detach external/openpi/modal/train_modal.py::main \
        --config-name pi05_busybox_multitask \
        --hf-repo-id lorenzouttini/pi05-so101-busybox-multitask-modal

Use ``--detach`` for real runs so the job survives your laptop closing.

The GPU is set by ``OPENPI_MODAL_GPU`` (default ``H100:2``). pi0.5 full
fine-tuning needs >70GB and the SLURM profile ran batch 16/GPU inside 91GB
usable on a 96GB GH200, so 80GB is tight: FSDP_DEVICES defaults to 2 to shard
parameters and optimizer state across both GPUs. If it still OOMs, either drop
the batch size or set ``OPENPI_MODAL_GPU=H200:2`` for 141GB a card.

Prerequisites in the Modal workspace:
  - secret ``huggingface-secret`` with HF_TOKEN (needs *write* scope to publish)
  - secret ``wandb-secret`` with WANDB_API_KEY (or pass --no-wandb)
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import modal

APP_NAME = "openpi-train"

# Repo root — this file lives at <openpi>/modal/train_modal.py.
LOCAL_ROOT = pathlib.Path(__file__).parent.parent
REMOTE_ROOT = "/root/openpi"
VENV_PYTHON = "/.venv/bin/python"

# Volume mount points. checkpoint_base_dir and assets_dir are both plain
# TrainConfig fields, so pointing them into /data avoids any bind-mount
# trickery — unlike the SLURM script, which had to bind over ./checkpoints.
DATA_DIR = "/data"
WEIGHTS_DIR = f"{REMOTE_ROOT}/weights"
HF_CACHE_DIR = "/root/.cache/huggingface"

GPU = os.environ.get("OPENPI_MODAL_GPU", "H100:2")
FSDP_DEVICES = int(os.environ.get("OPENPI_MODAL_FSDP_DEVICES", "2"))
# The Isambard config saves every 5k steps. That is 3.6h of work at risk per
# preemption at ~1,390 steps/h; 1k steps puts the exposure at ~43 minutes.
SAVE_INTERVAL = int(os.environ.get("OPENPI_MODAL_SAVE_INTERVAL", "1000"))

# The workspace secrets are shared and not necessarily owned by whoever is
# launching the run, so both the secret names and the W&B entity are
# overridable. Run ::check_secrets to see who the current ones belong to.
HF_SECRET_NAME = os.environ.get("OPENPI_HF_SECRET", "huggingface-secret")
WANDB_SECRET_NAME = os.environ.get("OPENPI_WANDB_SECRET", "wandb-secret")
# Empty means "whatever the API key's own default entity is", which is the only
# safe default when the key may belong to someone else's account.
WANDB_ENTITY = os.environ.get("OPENPI_WANDB_ENTITY", "")

BASE_WEIGHTS_URL = "gs://openpi-assets/checkpoints/pi05_base"

# The repo is only a few MB, but weights/ must stay absent so the Volume can
# mount onto it — Modal refuses to mount over a non-empty path.
IMAGE_IGNORE = [
    "**/.git",
    "**/.git/**",
    "**/__pycache__",
    "**/*.pyc",
    "**/.venv",
    "**/.venv/**",
    "weights",
    "weights/**",
    "checkpoints",
    "checkpoints/**",
    "assets",
    "assets/**",
    "wandb",
    "wandb/**",
    "run_logs/**",
    "hf_publish/**",
    "eval_logs/**",
    "eval_outputs/**",
]

# Ported from docker/Dockerfile, the recipe the Isambard container was built
# from. `uv sync` against the repo's own uv.lock is what makes this reliable:
# the lock already resolves the jax/torch/lerobot constraints that are painful
# to reproduce by hand.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install(
        "git",
        "git-lfs",
        "ffmpeg",
        "libgl1",
        "build-essential",
        "cmake",
        "pkg-config",
        "clang",
        # evdev, pulled in transitively by lerobot, needs kernel headers.
        "linux-libc-dev",
        "libavcodec-dev",
        "libavformat-dev",
        "libswscale-dev",
        "libavfilter-dev",
        "libavdevice-dev",
    )
    # Into the container's system Python, which is what Modal runs the function
    # bodies with — openpi itself lives in /.venv and is only ever invoked as a
    # subprocess via VENV_PYTHON.
    .pip_install("uv", "huggingface_hub")
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "GIT_LFS_SKIP_SMUDGE": "1",
            "UV_PYTHON": "3.11",
            "UV_PROJECT_ENVIRONMENT": "/.venv",
            "HF_HOME": HF_CACHE_DIR,
            "OPENPI_DATA_HOME": DATA_DIR,
            # openpi's README: raise this or full fine-tuning OOMs.
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
        }
    )
    .add_local_dir(LOCAL_ROOT, REMOTE_ROOT, copy=True, ignore=IMAGE_IGNORE)
    .run_commands(
        f"cd {REMOTE_ROOT} && uv sync --group dev",
        f"cd {REMOTE_ROOT} && uv pip install -e .",
        f"cd {REMOTE_ROOT} && uv pip install --prerelease=allow decord",
        # uv writes into the HF cache during sync; the Volume cannot mount onto
        # a non-empty path.
        f"rm -rf {HF_CACHE_DIR}",
    )
)

app = modal.App(APP_NAME)

# Split by lifetime, not by content: base weights are written once and read
# forever, run outputs churn every run, and the HF cache is shared across runs.
weights_volume = modal.Volume.from_name("openpi-base-weights", create_if_missing=True)
data_volume = modal.Volume.from_name("openpi-train-data", create_if_missing=True)
hf_volume = modal.Volume.from_name("openpi-hf-cache", create_if_missing=True)

VOLUMES = {
    WEIGHTS_DIR: weights_volume,
    DATA_DIR: data_volume,
    HF_CACHE_DIR: hf_volume,
}

hf_secret = modal.Secret.from_name(HF_SECRET_NAME)
wandb_secret = modal.Secret.from_name(WANDB_SECRET_NAME)

# Deliberately not the training image: checking a credential should not cost a
# multi-minute build of the full CUDA stack.
check_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub", "wandb"
)


@app.function(image=check_image, secrets=[wandb_secret], timeout=600)
def wandb_preflight(project: str, entity: str = "") -> str:
    """Prove we can write to entity/project, on CPU, before booking a GPU.

    wandb.init failures are deterministic config errors, but they surface deep
    inside scripts/train.py after the model and data loader are built — so on
    the GPU path each one costs a container start, times the retry count.
    Creating and deleting a throwaway run here costs cents and returns the
    entity that actually worked.
    """
    import wandb

    os.environ["WANDB_MODE"] = "online"
    api = wandb.Api()
    viewer = api.viewer
    print(f"account: {viewer.username}")
    print(f"default entity: {viewer.entity}")
    print(f"teams: {', '.join(viewer.teams) if viewer.teams else '(none)'}")

    # Try the requested entity first, then the key's own default, then each
    # team. One of these is almost always writable.
    candidates: list[str] = []
    for name in [entity, viewer.entity, viewer.username, *viewer.teams]:
        if name and name not in candidates:
            candidates.append(name)

    errors = []
    for candidate in candidates:
        print(f"\nTrying {candidate}/{project} ...")
        try:
            run = wandb.init(
                project=project, entity=candidate, name="preflight", job_type="preflight"
            )
            run_path = f"{candidate}/{project}/{run.id}"
            wandb.finish()
            try:
                api.run(run_path).delete()
            except Exception as exc:  # noqa: BLE001 - cleanup is best effort
                print(f"  (left preflight run behind, delete it by hand: {exc})")
            print(f"OK: {candidate}/{project} is writable")
            return candidate
        except Exception as exc:  # noqa: BLE001 - report and try the next one
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            errors.append(f"{candidate}: {exc}")

    raise RuntimeError(
        "No writable W&B entity found for project "
        f"'{project}'. Tried:\n  " + "\n  ".join(errors)
    )


@app.function(image=check_image, secrets=[hf_secret, wandb_secret], timeout=300)
def check_secrets(hf_label: str, wandb_label: str) -> None:
    """Report which accounts the workspace secrets belong to, and their scope.

    Modal never discloses secret values, so the only way to find out what a
    shared secret can actually do is to use it.

    The labels are passed in rather than read from HF_SECRET_NAME: the module is
    re-imported inside the container, where the launch-time OPENPI_* overrides
    do not exist, so those globals would always report their defaults.
    """
    from huggingface_hub import HfApi

    print(f"--- {hf_label} ---")
    token = _normalize_hf_token()
    if not token:
        print(f"No token found. Expected one of: {', '.join(HF_TOKEN_ALIASES)}")
    else:
        info = HfApi().whoami(token=token)
        print(f"account:   {info.get('name')} ({info.get('type')})")
        orgs = [o.get("name") for o in info.get("orgs", [])]
        print(f"orgs:      {', '.join(orgs) if orgs else '(none)'}")

        access = (info.get("auth") or {}).get("accessToken") or {}
        role = access.get("role")
        print(f"token:     {access.get('displayName')} role={role}")
        if role == "fineGrained":
            for scope in (access.get("fineGrained") or {}).get("scoped", []):
                entity = (scope.get("entity") or {}).get("name")
                print(f"  scoped:  {entity} -> {scope.get('permissions')}")
        elif role != "write":
            print("  WARNING: not a write token, publishing will fail.")

    print(f"\n--- {wandb_label} ---")
    if not os.environ.get("WANDB_API_KEY"):
        print("No WANDB_API_KEY key in this secret.")
    else:
        import wandb

        viewer = wandb.Api().viewer
        print(f"account:   {viewer.username}")
        print(f"entity:    {viewer.entity}   (the default runs will log to)")
        print(f"teams:     {', '.join(viewer.teams) if viewer.teams else '(none)'}")


def _paths(config_name: str, exp_name: str) -> tuple[str, str]:
    """Checkpoint dir and assets dir for a run, matching the SLURM layout."""
    checkpoint_dir = f"{DATA_DIR}/checkpoints/{config_name}/{exp_name}"
    assets_dir = f"{DATA_DIR}/assets/{config_name}/{exp_name}/assets"
    return checkpoint_dir, assets_dir


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REMOTE_ROOT, check=True, stdout=sys.stdout, stderr=sys.stderr)


def _capture(cmd: list[str]) -> str:
    out = subprocess.run(cmd, cwd=REMOTE_ROOT, check=True, capture_output=True, text=True)
    return out.stdout.strip()


# huggingface_hub only reads HF_TOKEN, but Modal's Hugging Face secret template
# and hand-rolled secrets disagree about the name.
HF_TOKEN_ALIASES = (
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGING_FACE_ACCESS_TOKEN",
    "HUGGINGFACE_ACCESS_TOKEN",
    "HF_API_TOKEN",
)


def _normalize_hf_token() -> str | None:
    for name in HF_TOKEN_ALIASES:
        value = os.environ.get(name)
        if value:
            os.environ["HF_TOKEN"] = value
            return value
    return None


@app.function(
    image=image,
    volumes=VOLUMES,
    secrets=[hf_secret],
    cpu=8,
    memory=32768,
    timeout=6 * 60 * 60,
)
def prepare(config_name: str, exp_name: str) -> str:
    """Stage base weights and precompute norm stats. CPU only, by design.

    Returns the config's W&B project name.
    """
    _normalize_hf_token()
    _, assets_dir = _paths(config_name, exp_name)
    pathlib.Path(assets_dir).mkdir(parents=True, exist_ok=True)

    base_params = pathlib.Path(WEIGHTS_DIR) / "pi05_base" / "params"
    if base_params.exists():
        print(f"Base weights already staged at {base_params}")
    else:
        print(f"Downloading {BASE_WEIGHTS_URL} ...")
        _run(
            [
                VENV_PYTHON,
                "-c",
                (
                    "import shutil;"
                    "from openpi.shared import download;"
                    f"src = download.maybe_download('{BASE_WEIGHTS_URL}');"
                    f"shutil.copytree(src, '{WEIGHTS_DIR}/pi05_base', dirs_exist_ok=True);"
                    f"print('Staged base weights to {WEIGHTS_DIR}/pi05_base')"
                ),
            ]
        )
        weights_volume.commit()

    norm_stats = pathlib.Path(assets_dir) / "norm_stats.json"
    per_timestep = pathlib.Path(assets_dir) / "norm_stats_actions_per_timestep.json"
    if norm_stats.exists() and per_timestep.exists():
        print("Norm stats already present, skipping precompute.")
    else:
        _run(
            [
                VENV_PYTHON,
                "scripts/compute_norm_stats_per_timestep.py",
                f"--config-name={config_name}",
                f"--assets-dir={assets_dir}",
            ]
        )

    # openpi expects a valid_indices file alongside the norm stats; every frame
    # of these datasets is usable, so this is just the full range.
    valid_indices = pathlib.Path(assets_dir) / "valid_indices.txt"
    if valid_indices.exists():
        print("valid_indices.txt already present.")
    else:
        _run(
            [
                VENV_PYTHON,
                "-c",
                (
                    "from openpi.training import config as _config;"
                    "from openpi.training.data_loader import create_torch_dataset;"
                    f"cfg = _config.get_config('{config_name}');"
                    "dc = cfg.data.create(cfg.assets_dirs, cfg.model);"
                    "ds = create_torch_dataset(dc, cfg.model.action_horizon, cfg.model);"
                    f"open('{valid_indices}', 'w').write(','.join(str(i) for i in range(len(ds))));"
                    "print('Wrote', len(ds), 'valid indices')"
                ),
            ]
        )

    data_volume.commit()
    hf_volume.commit()

    # Returned so the launcher can preflight W&B without needing openpi locally.
    return _capture(
        [
            VENV_PYTHON,
            "-c",
            (
                "from openpi.training import config as _config;"
                f"print(_config.get_config('{config_name}').project_name)"
            ),
        ]
    )


@app.function(
    image=image,
    gpu=GPU,
    volumes=VOLUMES,
    secrets=[hf_secret, wandb_secret],
    cpu=16,
    memory=65536,
    timeout=24 * 60 * 60,
    # Modal reclaims containers. Each retry re-enters from the top, sees the
    # existing checkpoint dir, and passes --resume. Kept low because Retries
    # cannot tell a preemption from a deterministic config error, and each
    # attempt costs a GPU container start.
    retries=modal.Retries(max_retries=2, initial_delay=0.0),
)
def train(
    config_name: str,
    exp_name: str,
    num_train_steps: int = 0,
    wandb: bool = True,
    fresh: bool = False,
    # Passed in, not read from the module globals: the container re-imports this
    # file without the launch-time OPENPI_* env vars, so reading them here would
    # silently ignore whatever the caller asked for.
    fsdp_devices: int = FSDP_DEVICES,
    save_interval: int = SAVE_INTERVAL,
    wandb_entity: str = WANDB_ENTITY,
) -> None:
    _normalize_hf_token()
    checkpoint_dir, assets_dir = _paths(config_name, exp_name)
    if wandb_entity:
        os.environ["WANDB_ENTITY"] = wandb_entity

    # RLT configs point their weight loader at a repo-relative path such as
    # "checkpoints/<config>/<exp>/9999/params", which worked on Isambard because
    # SLURM bind-mounted the checkpoints tree into the repo. Reproduce that
    # layout with a symlink so those configs stay environment-agnostic.
    repo_checkpoints = pathlib.Path(REMOTE_ROOT) / "checkpoints"
    if not repo_checkpoints.exists():
        repo_checkpoints.symlink_to(f"{DATA_DIR}/checkpoints")
        print(f"Linked {repo_checkpoints} -> {DATA_DIR}/checkpoints")

    # scripts/train.py does os.environ.setdefault("WANDB_MODE", "offline") for
    # Isambard's air-gapped compute nodes. Modal has internet, and an offline
    # run would be written into the container filesystem and lost on exit, so
    # opt back in. WANDB_DIR keeps whatever wandb buffers locally on a Volume,
    # so a run that drops offline mid-flight can still be synced afterwards.
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_DIR"] = f"{DATA_DIR}/wandb"
    pathlib.Path(f"{DATA_DIR}/wandb").mkdir(parents=True, exist_ok=True)

    if not (pathlib.Path(WEIGHTS_DIR) / "pi05_base" / "params").exists():
        raise RuntimeError("Base weights missing — run prepare first.")
    if not (pathlib.Path(assets_dir) / "norm_stats.json").exists():
        raise RuntimeError(f"Norm stats missing in {assets_dir} — run prepare first.")

    ckpt_root = pathlib.Path(checkpoint_dir)
    existing = (
        [p for p in ckpt_root.glob("*") if p.is_dir() and p.name != "assets"]
        if ckpt_root.exists() and not fresh
        else []
    )
    # A leftover smoke-test checkpoint would otherwise be resumed from, and the
    # resume path reads wandb_id.txt and calls wandb.init(resume="must") — which
    # fails if that run was never uploaded.
    if fresh and ckpt_root.exists():
        print(f"--fresh: discarding existing checkpoints in {checkpoint_dir}")

    cmd = [
        VENV_PYTHON,
        "scripts/train.py",
        config_name,
        f"--exp-name={exp_name}",
        f"--assets-dir={assets_dir}",
        f"--checkpoint-base-dir={DATA_DIR}/checkpoints",
        f"--fsdp-devices={fsdp_devices}",
        f"--save-interval={save_interval}",
        # openpi defaults to 2; this container has 16 CPUs to feed 2 GPUs.
        "--num-workers=4",
        "--resume" if existing else "--overwrite",
    ]
    if num_train_steps:
        cmd.append(f"--num-train-steps={num_train_steps}")
    if not wandb:
        cmd.append("--no-wandb-enabled")

    if existing:
        print(f"Resuming from {len(existing)} checkpoint(s) in {checkpoint_dir}")

    try:
        _run(cmd)
    finally:
        # Commit even on failure so a preempted run keeps its last checkpoint.
        data_volume.commit()

    print(f"Training finished. Checkpoints at {checkpoint_dir}")


@app.function(
    image=image,
    volumes=VOLUMES,
    secrets=[hf_secret],
    cpu=4,
    memory=16384,
    timeout=4 * 60 * 60,
)
def publish(
    config_name: str,
    exp_name: str,
    hf_repo_id: str,
    steps: str = "9999 10000",
) -> None:
    """Upload params/ + assets/ (not train_state/) — mirrors the SLURM publish.

    Space-separated rather than a list so this is directly runnable as
    ``modal run ...::publish`` when the launcher died before reaching it. The
    checkpoints live on a Volume, so publishing is safe to do at any later time.
    """
    from huggingface_hub import HfApi

    if not _normalize_hf_token():
        raise RuntimeError(
            "No Hugging Face token in the mounted secret. "
            f"Expected one of: {', '.join(HF_TOKEN_ALIASES)}"
        )
    checkpoint_dir, _ = _paths(config_name, exp_name)
    api = HfApi()
    api.create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True)
    print(f"Target repo: {hf_repo_id}")

    uploaded = []
    for step in steps.split():
        folder = pathlib.Path(checkpoint_dir) / step
        if not folder.is_dir():
            print(f"SKIP step {step}: not found at {folder}")
            continue
        print(f"Uploading step {step} from {folder} ...")
        api.upload_folder(
            folder_path=str(folder),
            repo_id=hf_repo_id,
            path_in_repo=f"step_{step}",
            repo_type="model",
            ignore_patterns=["train_state/**"],
            commit_message=f"Add {config_name} checkpoint step {step}",
        )
        uploaded.append(step)

    if not uploaded:
        raise RuntimeError("No checkpoints uploaded — none of the requested steps exist.")
    print(f"Uploaded steps: {', '.join(uploaded)}")
    print(f"https://huggingface.co/{hf_repo_id}")


@app.local_entrypoint()
def check() -> None:
    """Report who the mounted secrets belong to. Run this before a real launch."""
    check_secrets.remote(HF_SECRET_NAME, WANDB_SECRET_NAME)


@app.local_entrypoint()
def main(
    config_name: str = "pi05_busybox_multitask",
    exp_name: str = "",
    hf_repo_id: str = "",
    num_train_steps: int = 0,
    wandb: bool = True,
    fresh: bool = False,
    skip_prepare: bool = False,
    skip_train: bool = False,
    skip_publish: bool = False,
    # keep_period=10000 deletes the intermediate checkpoint, so the final step
    # is the only one persisted. 9999 is the 0-indexed last step of a 10k run;
    # 10000 is a fallback in case the trainer wrote that name.
    publish_steps: str = "9999 10000",
) -> None:
    exp = exp_name or config_name

    print(f"config={config_name} exp={exp} gpu={GPU} fsdp_devices={FSDP_DEVICES}")
    print(f"secrets: hf={HF_SECRET_NAME} wandb={WANDB_SECRET_NAME}")
    print("wandb entity: " + (WANDB_ENTITY or "(the API key's own default)"))

    project = ""
    if not skip_prepare:
        project = prepare.remote(config_name, exp)

    entity = WANDB_ENTITY
    if wandb and not skip_train:
        if project:
            entity = wandb_preflight.remote(project, WANDB_ENTITY)
            print(f"W&B will log to {entity}/{project}")
        else:
            print("Skipping W&B preflight (--skip-prepare gave no project name).")

    if not skip_train:
        train.remote(
            config_name,
            exp,
            num_train_steps=num_train_steps,
            wandb=wandb,
            fresh=fresh,
            fsdp_devices=FSDP_DEVICES,
            save_interval=SAVE_INTERVAL,
            wandb_entity=entity,
        )

    if not skip_publish:
        if not hf_repo_id:
            print("No --hf-repo-id given, skipping publish.")
            return
        publish.remote(config_name, exp, hf_repo_id, publish_steps)
