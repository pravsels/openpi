"""GMAN helpers: GPU preflight and command payloads (CRA-style secret refs)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Sequence

REQUIRED_GPUS = 8
MIN_VRAM_MIB = 80_000
SMOKE_STEPS = ("5", "9")
# CRA documents chip="h100-1", which is a 1-GPU node. This job needs 8 GPUs.
GMAN_FORBIDDEN_CHIPS = frozenset({"h100-1"})
GMAN_NODE_REQUIREMENT = (
    "create_node must be 8x >=80GB H100. Do not use chip=h100-1 (CRA 1-GPU)."
)
BUSYBOX_CLONE_PATHS = (
    "gman/train.sh",
    "gman/bootstrap.sh",
    "scripts/gman_payload.py",
    "scripts/train.py",
    "src/openpi/models/pi0.py",
    "src/openpi/training/config.py",
)
BUSYBOX_CONFIG_MARKER = "pi0_busybox_push_green_button"

TYPED_SECRETS = {
    "HF_TOKEN": {"secret": "hf-token"},
    "WANDB_API_KEY": {"secret": "wandb-api-key"},
    "GITHUB_TOKEN": {"secret": "github-repo"},
}


def require_eight_80gb_gpus(memory_mib: Sequence[int]) -> None:
    memories = [int(value) for value in memory_mib]
    if len(memories) != REQUIRED_GPUS:
        raise SystemExit(f"Expected exactly {REQUIRED_GPUS} GPUs, found {len(memories)}")
    for index, memory in enumerate(memories):
        if memory < MIN_VRAM_MIB:
            raise SystemExit(
                f"Expected at least {MIN_VRAM_MIB} MiB on GPU {index}, found {memory} MiB"
            )


def require_jax_device_count(device_count: int, expected: int = REQUIRED_GPUS) -> None:
    if int(device_count) != expected:
        raise SystemExit(f"Expected {expected} JAX devices, found {device_count}")


def assert_not_forbidden_chip(chip: str) -> None:
    if chip in GMAN_FORBIDDEN_CHIPS:
        raise SystemExit(f"ERROR: chip={chip} is a 1-GPU CRA SKU. {GMAN_NODE_REQUIREMENT}")


def bootstrap_shell_command(repo_root: Path) -> str:
    """Run the inlined bootstrap with Bash before $HOME/openpi exists."""
    script = (Path(repo_root) / "gman" / "bootstrap.sh").read_text().rstrip("\n")
    return (
        "bash --noprofile --norc -s <<'OPENPI_BOOTSTRAP'\n"
        f"{script}\n"
        "OPENPI_BOOTSTRAP"
    )


def train_shell_command(*, config_name: str, smoke: bool) -> str:
    smoke_flag = "SMOKE=1 " if smoke else ""
    return (
        "bash -lc "
        f"'cd \"$HOME/openpi\" && {smoke_flag}CONFIG_NAME={config_name} bash gman/train.sh'"
    )


def delete_smoke_shell_command(*, config_name: str) -> str:
    return (
        "bash -lc "
        f"'cd \"$HOME/openpi\" && CONFIG_NAME={config_name} bash gman/delete_smoke.sh'"
    )


def experiment_name(config_name: str, *, smoke: bool) -> str:
    return f"{config_name}_smoke" if smoke else config_name


def refuse_production_if_smoke_checkpoints(
    checkpoint_dir: Path, smoke_steps: Sequence[str] = SMOKE_STEPS
) -> None:
    root = Path(checkpoint_dir)
    if not root.is_dir():
        return
    existing = {path.name for path in root.iterdir() if path.is_dir() and path.name != "assets"}
    smoke = {str(step) for step in smoke_steps}
    if smoke <= existing and not (existing - smoke):
        raise SystemExit(
            f"ERROR: {root} only has smoke checkpoints {sorted(smoke)}; "
            "delete them (or use a production exp name) before 30k"
        )


def assert_openpi_branch_has_busybox_assets(repo_dir: Path) -> None:
    root = Path(repo_dir)
    missing = [rel for rel in BUSYBOX_CLONE_PATHS if not (root / rel).exists()]
    if missing:
        raise SystemExit(
            "ERROR: cloned openpi is missing "
            f"{', '.join(missing)}. Push the task branch before bootstrap."
        )
    config_text = (root / "src/openpi/training/config.py").read_text()
    if BUSYBOX_CONFIG_MARKER not in config_text:
        raise SystemExit(
            f"ERROR: cloned branch does not contain {BUSYBOX_CONFIG_MARKER}. "
            "Push the task branch first."
        )


def training_request_payload(*, command: str, mission: str) -> dict[str, Any]:
    """JSON body for `gman api post /nodes/<node>/commands`.

    Secrets must be typed refs. Do not put secret *names* in env values
    (CRA already failed that way with a 7-character W&B key).
    """
    return {
        "command": command,
        "detach": True,
        "mission": mission,
        "env": {
            **TYPED_SECRETS,
            "WANDB_ENTITY": "pravsels",
            "WANDB_MODE": "online",
        },
        "hold_on_failure_minutes": 30,
    }


def bootstrap_request_payload(*, command: str, mission: str) -> dict[str, Any]:
    return {
        "command": command,
        "detach": True,
        "mission": mission,
        "env": {
            "GITHUB_TOKEN": {"secret": "github-repo"},
            "HF_TOKEN": {"secret": "hf-token"},
        },
        "hold_on_failure_minutes": 30,
    }
