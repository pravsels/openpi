from __future__ import annotations

from pathlib import Path

import pytest

from scripts.gman_payload import GMAN_FORBIDDEN_CHIPS
from scripts.gman_payload import GMAN_NODE_REQUIREMENT
from scripts.gman_payload import assert_not_forbidden_chip
from scripts.gman_payload import assert_openpi_branch_has_busybox_assets
from scripts.gman_payload import bootstrap_request_payload
from scripts.gman_payload import bootstrap_shell_command
from scripts.gman_payload import experiment_name
from scripts.gman_payload import refuse_production_if_smoke_checkpoints
from scripts.gman_payload import require_eight_80gb_gpus
from scripts.gman_payload import require_jax_device_count
from scripts.gman_payload import training_request_payload
from scripts.gman_payload import train_shell_command


def test_require_eight_80gb_gpus_accepts_eight_h100s():
    require_eight_80gb_gpus([81920] * 8)


def test_require_eight_80gb_gpus_rejects_count():
    with pytest.raises(SystemExit, match="Expected exactly 8 GPUs"):
        require_eight_80gb_gpus([81920] * 1)


def test_require_eight_80gb_gpus_rejects_small_vram():
    memories = [81920] * 8
    memories[3] = 32768
    with pytest.raises(SystemExit, match="GPU 3"):
        require_eight_80gb_gpus(memories)


def test_training_payload_uses_typed_secret_refs():
    payload = training_request_payload(command="uv run scripts/train.py", mission="pi-busybox")
    assert payload["detach"] is True
    assert payload["hold_on_failure_minutes"] == 30
    assert payload["env"]["WANDB_MODE"] == "online"
    assert payload["env"]["WANDB_ENTITY"] == "pravsels"
    assert payload["env"]["HF_TOKEN"] == {"secret": "hf-token"}
    assert payload["env"]["WANDB_API_KEY"] == {"secret": "wandb-api-key"}
    assert payload["env"]["GITHUB_TOKEN"] == {"secret": "github-repo"}
    for key in ("HF_TOKEN", "WANDB_API_KEY", "GITHUB_TOKEN"):
        assert not isinstance(payload["env"][key], str)


def test_bootstrap_payload_does_not_inline_tokens():
    payload = bootstrap_request_payload(command="./gman/bootstrap.sh", mission="pi-busybox")
    assert payload["env"]["GITHUB_TOKEN"] == {"secret": "github-repo"}
    assert payload["env"]["HF_TOKEN"] == {"secret": "hf-token"}


def test_bootstrap_shell_command_inlines_script_not_missing_repo():
    repo = Path(__file__).resolve().parents[1]
    command = bootstrap_shell_command(repo)
    assert "apt-get" in command
    assert "git clone" in command
    assert "cd \"$HOME/openpi\" && bash gman/bootstrap.sh" not in command
    assert command.startswith("bash --noprofile --norc -s <<'OPENPI_BOOTSTRAP'\n")
    assert command.endswith("\nOPENPI_BOOTSTRAP")


def test_train_shell_command_cds_after_clone():
    command = train_shell_command(config_name="pi0_busybox_push_green_button", smoke=True)
    assert "SMOKE=1" in command
    assert "cd \"$HOME/openpi\"" in command
    assert "CONFIG_NAME=pi0_busybox_push_green_button" in command


def test_gman_train_uses_proven_xla_memory_fraction():
    script = (Path(__file__).resolve().parents[1] / "gman/train.sh").read_text()
    assert 'XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"' in script


def test_gman_production_train_publishes_latest_checkpoint_to_hub_root():
    script = (Path(__file__).resolve().parents[1] / "gman/train.sh").read_text()
    assert "scripts/gman_publish_latest.py" in script
    assert '--checkpoint-root="${CHECKPOINT_DIR}"' in script
    assert '--repo-id="${HF_REPO}"' in script


def test_experiment_name_separates_smoke_from_production():
    assert experiment_name("pi0_busybox_push_green_button", smoke=True) == "pi0_busybox_push_green_button_smoke"
    assert experiment_name("pi0_busybox_push_green_button", smoke=False) == "pi0_busybox_push_green_button"


def test_refuse_production_if_only_smoke_checkpoints_exist(tmp_path: Path):
    for step in ("5", "9"):
        (tmp_path / step / "params").mkdir(parents=True)
    with pytest.raises(SystemExit, match="smoke"):
        refuse_production_if_smoke_checkpoints(tmp_path)
    (tmp_path / "5000" / "params").mkdir(parents=True)
    refuse_production_if_smoke_checkpoints(tmp_path)


def test_require_jax_device_count_rejects_single_gpu():
    require_jax_device_count(8)
    with pytest.raises(SystemExit, match="Expected 8 JAX devices"):
        require_jax_device_count(1)


def test_create_node_rejects_cra_single_gpu_chip():
    assert "h100-1" in GMAN_FORBIDDEN_CHIPS
    assert "h100-1" in GMAN_NODE_REQUIREMENT
    with pytest.raises(SystemExit, match="h100-1"):
        assert_not_forbidden_chip("h100-1")
    assert_not_forbidden_chip("h100-8")


def test_cloned_repo_must_contain_busybox_assets(tmp_path: Path):
    with pytest.raises(SystemExit, match="Push the task branch"):
        assert_openpi_branch_has_busybox_assets(tmp_path)
    for relative_path in (
        "gman/train.sh",
        "gman/bootstrap.sh",
        "scripts/gman_payload.py",
        "scripts/train.py",
        "src/openpi/models/pi0.py",
        "src/openpi/training/config.py",
    ):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# placeholder\n")
    with pytest.raises(SystemExit, match="pi0_busybox_push_green_button"):
        assert_openpi_branch_has_busybox_assets(tmp_path)
    (tmp_path / "src/openpi/training/config.py").write_text(
        'name="pi0_busybox_push_green_button"\n'
    )
    assert_openpi_branch_has_busybox_assets(tmp_path)


@pytest.mark.parametrize(
    "required_path",
    ("scripts/gman_payload.py", "scripts/train.py", "src/openpi/models/pi0.py"),
)
def test_cloned_repo_requires_runtime_fix_files(tmp_path: Path, required_path: str):
    paths = (
        "gman/train.sh",
        "gman/bootstrap.sh",
        "scripts/gman_payload.py",
        "scripts/train.py",
        "src/openpi/models/pi0.py",
        "src/openpi/training/config.py",
    )
    for relative_path in paths:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('name="pi0_busybox_push_green_button"\n')
    (tmp_path / required_path).unlink()
    with pytest.raises(SystemExit, match=required_path):
        assert_openpi_branch_has_busybox_assets(tmp_path)


def test_bootstrap_checks_branch_before_uv_sync():
    script = (Path(__file__).resolve().parents[1] / "gman/bootstrap.sh").read_text()
    assert script.index("branch assets") < script.index("uv sync")
    assert "python3 - <<'PY'" in script
