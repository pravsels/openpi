"""Publish OpenPI orbax checkpoints to Hugging Face."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any
from typing import Protocol
from typing import Sequence


class HfPublishApi(Protocol):
    def create_repo(self, repo_id: str, *, repo_type: str, exist_ok: bool) -> Any: ...

    def upload_folder(
        self,
        *,
        folder_path: str,
        repo_id: str,
        path_in_repo: str,
        repo_type: str,
        ignore_patterns: list[str] | None = None,
        delete_patterns: list[str] | None = None,
        commit_message: str | None = None,
    ) -> Any: ...

    def list_repo_files(self, repo_id: str, *, repo_type: str = "model") -> list[str]: ...

    def repo_exists(self, repo_id: str, *, repo_type: str = "model") -> bool: ...

    def delete_folder(self, *, repo_id: str, path_in_repo: str, repo_type: str) -> Any: ...


def checkpoint_has_params(checkpoint_dir: Path) -> bool:
    params = Path(checkpoint_dir) / "params"
    return params.is_dir() or params.is_file()


def finalized_checkpoint_steps(checkpoint_root: Path) -> list[int]:
    root = Path(checkpoint_root)
    if not root.is_dir():
        return []
    return sorted(
        int(path.name)
        for path in root.iterdir()
        if path.is_dir() and path.name.isdigit() and checkpoint_has_params(path)
    )


def publish_checkpoint_root(
    api: HfPublishApi,
    *,
    repo_id: str,
    checkpoint_dir: Path,
    config_name: str,
    step: int,
) -> None:
    """Replace inference checkpoint files at the model repo root."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir() or not checkpoint_has_params(checkpoint_dir):
        raise ValueError(f"Checkpoint step {step} is incomplete: {checkpoint_dir}")

    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    preserved = {".gitattributes", "README.md"}
    delete_patterns = [
        path for path in api.list_repo_files(repo_id, repo_type="model") if path not in preserved
    ]
    api.upload_folder(
        folder_path=str(checkpoint_dir),
        repo_id=repo_id,
        path_in_repo=".",
        repo_type="model",
        ignore_patterns=["train_state/**"],
        delete_patterns=delete_patterns,
        commit_message=f"Update {config_name} checkpoint to step {step}",
    )


def publish_checkpoint_steps(
    api: HfPublishApi,
    *,
    repo_id: str,
    checkpoint_root: Path,
    steps: Sequence[str],
    config_name: str,
) -> list[str]:
    """Upload selected step dirs. Skips missing steps; errors if none uploaded."""
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    uploaded: list[str] = []
    for step in steps:
        folder = Path(checkpoint_root) / str(step)
        if not folder.is_dir() or not checkpoint_has_params(folder):
            continue
        api.upload_folder(
            folder_path=str(folder),
            repo_id=repo_id,
            path_in_repo=f"step_{step}",
            repo_type="model",
            ignore_patterns=["train_state/**"],
            commit_message=f"Add {config_name} checkpoint step {step}",
        )
        uploaded.append(str(step))
    if not uploaded:
        raise SystemExit("ERROR: no checkpoints uploaded (none of the requested steps were found)")
    return uploaded


def assert_hub_steps_exist(api: HfPublishApi, repo_id: str, steps: Sequence[str]) -> None:
    files = api.list_repo_files(repo_id, repo_type="model")
    missing: list[str] = []
    for step in steps:
        prefix = f"step_{step}/params"
        if not any(path == prefix or path.startswith(prefix + "/") for path in files):
            missing.append(prefix)
    if missing:
        raise SystemExit(f"ERROR: Hub repo {repo_id} missing {', '.join(missing)}")


def assert_wandb_history_logged(history_rows: Sequence[dict[str, Any]]) -> None:
    """Fail unless the run logged at least one training step after the camera dump."""
    rows = list(history_rows)
    if not rows:
        raise SystemExit("ERROR: W&B run has no logged history")
    if not any(int(row.get("step") or row.get("_step") or 0) > 0 for row in rows):
        raise SystemExit("ERROR: W&B run never logged a step after 0")


def delete_local_checkpoint_steps(checkpoint_root: Path, steps: Sequence[str]) -> list[str]:
    deleted: list[str] = []
    for step in steps:
        folder = Path(checkpoint_root) / str(step)
        if folder.is_dir():
            shutil.rmtree(folder)
            deleted.append(str(step))
    return deleted


def delete_hub_step_folders(api: HfPublishApi, repo_id: str, steps: Sequence[str]) -> list[str]:
    if not api.repo_exists(repo_id, repo_type="model"):
        return []
    files = api.list_repo_files(repo_id, repo_type="model")
    deleted: list[str] = []
    for step in steps:
        prefix = f"step_{step}"
        if not any(path == prefix or path.startswith(prefix + "/") for path in files):
            continue
        api.delete_folder(repo_id=repo_id, path_in_repo=prefix, repo_type="model")
        deleted.append(str(step))
    return deleted
