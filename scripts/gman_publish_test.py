from __future__ import annotations

from pathlib import Path

import pytest

from scripts import gman_publish
from scripts.gman_publish import assert_hub_steps_exist
from scripts.gman_publish import assert_wandb_history_logged
from scripts.gman_publish import delete_hub_step_folders
from scripts.gman_publish import delete_local_checkpoint_steps
from scripts.gman_publish import publish_checkpoint_steps


class FakeHf:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.uploads: list[dict[str, object]] = []
        self.files: list[str] = []
        self.exists = True

    def create_repo(self, repo_id: str, *, repo_type: str, exist_ok: bool) -> None:
        self.created.append(repo_id)

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
    ) -> None:
        self.uploads.append(
            {
                "folder_path": folder_path,
                "repo_id": repo_id,
                "path_in_repo": path_in_repo,
                "ignore_patterns": ignore_patterns,
                "delete_patterns": delete_patterns,
                "commit_message": commit_message,
            }
        )
        self.files.append(f"{path_in_repo}/params/array")

    def list_repo_files(self, repo_id: str, *, repo_type: str = "model") -> list[str]:
        return list(self.files)

    def repo_exists(self, repo_id: str, *, repo_type: str = "model") -> bool:
        return self.exists

    def delete_folder(self, *, repo_id: str, path_in_repo: str, repo_type: str) -> None:
        self.files = [path for path in self.files if not path.startswith(path_in_repo.rstrip("/") + "/")]
        self.uploads.append({"deleted": path_in_repo, "repo_id": repo_id})


def _write_step(root: Path, step: str) -> None:
    params = root / step / "params"
    params.mkdir(parents=True)
    (params / "marker").write_text("ok")


def test_publish_skips_missing_steps_and_uploads_params(tmp_path: Path):
    _write_step(tmp_path, "5")
    api = FakeHf()
    uploaded = publish_checkpoint_steps(
        api,
        repo_id="pravsels/pi0_busybox_push_green_button",
        checkpoint_root=tmp_path,
        steps=("5", "9"),
        config_name="pi0_busybox_push_green_button",
    )
    assert uploaded == ["5"]
    assert api.created == ["pravsels/pi0_busybox_push_green_button"]
    assert api.uploads[0]["path_in_repo"] == "step_5"
    assert api.uploads[0]["ignore_patterns"] == ["train_state/**"]


def test_publish_errors_when_nothing_uploaded(tmp_path: Path):
    api = FakeHf()
    with pytest.raises(SystemExit, match="no checkpoints uploaded"):
        publish_checkpoint_steps(
            api,
            repo_id="pravsels/pi0_busybox_push_green_button",
            checkpoint_root=tmp_path,
            steps=("5", "9"),
            config_name="pi0_busybox_push_green_button",
        )


def test_publish_checkpoint_root_replaces_model_files_but_preserves_hub_metadata(tmp_path: Path):
    _write_step(tmp_path, "10000")
    api = FakeHf()
    api.files = [".gitattributes", "README.md", "params/old-chunk", "assets/old.json"]

    gman_publish.publish_checkpoint_root(
        api,
        repo_id="pravsels/pi0_busybox_push_green_button",
        checkpoint_dir=tmp_path / "10000",
        config_name="pi0_busybox_push_green_button",
        step=10_000,
    )

    assert api.uploads[0] == {
        "folder_path": str(tmp_path / "10000"),
        "repo_id": "pravsels/pi0_busybox_push_green_button",
        "path_in_repo": ".",
        "ignore_patterns": ["train_state/**"],
        "delete_patterns": ["params/old-chunk", "assets/old.json"],
        "commit_message": "Update pi0_busybox_push_green_button checkpoint to step 10000",
    }


def test_finalized_checkpoint_steps_ignores_temporary_and_incomplete_dirs(tmp_path: Path):
    _write_step(tmp_path, "5000")
    _write_step(tmp_path, "10000")
    _write_step(tmp_path, "15000.orbax-checkpoint-tmp-1")
    (tmp_path / "20000").mkdir()

    assert gman_publish.finalized_checkpoint_steps(tmp_path) == [5_000, 10_000]


def test_assert_hub_steps_exist_requires_params():
    api = FakeHf()
    api.files = ["step_5/params/array"]
    with pytest.raises(SystemExit, match="step_9/params"):
        assert_hub_steps_exist(api, "pravsels/pi0_busybox_push_green_button", ("5", "9"))
    api.files.append("step_9/params/array")
    assert_hub_steps_exist(api, "pravsels/pi0_busybox_push_green_button", ("5", "9"))


def test_delete_local_smoke_checkpoint_steps(tmp_path: Path):
    _write_step(tmp_path, "5")
    _write_step(tmp_path, "9")
    _write_step(tmp_path, "5000")
    deleted = delete_local_checkpoint_steps(tmp_path, ("5", "9"))
    assert deleted == ["5", "9"]
    assert not (tmp_path / "5").exists()
    assert (tmp_path / "5000" / "params" / "marker").exists()


def test_delete_hub_smoke_step_folders():
    api = FakeHf()
    api.files = ["step_5/params/array", "step_5000/params/array"]
    deleted = delete_hub_step_folders(api, "pravsels/pi0_busybox_push_green_button", ("5", "9"))
    assert deleted == ["5"]
    assert api.files == ["step_5000/params/array"]
    assert delete_hub_step_folders(
        api, "pravsels/pi0_busybox_push_green_button", ("5", "9")
    ) == []


def test_delete_hub_smoke_steps_skips_missing_repo():
    api = FakeHf()
    api.exists = False
    assert delete_hub_step_folders(
        api, "pravsels/pi0_busybox_push_green_button", ("5", "9")
    ) == []
    assert api.uploads == []


def test_assert_wandb_history_logged_requires_nonzero_step():
    with pytest.raises(SystemExit, match="no logged history"):
        assert_wandb_history_logged([])
    with pytest.raises(SystemExit, match="never logged a step after 0"):
        assert_wandb_history_logged([{"_step": 0}])
    assert_wandb_history_logged([{"_step": 0}, {"_step": 1, "loss": 0.5}])
