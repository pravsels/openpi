"""Fail closed unless RCW is git main and BusyBox prompts are remapped.

PyPI robocandywrapper and git main both report 0.2.18. Only git main
(`de4e4eb` remap, lock 597aa9ad) maps sample task_index onto sorted
meta.tasks. Without that, prompt_from_task trains on shuffled language.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

EXPECTED_RCW_SHA = "597aa9ad21176e7f7dcee4aede5dc1ffc07eacee"


def rcw_direct_url() -> dict:
    dist = importlib.metadata.distribution("robocandywrapper")
    direct_path = None
    for file in dist.files or []:
        if Path(str(file)).name == "direct_url.json":
            direct_path = Path(dist.locate_file(file))
            break
    if direct_path is None:
        locate_root = Path(dist.locate_file("."))
        hits = list(locate_root.glob("direct_url.json"))
        if hits:
            direct_path = hits[0]
    if direct_path is None or not direct_path.is_file():
        raise SystemExit("refusing to train on PyPI RCW: missing direct_url.json")
    return json.loads(direct_path.read_text())


def assert_rcw_git_main(expected_sha: str = EXPECTED_RCW_SHA) -> str:
    data = rcw_direct_url()
    url = str(data.get("url") or "")
    commit = str((data.get("vcs_info") or {}).get("commit_id") or "")
    if "github.com" not in url or "RoboCandyWrapper" not in url:
        raise SystemExit(f"refusing to train on PyPI RCW: url={url!r}")
    if not commit.startswith(expected_sha):
        raise SystemExit(
            f"refusing to train on PyPI RCW: commit {commit or 'missing'} "
            f"!= {expected_sha}"
        )
    print("rcw_sha_ok", commit)
    return commit


def assert_prompt_ok() -> None:
    from openpi.training.data_loader import _coerce_task_mapping
    from openpi.training.lerobot_rcw_compat import make_dataset_without_config
    from openpi.transforms import PromptFromLeRobotTask

    ds = make_dataset_without_config("villekuosmanen/busybox_multitask", load_videos=False)
    for inner_ds in ds._datasets:
        inner_ds._query_videos = lambda *a, **k: {}
    inner = _coerce_task_mapping(ds._datasets[0].meta.tasks)
    wrapped = _coerce_task_mapping(ds.meta.tasks)
    xf = PromptFromLeRobotTask(wrapped)
    mismatches = []
    for i in range(0, len(ds), max(1, len(ds) // 80)):
        item = ds[i]
        prompt = xf(dict(item))["prompt"]
        name = item.get("task", wrapped[int(item["task_index"])])
        if hasattr(name, "item"):
            name = name.item()
        if isinstance(name, bytes):
            name = name.decode()
        if prompt != name:
            mismatches.append((i, prompt, name))
    print("wrapped_tasks", len(wrapped), "inner_tasks", len(inner))
    print("mismatches", len(mismatches))
    if mismatches:
        raise SystemExit(f"prompt remap failed: {mismatches[:5]}")
    if not wrapped:
        raise SystemExit("wrapped task table is empty")
    index0 = wrapped[0]
    print("index0", index0)
    if index0 != "Move the left slider to position 1":
        raise SystemExit(f"wrapped index 0 is {index0!r}, expected slider sentence")
    print("prompt_ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-prompt-check", action="store_true")
    args = parser.parse_args()
    expected = os.environ.get("EXPECTED_RCW_SHA", EXPECTED_RCW_SHA)
    assert_rcw_git_main(expected)
    if not args.skip_prompt_check:
        assert_prompt_ok()


if __name__ == "__main__":
    main()
