"""Continuously publish the latest finalized GMAN checkpoint to a Hub repo root."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time

from huggingface_hub import HfApi

from scripts.gman_publish import finalized_checkpoint_steps
from scripts.gman_publish import publish_checkpoint_root


def watch_checkpoints(
    *,
    checkpoint_root: Path,
    repo_id: str,
    config_name: str,
    done_file: Path,
    poll_seconds: float,
) -> None:
    api = HfApi()
    uploaded_step: int | None = None
    final_failures = 0

    while True:
        steps = finalized_checkpoint_steps(checkpoint_root)
        latest_step = steps[-1] if steps else None
        if latest_step is not None and latest_step != uploaded_step:
            try:
                publish_checkpoint_root(
                    api,
                    repo_id=repo_id,
                    checkpoint_dir=checkpoint_root / str(latest_step),
                    config_name=config_name,
                    step=latest_step,
                )
                uploaded_step = latest_step
                final_failures = 0
                logging.info("Published checkpoint step %d to %s root", latest_step, repo_id)
            except Exception:
                logging.exception("Failed to publish checkpoint step %d", latest_step)
                if done_file.exists():
                    final_failures += 1
                    if final_failures >= 5:
                        raise

        if done_file.exists() and (latest_step is None or latest_step == uploaded_step):
            return
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--done-file", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    watch_checkpoints(
        checkpoint_root=args.checkpoint_root,
        repo_id=args.repo_id,
        config_name=args.config_name,
        done_file=args.done_file,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    main()
