"""Continuously publish the latest finalized GMAN checkpoint to a Hub repo root."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import signal
import time

from huggingface_hub import HfApi

try:
    from gman_publish import finalized_checkpoint_steps
    from gman_publish import publish_checkpoint_root
    from gman_publish import publish_checkpoint_subfolder
except ImportError:
    from scripts.gman_publish import finalized_checkpoint_steps
    from scripts.gman_publish import publish_checkpoint_root
    from scripts.gman_publish import publish_checkpoint_subfolder


def exit_on_signal(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


def watch_checkpoints(
    *,
    checkpoint_root: Path,
    repo_id: str,
    config_name: str,
    done_file: Path,
    poll_seconds: float,
    layout: str = "root",
) -> None:
    """Publish finalized checkpoints to the Hub as they appear.

    ``layout="root"`` (default, unchanged behaviour) keeps only the latest
    checkpoint at the repo root. ``layout="subfolder"`` publishes every
    finalized step to its own ``checkpoint-<step>/`` and keeps them all.
    """
    api = HfApi()
    uploaded_steps: set[int] = set()
    final_failures = 0

    while True:
        steps = finalized_checkpoint_steps(checkpoint_root)
        pending = steps if layout == "subfolder" else ([steps[-1]] if steps else [])
        pending = [step for step in pending if step not in uploaded_steps]
        if pending:
            try:
                for step in pending:
                    if layout == "subfolder":
                        publish_checkpoint_subfolder(
                            api,
                            repo_id=repo_id,
                            checkpoint_dir=checkpoint_root / str(step),
                            config_name=config_name,
                            step=step,
                        )
                        logging.info("Published checkpoint step %d to %s (checkpoint-%d/)", step, repo_id, step)
                    else:
                        publish_checkpoint_root(
                            api,
                            repo_id=repo_id,
                            checkpoint_dir=checkpoint_root / str(step),
                            config_name=config_name,
                            step=step,
                        )
                        logging.info("Published checkpoint step %d to %s root", step, repo_id)
                    uploaded_steps.add(step)
                final_failures = 0
            except Exception:
                logging.exception("Failed to publish checkpoints")
                if done_file.exists():
                    final_failures += 1
                    if final_failures >= 5:
                        raise

        if layout == "subfolder":
            remaining = [step for step in finalized_checkpoint_steps(checkpoint_root) if step not in uploaded_steps]
            done = not remaining
        else:
            steps = finalized_checkpoint_steps(checkpoint_root)
            done = not steps or steps[-1] in uploaded_steps
        if done_file.exists() and done:
            return
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--done-file", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument(
        "--layout",
        choices=("root", "subfolder"),
        default="root",
        help="root (default): latest checkpoint at repo root. subfolder: keep every step as checkpoint-<step>/.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    signal.signal(signal.SIGINT, exit_on_signal)
    signal.signal(signal.SIGTERM, exit_on_signal)
    watch_checkpoints(
        checkpoint_root=args.checkpoint_root,
        repo_id=args.repo_id,
        config_name=args.config_name,
        done_file=args.done_file,
        poll_seconds=args.poll_seconds,
        layout=args.layout,
    )


if __name__ == "__main__":
    main()
