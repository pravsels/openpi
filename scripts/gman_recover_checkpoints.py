"""Prepare a failed GMAN checkpoint directory for a safe resume."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from gman_publish import prepare_resume_checkpoint
except ImportError:
    from scripts.gman_publish import prepare_resume_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    args = parser.parse_args()

    resume_step = prepare_resume_checkpoint(args.checkpoint_root)
    print(f"resume_step={resume_step}")


if __name__ == "__main__":
    main()
