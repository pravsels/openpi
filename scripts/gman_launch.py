#!/usr/bin/env python3
"""Print CRA-style GMAN command payloads (typed secret refs).

create_node must be 8x >=80GB H100. Do not use chip=h100-1 (CRA 1-GPU).
Commit and push task/train_pi_policies_green_button before bootstrap.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from gman_payload import GMAN_NODE_REQUIREMENT
    from gman_payload import assert_not_forbidden_chip
    from gman_payload import bootstrap_request_payload
    from gman_payload import bootstrap_shell_command
    from gman_payload import delete_smoke_shell_command
    from gman_payload import train_shell_command
    from gman_payload import training_request_payload
except ImportError:
    from scripts.gman_payload import GMAN_NODE_REQUIREMENT
    from scripts.gman_payload import assert_not_forbidden_chip
    from scripts.gman_payload import bootstrap_request_payload
    from scripts.gman_payload import bootstrap_shell_command
    from scripts.gman_payload import delete_smoke_shell_command
    from scripts.gman_payload import train_shell_command
    from scripts.gman_payload import training_request_payload


_REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print GMAN command JSON. " + GMAN_NODE_REQUIREMENT,
    )
    parser.add_argument("kind", choices=("bootstrap", "smoke", "train", "delete-smoke"))
    parser.add_argument("--mission", required=True)
    parser.add_argument(
        "--config-name",
        default="pi0_busybox_push_green_button",
        choices=("pi0_busybox_push_green_button", "pi05_busybox_push_green_button"),
    )
    parser.add_argument(
        "--chip",
        default="",
        help="Optional create_node chip id. Rejects h100-1 (1-GPU CRA SKU).",
    )
    args = parser.parse_args()
    if args.chip:
        assert_not_forbidden_chip(args.chip)

    print(GMAN_NODE_REQUIREMENT, file=sys.stderr)

    if args.kind == "bootstrap":
        payload = bootstrap_request_payload(
            command=bootstrap_shell_command(_REPO),
            mission=args.mission,
        )
    elif args.kind == "delete-smoke":
        payload = training_request_payload(
            command=delete_smoke_shell_command(config_name=args.config_name),
            mission=args.mission,
        )
    else:
        payload = training_request_payload(
            command=train_shell_command(config_name=args.config_name, smoke=args.kind == "smoke"),
            mission=args.mission,
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
