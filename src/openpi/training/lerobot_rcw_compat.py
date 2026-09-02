"""Make RCW main importable on LeRobot v0.4.3.

RCW `factory.py` imports the v2.1 dataset path at module load, which pulls
`lerobot.datasets.feature_utils` and `lerobot.datasets.io_utils`. Those names
exist in LeRobot 0.5; on v0.4.3 the same helpers live in
`lerobot.datasets.utils`. BusyBox is LeRobot v3 and never constructs the v2.1
class — we only need the import to succeed.

Import RCW symbols from this module so isort cannot hoist
`robocandywrapper` above the aliases.
"""

from __future__ import annotations

import sys

import lerobot.datasets.utils as lerobot_dataset_utils


def ensure_lerobot_04_module_aliases() -> None:
    sys.modules.setdefault("lerobot.datasets.feature_utils", lerobot_dataset_utils)
    sys.modules.setdefault("lerobot.datasets.io_utils", lerobot_dataset_utils)


ensure_lerobot_04_module_aliases()

# isort: off
from robocandywrapper.factory import make_dataset_without_config  # noqa: E402
from robocandywrapper.plugins import ControlModePlugin, EpisodeOutcomePlugin  # noqa: E402
from robocandywrapper.plugins.subtask import SubtaskPlugin  # noqa: E402

# isort: on

__all__ = [
    "ControlModePlugin",
    "EpisodeOutcomePlugin",
    "SubtaskPlugin",
    "ensure_lerobot_04_module_aliases",
    "make_dataset_without_config",
]
