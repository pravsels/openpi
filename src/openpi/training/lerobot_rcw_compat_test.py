import sys


def test_ensure_aliases_makes_rcw_factory_importable_on_lerobot_04():
    # RCW main imports lerobot.datasets.feature_utils at factory load. LeRobot
    # v0.4.3 does not ship that module. After the OpenPI aliases, factory import
    # must succeed so BusyBox v3 can load without bumping LeRobot.
    sys.modules.pop("lerobot.datasets.feature_utils", None)
    sys.modules.pop("lerobot.datasets.io_utils", None)
    for name in list(sys.modules):
        if name == "robocandywrapper" or name.startswith("robocandywrapper."):
            sys.modules.pop(name)

    from openpi.training.lerobot_rcw_compat import ensure_lerobot_04_module_aliases

    ensure_lerobot_04_module_aliases()
    from robocandywrapper.factory import make_dataset_without_config

    assert callable(make_dataset_without_config)
