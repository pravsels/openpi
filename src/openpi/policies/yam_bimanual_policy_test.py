"""Tests for bimanual YAM / SO-101 top+wrist transforms and RoboBuilderBench Pi0 configs."""

from __future__ import annotations

import numpy as np

from openpi.policies import so101_policy, yam_bimanual_policy
from openpi.training import config as _config
from openpi import transforms as _transforms


EXPECTED_RBB_YAM_CONFIGS: dict[str, str] = {
    "pi0_rbb_assemble_rail": "armnet/assemble_rail",
    "pi0_rbb_insert_actuator_into_camera_mount": "fanqi-robo/insert_actuator_into_camera_mount",
    "pi0_rbb_insert_gear_in_gripper": "fanqi-robo/insert_gear_in_gripper",
    "pi0_rbb_take_actuator_from_box": "armnet/take_actuator_from_box",
    "pi0_rbb_open_clamp": "armnet/open_clamp",
    "pi0_rbb_assemble_enpire_gripper": "armnet/assemble_enpire_gripper",
}

EXPECTED_RBB_SO101_CONFIGS: dict[str, str] = {
    "pi0_rbb_so101_track_rails_depth_2": "armnet/so101_track_rails_depth_2",
    "pi0_rbb_so101_insert_actuator_into_camera_mount": "armnet/so101_insert_actuator_into_camera_mount",
    "pi0_rbb_so101_insert_gear_in_gripper": "armnet/so101_insert_gear_in_gripper",
    "pi0_rbb_so101_take_actuator_from_box": "armnet/so101_take_actuator_from_box",
}


def test_bimanual_yam_inputs_and_outputs() -> None:
    inp = yam_bimanual_policy.BimanualYAMInputs(default_prompt="Open the clamp")
    out = yam_bimanual_policy.BimanualYAMOutputs()

    sample = {
        "observation.images.top": np.zeros((3, 64, 64), dtype=np.uint8),
        "observation.images.left_wrist": np.ones((3, 64, 64), dtype=np.uint8) * 2,
        "observation.images.right_wrist": np.ones((3, 64, 64), dtype=np.uint8) * 3,
        "observation.state": np.arange(14, dtype=np.float32),
        "actions": np.zeros((30, 14), dtype=np.float32),
    }

    parsed = inp(sample)
    assert parsed["state"].shape == (14,)
    assert parsed["actions"].shape == (30, 14)
    assert parsed["image"]["base_0_rgb"].shape == (64, 64, 3)
    assert parsed["image"]["left_wrist_0_rgb"].shape == (64, 64, 3)
    assert parsed["image"]["right_wrist_0_rgb"].shape == (64, 64, 3)
    assert all(bool(v) for v in parsed["image_mask"].values())
    assert parsed["prompt"] == "Open the clamp"

    sliced = out({"actions": np.zeros((30, 32), dtype=np.float32)})
    assert sliced["actions"].shape == (30, 14)


def test_so101_top_wrist_inputs() -> None:
    inp = so101_policy.SO101TopWristInputs(default_prompt="Insert gear into gripper")
    sample = {
        "observation.images.top": np.zeros((3, 64, 64), dtype=np.uint8),
        "observation.images.wrist": np.ones((3, 64, 64), dtype=np.uint8),
        "observation.state": np.arange(6, dtype=np.float32),
        "actions": np.zeros((30, 6), dtype=np.float32),
    }

    parsed = inp(sample)
    assert parsed["state"].shape == (6,)
    assert parsed["image"]["base_0_rgb"].shape == (64, 64, 3)
    assert parsed["image"]["left_wrist_0_rgb"].shape == (64, 64, 3)
    assert parsed["image"]["right_wrist_0_rgb"].shape == (64, 64, 3)
    assert bool(parsed["image_mask"]["base_0_rgb"]) is True
    assert bool(parsed["image_mask"]["left_wrist_0_rgb"]) is True
    assert bool(parsed["image_mask"]["right_wrist_0_rgb"]) is False
    assert parsed["prompt"] == "Insert gear into gripper"


def test_all_ten_rbb_pi0_configs_registered() -> None:
    for cfg_name, expected_repo in EXPECTED_RBB_YAM_CONFIGS.items():
        cfg = _config.get_config(cfg_name)
        assert cfg.model.pi05 is False
        assert cfg.model.action_horizon == 30
        assert cfg.data.repo_id == expected_repo
        assert cfg.data.video_backend == "pyav"
        assert cfg.data.use_delta_actions is True
        assert cfg.num_train_steps == 20_000
        assert cfg.batch_size == 32
        assert cfg.weight_loader.params_path == "weights/pi0_base/params"

        data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
        assert data_cfg.video_backend == "pyav"
        delta_tfs = [tf for tf in data_cfg.data_transforms.inputs if isinstance(tf, _transforms.DeltaActions)]
        assert len(delta_tfs) == 1
        assert tuple(delta_tfs[0].mask) == tuple([True] * 6 + [False] + [True] * 6 + [False])

    for cfg_name, expected_repo in EXPECTED_RBB_SO101_CONFIGS.items():
        cfg = _config.get_config(cfg_name)
        assert cfg.model.pi05 is False
        assert cfg.model.action_horizon == 30
        assert cfg.data.repo_id == expected_repo
        assert cfg.data.video_backend == "pyav"
        assert cfg.data.use_delta_actions is True
        assert cfg.num_train_steps == 20_000
        assert cfg.batch_size == 32
        assert cfg.weight_loader.params_path == "weights/pi0_base/params"

        data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
        assert data_cfg.video_backend == "pyav"
        delta_tfs = [tf for tf in data_cfg.data_transforms.inputs if isinstance(tf, _transforms.DeltaActions)]
        assert len(delta_tfs) == 1
        assert tuple(delta_tfs[0].mask) == tuple([True] * 5 + [False])
