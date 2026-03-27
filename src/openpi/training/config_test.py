import numpy as np

from openpi.models import pi0_config
from openpi.shared import normalize as _normalize
from openpi.training import config as _config
import openpi.transforms as _transforms


def test_loads_per_timestep_action_stats(tmp_path):
    stats = _normalize.NormStats(
        mean=np.zeros((2, 17)),
        std=np.ones((2, 17)),
        q01=np.zeros((2, 17)),
        q99=np.ones((2, 17)) * 2.0,
    )
    asset_dir = tmp_path / "asset"
    _normalize.save_actions_per_timestep(asset_dir, stats)

    assets = _config.AssetsConfig(assets_dir=str(tmp_path), asset_id="asset")
    factory = _config.LeRobotBinPackDataConfig(repo_id="repo", assets=assets)
    model_config = pi0_config.Pi0Config(action_dim=7, action_horizon=2)
    data_config = factory.create(tmp_path, model_config)

    loaded = data_config.per_timestep_action_norm_stats
    assert loaded is not None
    assert np.allclose(loaded.mean, stats.mean)
    assert np.allclose(loaded.std, stats.std)
    # rot6d dims (10:16) are patched to identity normalization by the config
    expected_q01 = np.array(stats.q01, copy=True)
    expected_q01[..., 10:16] = -1.0
    expected_q99 = np.array(stats.q99, copy=True)
    expected_q99[..., 10:16] = 1.0
    assert np.allclose(loaded.q01, expected_q01)
    assert np.allclose(loaded.q99, expected_q99)


def test_auto_enable_per_timestep_for_binpack_delta(tmp_path):
    base = _config.DataConfig(use_per_timestep_action_norm=None)
    factory = _config.LeRobotBinPackDataConfig(repo_id="repo", base_config=base, use_delta_actions=True)
    model_config = pi0_config.Pi0Config(action_dim=7, action_horizon=2)
    data_config = factory.create(tmp_path, model_config)
    assert data_config.use_per_timestep_action_norm is True


def test_auto_enable_per_timestep_for_aloha_delta(tmp_path):
    base = _config.DataConfig(use_per_timestep_action_norm=None)
    factory = _config.LeRobotAlohaDataConfig(repo_id="repo", base_config=base, use_delta_joint_actions=True)
    model_config = pi0_config.Pi0Config(action_dim=7, action_horizon=2)
    data_config = factory.create(tmp_path, model_config)
    assert data_config.use_per_timestep_action_norm is True


def test_auto_enable_per_timestep_for_libero_extra_delta(tmp_path):
    base = _config.DataConfig(use_per_timestep_action_norm=None)
    factory = _config.LeRobotLiberoDataConfig(repo_id="repo", base_config=base, extra_delta_transform=True)
    model_config = pi0_config.Pi0Config(action_dim=7, action_horizon=2)
    data_config = factory.create(tmp_path, model_config)
    assert data_config.use_per_timestep_action_norm is True


def test_explicit_disable_overrides_auto_enable(tmp_path):
    base = _config.DataConfig(use_per_timestep_action_norm=False)
    factory = _config.LeRobotBinPackDataConfig(repo_id="repo", base_config=base, use_delta_actions=True)
    model_config = pi0_config.Pi0Config(action_dim=7, action_horizon=2)
    data_config = factory.create(tmp_path, model_config)
    assert data_config.use_per_timestep_action_norm is False


def test_binpack_config_can_enable_control_mode_advantage_prompt(tmp_path):
    factory = _config.LeRobotBinPackDataConfig(
        repo_id="repo",
        use_control_mode_advantage_prompt=True,
    )
    model_config = pi0_config.Pi0Config(pi05=True, action_horizon=2)
    data_config = factory.create(tmp_path, model_config)

    assert any(isinstance(t, _transforms.InjectAdvantagePrompt) for t in data_config.data_transforms.inputs)


def test_binpack_advantage_prompt_runs_before_binpack_inputs(tmp_path):
    factory = _config.LeRobotBinPackDataConfig(
        repo_id="repo",
        use_control_mode_advantage_prompt=True,
    )
    model_config = pi0_config.Pi0Config(pi05=True, action_horizon=2)
    data_config = factory.create(tmp_path, model_config)

    assert isinstance(data_config.data_transforms.inputs[0], _transforms.InjectAdvantagePrompt)


def test_binpack_config_can_select_positive_only_advantage_mode(tmp_path):
    factory = _config.LeRobotBinPackDataConfig(
        repo_id="repo",
        use_control_mode_advantage_prompt=True,
        advantage_prompt_mode="positive_only",
    )
    model_config = pi0_config.Pi0Config(pi05=True, action_horizon=2)
    data_config = factory.create(tmp_path, model_config)

    transform = data_config.data_transforms.inputs[0]
    assert isinstance(transform, _transforms.InjectAdvantagePrompt)
    assert transform.mode == "positive_only"


def test_reward_recap_binpack_configs_exist():
    positive_only = _config.get_config("pi05_bin_pack_coffee_capsules_reward_recap_positive_only")
    mixed = _config.get_config("pi05_bin_pack_coffee_capsules_reward_recap_mixed")
    positive_only_base = _config.get_config("pi05_bin_pack_coffee_capsules_reward_recap_positive_only_from_base")
    mixed_base = _config.get_config("pi05_bin_pack_coffee_capsules_reward_recap_mixed_from_base")

    for cfg in [positive_only, positive_only_base]:
        assert cfg.data.use_control_mode_advantage_prompt is True
        assert cfg.data.advantage_prompt_mode == "positive_only"
    for cfg in [mixed, mixed_base]:
        assert cfg.data.use_control_mode_advantage_prompt is True
        assert cfg.data.advantage_prompt_mode == "mixed"
