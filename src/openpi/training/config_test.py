import numpy as np

from openpi.models import pi0_config
from openpi.shared import normalize as _normalize
from openpi.training import config as _config


def test_loads_per_timestep_action_stats(tmp_path):
    stats = _normalize.NormStats(
        mean=np.zeros((2, 3)),
        std=np.ones((2, 3)),
        q01=np.zeros((2, 3)),
        q99=np.ones((2, 3)) * 2.0,
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
    assert np.allclose(loaded.q01, stats.q01)
    assert np.allclose(loaded.q99, stats.q99)


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
