import numpy as np
import pathlib

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


def test_binpack_config_passes_advantage_dropout_rate(tmp_path):
    factory = _config.LeRobotBinPackDataConfig(
        repo_id="repo",
        use_control_mode_advantage_prompt=True,
        advantage_dropout_rate=0.3,
    )
    model_config = pi0_config.Pi0Config(pi05=True, action_horizon=2)
    data_config = factory.create(tmp_path, model_config)

    transform = data_config.data_transforms.inputs[0]
    assert isinstance(transform, _transforms.InjectAdvantagePrompt)
    assert transform.dropout_rate == 0.3


def test_reward_recap_binpack_configs_exist():
    positive_only = _config.get_config("pi05_bin_pack_coffee_capsules_reward_recap_positive_only")
    mixed = _config.get_config("pi05_bin_pack_coffee_capsules_reward_recap_mixed")

    for cfg in [positive_only]:
        assert cfg.data.use_control_mode_advantage_prompt is True
        assert cfg.data.advantage_prompt_mode == "positive_only"
        assert cfg.data.advantage_dropout_rate == 0.3
    for cfg in [mixed]:
        assert cfg.data.use_control_mode_advantage_prompt is True
        assert cfg.data.advantage_prompt_mode == "mixed"
        assert cfg.data.advantage_dropout_rate == 0.3


def test_reward_recap_block_tower_configs_exist():
    positive_only = _config.get_config("pi05_build_block_tower_positive_only")
    mixed = _config.get_config("pi05_build_block_tower_mixed")

    assert positive_only.data.use_control_mode_advantage_prompt is True
    assert positive_only.data.advantage_prompt_mode == "positive_only"
    assert positive_only.data.advantage_dropout_rate == 0.3

    assert mixed.data.use_control_mode_advantage_prompt is True
    assert mixed.data.advantage_prompt_mode == "mixed"
    assert mixed.data.advantage_dropout_rate == 0.3


def test_build_block_tower_baseline_uses_base_and_dagger_datasets():
    baseline = _config.get_config("pi05_build_block_tower_baseline")

    for repo_id in (
        "villekuosmanen/build_block_tower",
        "villekuosmanen/dAgger_build_block_tower_1.0.0",
        "villekuosmanen/dAgger_build_block_tower_1.1.0",
        "villekuosmanen/dAgger_build_block_tower_1.2.0",
        "villekuosmanen/dAgger_build_block_tower_1.3.0",
        "villekuosmanen/dAgger_build_block_tower_1.4.0",
    ):
        assert repo_id in baseline.data.repo_id


def test_build_block_tower_rlt_references_latest_baseline_step():
    baseline = _config.get_config("pi05_build_block_tower_baseline")
    rlt = _config.get_config("pi05_rl_token_build_block_tower")
    expected_step = str(baseline.num_train_steps)

    assert expected_step == "50000"
    assert isinstance(rlt.weight_loader, _config.weight_loaders.RLTokenCheckpointWeightLoader)
    assert f"/{expected_step}/params" in rlt.weight_loader.params_path

    script = pathlib.Path("slurm/train_build_block_tower_rlt_slurm.sh").read_text()
    assert f'BASELINE_STEP="{expected_step}"' in script


def test_reward_recap_slurm_script_references_existing_configs():
    script = pathlib.Path("slurm/train_bin_pack_reward_recap_slurm.sh").read_text()

    for config_name in (
        "pi05_bin_pack_coffee_capsules_recap_positive_only",
        "pi05_bin_pack_coffee_capsules_recap_mixed",
    ):
        assert config_name in script
        _config.get_config(config_name)
