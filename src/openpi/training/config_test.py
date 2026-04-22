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


def test_block_tower_rot6d_stats_match_binpack_quantile_identity(tmp_path):
    stats = _normalize.NormStats(
        mean=np.full((17,), 3.0),
        std=np.full((17,), 4.0),
        q01=np.full((17,), -5.0),
        q99=np.full((17,), 6.0),
    )
    per_timestep = _normalize.NormStats(
        mean=np.full((2, 17), 7.0),
        std=np.full((2, 17), 8.0),
        q01=np.full((2, 17), -9.0),
        q99=np.full((2, 17), 10.0),
    )
    asset_dir = tmp_path / "asset"
    _normalize.save(asset_dir, {"state": stats, "actions": stats})
    _normalize.save_actions_per_timestep(asset_dir, per_timestep)

    assets = _config.AssetsConfig(assets_dir=str(tmp_path), asset_id="asset")
    factory = _config.LeRobotBlockTowerDataConfig(repo_id="repo", assets=assets)
    model_config = pi0_config.Pi0Config(action_dim=7, action_horizon=2)
    data_config = factory.create(tmp_path, model_config)

    assert data_config.norm_stats is not None
    for key in ("state", "actions"):
        loaded = data_config.norm_stats[key]
        expected_mean = np.array(stats.mean, copy=True)
        expected_std = np.array(stats.std, copy=True)
        expected_q01 = np.array(stats.q01, copy=True)
        expected_q99 = np.array(stats.q99, copy=True)
        expected_mean[..., 10:16] = 0.0
        expected_std[..., 10:16] = 1.0
        expected_q01[..., 10:16] = -1.0
        expected_q99[..., 10:16] = 1.0
        assert np.allclose(loaded.mean, expected_mean)
        assert np.allclose(loaded.std, expected_std)
        assert np.allclose(loaded.q01, expected_q01)
        assert np.allclose(loaded.q99, expected_q99)

    loaded_per_timestep = data_config.per_timestep_action_norm_stats
    assert loaded_per_timestep is not None
    expected_mean = np.array(per_timestep.mean, copy=True)
    expected_std = np.array(per_timestep.std, copy=True)
    expected_q01 = np.array(per_timestep.q01, copy=True)
    expected_q99 = np.array(per_timestep.q99, copy=True)
    expected_mean[..., 10:16] = 0.0
    expected_std[..., 10:16] = 1.0
    expected_q01[..., 10:16] = -1.0
    expected_q99[..., 10:16] = 1.0
    assert np.allclose(loaded_per_timestep.mean, expected_mean)
    assert np.allclose(loaded_per_timestep.std, expected_std)
    assert np.allclose(loaded_per_timestep.q01, expected_q01)
    assert np.allclose(loaded_per_timestep.q99, expected_q99)


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

    assert any(isinstance(t, _transforms.SetAdvantageLabelFromControlMode) for t in data_config.data_transforms.inputs)


def test_binpack_advantage_prompt_runs_before_binpack_inputs(tmp_path):
    factory = _config.LeRobotBinPackDataConfig(
        repo_id="repo",
        use_control_mode_advantage_prompt=True,
    )
    model_config = pi0_config.Pi0Config(pi05=True, action_horizon=2)
    data_config = factory.create(tmp_path, model_config)

    assert isinstance(data_config.data_transforms.inputs[0], _transforms.SetAdvantageLabelFromControlMode)


def test_binpack_config_can_select_positive_only_advantage_mode(tmp_path):
    factory = _config.LeRobotBinPackDataConfig(
        repo_id="repo",
        use_control_mode_advantage_prompt=True,
        advantage_prompt_mode="positive_only",
    )
    model_config = pi0_config.Pi0Config(pi05=True, action_horizon=2)
    data_config = factory.create(tmp_path, model_config)

    transform = data_config.data_transforms.inputs[0]
    assert isinstance(transform, _transforms.SetAdvantageLabelFromControlMode)
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
    assert isinstance(transform, _transforms.SetAdvantageLabelFromControlMode)
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


def test_build_block_tower_recap_configs_exist():
    nonhier_positive_only = _config.get_config("pi05_build_block_tower_recap_positive_only")
    nonhier_mixed = _config.get_config("pi05_build_block_tower_recap_mixed")
    hier_positive_only = _config.get_config("pi05_build_block_tower_subtask_recap_positive_only")
    hier_mixed = _config.get_config("pi05_build_block_tower_subtask_recap_mixed")

    for cfg, mode in (
        (nonhier_positive_only, "positive_only"),
        (nonhier_mixed, "mixed"),
        (hier_positive_only, "positive_only"),
        (hier_mixed, "mixed"),
    ):
        assert cfg.data.use_control_mode_advantage_prompt is True
        assert cfg.data.advantage_prompt_mode == mode
        assert cfg.data.advantage_dropout_rate == 0.3


def test_build_block_tower_recap_configs_use_same_dataset_mix():
    expected_repo_ids = (
        "villekuosmanen/build_block_tower",
        "villekuosmanen/dAgger_build_block_tower_1.0.0",
        "villekuosmanen/dAgger_build_block_tower_1.1.0",
        "villekuosmanen/dAgger_build_block_tower_1.2.0",
        "villekuosmanen/dAgger_build_block_tower_1.3.0",
        "villekuosmanen/dAgger_build_block_tower_1.4.0",
    )

    configs = (
        _config.get_config("pi05_build_block_tower_recap_positive_only"),
        _config.get_config("pi05_build_block_tower_recap_mixed"),
        _config.get_config("pi05_build_block_tower_subtask_recap_positive_only"),
        _config.get_config("pi05_build_block_tower_subtask_recap_mixed"),
    )

    for cfg in configs:
        for repo_id in expected_repo_ids:
            assert repo_id in cfg.data.repo_id


def test_build_block_tower_recap_configs_split_hierarchical_prompt_paths(tmp_path, monkeypatch):
    class _DummyTokenizer:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(_config._tokenizer, "PaligemmaTokenizer", _DummyTokenizer)

    nonhier = _config.get_config("pi05_build_block_tower_recap_mixed")
    hier = _config.get_config("pi05_build_block_tower_subtask_recap_mixed")

    nonhier_data = nonhier.data.create(tmp_path, nonhier.model)
    hier_data = hier.data.create(tmp_path, hier.model)

    assert any(
        isinstance(transform, _transforms.TokenizeHighPrompt)
        for transform in nonhier_data.model_transforms.inputs
    )
    assert not any(
        isinstance(transform, _transforms.TokenizeHighLowPrompt)
        for transform in nonhier_data.model_transforms.inputs
    )

    assert any(
        isinstance(transform, _transforms.TokenizeHighLowPrompt)
        for transform in hier_data.model_transforms.inputs
    )


def test_build_block_tower_hierarchical_recap_disables_fast_tokens(tmp_path, monkeypatch):
    class _DummyTokenizer:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(_config._tokenizer, "PaligemmaTokenizer", _DummyTokenizer)

    hier = _config.get_config("pi05_build_block_tower_subtask_recap_mixed")
    hier_data = hier.data.create(tmp_path, hier.model)

    assert hier.model.subtask_loss_weight > 0
    assert hier.model.fast_token_loss_weight == 0.0

    tokenize = next(
        transform for transform in hier_data.model_transforms.inputs if isinstance(transform, _transforms.TokenizeHighLowPrompt)
    )
    assert tokenize.use_fast_tokens is False


def test_build_block_tower_recap_uses_base_and_dagger_datasets():
    recap = _config.get_config("pi05_build_block_tower_recap_mixed")

    for repo_id in (
        "villekuosmanen/build_block_tower",
        "villekuosmanen/dAgger_build_block_tower_1.0.0",
        "villekuosmanen/dAgger_build_block_tower_1.1.0",
        "villekuosmanen/dAgger_build_block_tower_1.2.0",
        "villekuosmanen/dAgger_build_block_tower_1.3.0",
        "villekuosmanen/dAgger_build_block_tower_1.4.0",
    ):
        assert repo_id in recap.data.repo_id


def test_build_block_tower_recap_uses_17d_outputs(tmp_path, monkeypatch):
    class _DummyTokenizer:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(_config._tokenizer, "PaligemmaTokenizer", _DummyTokenizer)

    recap = _config.get_config("pi05_build_block_tower_recap_mixed")
    data_config = recap.data.create(tmp_path, recap.model)

    assert data_config.action_sequence_keys == ("action",)
    assert data_config.data_transforms.outputs[0].action_dim == 17


def test_build_block_tower_recap_only_deltas_real_7d_dims(tmp_path, monkeypatch):
    class _DummyTokenizer:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(_config._tokenizer, "PaligemmaTokenizer", _DummyTokenizer)

    recap = _config.get_config("pi05_build_block_tower_recap_mixed")
    data_config = recap.data.create(tmp_path, recap.model)

    delta = next(t for t in data_config.data_transforms.inputs if isinstance(t, _transforms.DeltaActionsFromState))
    assert tuple(delta.mask) == tuple([True] * 10 + [False] * 6 + [True])


def test_build_block_tower_rlt_6mix_references_published_baseline_checkpoint():
    recap = _config.get_config("pi05_build_block_tower_recap_mixed")
    rlt = _config.get_config("pi05_rlt_build_block_tower_6mix")
    expected_step = str(recap.num_train_steps - 1)

    assert expected_step == "49999"
    assert rlt.num_train_steps == 50_000
    assert isinstance(rlt.weight_loader, _config.weight_loaders.RLTokenCheckpointWeightLoader)
    assert (
        rlt.weight_loader.params_path
        == f"checkpoints/pi05_build_block_tower_baseline_6mix/baseline/{expected_step}/params"
    )

    train_script = pathlib.Path("slurm/train_build_block_tower_slurm.sh").read_text()
    assert 'CONFIG_NAME="pi05_build_block_tower_recap_mixed"' in train_script
    assert 'EXP_NAME="baseline"' in train_script

    rlt_script = pathlib.Path("slurm/train_build_block_tower_rlt_slurm.sh").read_text()
    assert 'CONFIG_NAME="pi05_rlt_build_block_tower_6mix"' in rlt_script
    assert 'EXP_NAME="rlt_6mix_v1"' in rlt_script
    assert 'BASELINE_HF_REPO="pravsels/pi05-build-block-tower-6mix"' in rlt_script
    assert f'BASELINE_STEP="{expected_step}"' in rlt_script
    assert 'BASELINE_LOCAL_DIR="${data_dir}/checkpoints/pi05_build_block_tower_baseline_6mix/baseline"' in rlt_script


def test_build_block_tower_rlt_6mix_uses_same_dataset_mix_as_baseline():
    recap = _config.get_config("pi05_build_block_tower_recap_mixed")
    rlt = _config.get_config("pi05_rlt_build_block_tower_6mix")

    assert rlt.data.repo_id == recap.data.repo_id


def test_build_block_tower_rlt_joints_only_config_and_script():
    rlt = _config.get_config("pi05_rlt_build_block_tower_6mix_joints_only")

    assert rlt.num_train_steps == 50_000
    assert isinstance(rlt.weight_loader, _config.weight_loaders.RLTokenCheckpointWeightLoader)
    assert (
        rlt.weight_loader.params_path
        == "checkpoints/pi05_build_block_tower_baseline_6mix_joints_only/joints_only/49999/params"
    )
    assert rlt.data.joints_only is True

    rlt_script = pathlib.Path("slurm/train_build_block_tower_rlt_joints_only_slurm.sh").read_text()
    assert 'CONFIG_NAME="pi05_rlt_build_block_tower_6mix_joints_only"' in rlt_script
    assert 'EXP_NAME="rlt_6mix_joints_only_v1"' in rlt_script
    assert 'BASELINE_HF_REPO="pravsels/build_block_tower_baseline_6mix_joints_only"' in rlt_script
    assert 'BASELINE_STEP="49999"' in rlt_script


def test_build_block_tower_recap_slurm_script_references_existing_configs():
    script = pathlib.Path("slurm/train_build_block_tower_recap_slurm.sh").read_text()

    for config_name in (
        "pi05_build_block_tower_recap_positive_only",
        "pi05_build_block_tower_recap_mixed",
        "pi05_build_block_tower_subtask_recap_positive_only",
        "pi05_build_block_tower_subtask_recap_mixed",
    ):
        assert config_name in script
        _config.get_config(config_name)


def test_reward_recap_slurm_script_references_existing_configs():
    script = pathlib.Path("slurm/train_bin_pack_reward_recap_slurm.sh").read_text()

    for config_name in (
        "pi05_bin_pack_coffee_capsules_recap_positive_only",
        "pi05_bin_pack_coffee_capsules_recap_mixed",
    ):
        assert config_name in script
        _config.get_config(config_name)
