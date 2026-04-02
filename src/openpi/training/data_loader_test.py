import dataclasses

import jax
import numpy as np
import pytest
import torch

from openpi.models import pi0_config
import openpi.shared.normalize as _normalize
import openpi.transforms as _transforms
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    # Expect a finite TorchDataLoader to yield exactly num_batches batches and each
    # batch to preserve the configured local batch size across all leaves.
    # Example: dataset=16, local_batch_size=4, num_batches=2 -> exactly 2 batches of 4.
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    # Expect an infinite TorchDataLoader (no num_batches limit) to keep producing
    # batches without StopIteration, even when the underlying dataset is small.
    # Example: dataset=4, local_batch_size=4 -> repeatedly returns one full batch.
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    # Expect TorchDataLoader with worker processes to still produce the requested
    # number of batches and keep the local batch size consistent across all leaves.
    # Example: dataset=10, local_batch_size=4, num_batches=2 -> 2 batches of 4.
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    # Expect create_data_loader to work end-to-end for the fake dataset config,
    # honoring the configured batch size and action tensor shapes.
    # Example: config.batch_size=4, action_horizon=50, action_dim=24 ->
    # actions shape (4, 50, 24).
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    # Expect create_data_loader to succeed for a real dataset config when norm
    # stats are skipped (so data doesn't need to be present) and produce batches
    # with the configured action shapes.
    # Example: batch_size=4, action_horizon=50, action_dim=24 ->
    # actions shape (4, 50, 24).
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


_BIN_PACK_REWARD_RECAP_REPO_CASES = [
    ("villekuosmanen/bin_pick_pack_coffee_capsules", False),
    ("villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.0.0", True),
    ("villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.1.0", True),
    ("villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.2.0", True),
    ("villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.3.1", True),
    ("villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.4.0", True),
    ("villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.5.0", True),
    ("villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.5.1", True),
    ("villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.7.0", True),
]


def _to_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, str):
            return value
    return str(value)


@pytest.mark.parametrize(("repo_id", "expect_negative_prompt"), _BIN_PACK_REWARD_RECAP_REPO_CASES)
def test_binpack_reward_recap_datasets_expose_control_mode_through_pipeline(
    repo_id: str, expect_negative_prompt: bool, tmp_path
):
    pytest.importorskip("robocandywrapper")

    model_config = pi0_config.Pi0Config(pi05=True, action_horizon=2)
    factory = _config.LeRobotBinPackDataConfig(
        repo_id=repo_id,
        base_config=_config.DataConfig(prompt_from_task=True),
        use_control_mode_advantage_prompt=True,
        advantage_prompt_mode="mixed",
    )
    data_config = factory.create(tmp_path, model_config)

    try:
        dataset = _data_loader.create_torch_dataset(
            data_config,
            action_horizon=model_config.action_horizon,
            model_config=model_config,
        )
    except Exception as e:
        pytest.skip(f"Cannot load dataset {repo_id}: {e}")

    data_transform = _transforms.compose(data_config.data_transforms.inputs)
    sample_count = min(len(dataset), 64)
    if sample_count == 0:
        pytest.skip(f"Dataset {repo_id} is empty")

    sample_indices = sorted(set(range(min(16, sample_count))) | set(range(0, sample_count, max(1, sample_count // 8))))
    seen_modes = set()
    seen_prompts = []

    for idx in sample_indices:
        sample = dataset[idx]
        assert "control_mode" in sample, f"{repo_id} sample {idx} is missing control_mode"

        mode = _to_text(sample["control_mode"]).strip().lower()
        seen_modes.add(mode)

        transformed = data_transform(sample)
        prompt = _to_text(transformed["prompt"])
        assert "Advantage:" in prompt, f"{repo_id} sample {idx} prompt missing advantage tag: {prompt!r}"
        seen_prompts.append(prompt)

    if expect_negative_prompt:
        assert any(mode != "unknown" for mode in seen_modes), (
            f"{repo_id} never exposed a concrete control_mode through the dataset pipeline. "
            f"Modes seen: {seen_modes}"
        )
        assert any("Advantage: negative" in prompt for prompt in seen_prompts), (
            f"{repo_id} never produced a negative advantage prompt through the dataset pipeline. "
            f"Prompts seen: {seen_prompts[:5]}"
        )
    else:
        assert all("Advantage: positive" in prompt for prompt in seen_prompts), (
            f"{repo_id} should stay positive-only under unknown control_mode. "
            f"Prompts seen: {seen_prompts[:5]}"
        )


def _expected_shuffled_indices(valid_indices: list[int], seed: int) -> list[int]:
    # Mirror the sampler's torch RNG-based shuffle so the expected ordering
    # matches the sampler's deterministic permutation for a given seed.
    # Example: valid_indices=[0,1,2,3], seed=7 -> deterministic permuted order.
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(len(valid_indices), generator=g).tolist()
    return [valid_indices[i] for i in perm]


def test_filtered_sampler_deterministic_per_epoch():
    valid_indices = list(range(10))
    sampler = _data_loader.FilteredSampler(valid_indices, shuffle=True, seed=123)

    # Epoch 0: ordering is deterministic from base seed.
    # Example: seed=123 -> same ordering every run.
    assert list(iter(sampler)) == _expected_shuffled_indices(valid_indices, 123)

    # Epoch 2: ordering is still deterministic but uses seed+epoch to reshuffle.
    # Example: seed=123, epoch=2 -> same ordering across runs, different from epoch 0.
    sampler.set_epoch(2)
    assert list(iter(sampler)) == _expected_shuffled_indices(valid_indices, 125)


def test_filtered_distributed_sampler_deterministic_per_epoch():
    valid_indices = list(range(10))
    sampler = _data_loader.FilteredDistributedSampler(
        valid_indices,
        num_replicas=2,
        rank=0,
        shuffle=True,
        drop_last=True,
        seed=7,
    )

    # Rank 0 should see its shard of the deterministic global ordering:
    # 1) shuffle with seed, 2) truncate/pad to total_size, 3) take every Nth index
    #    for this rank.
    # Example: valid_indices=0..9, num_replicas=2, rank=0 -> take even positions
    # from the global shuffled list after truncation/padding.
    expected = _expected_shuffled_indices(valid_indices, 7)
    expected = expected[: sampler.total_size]
    expected = expected[0 : sampler.total_size : sampler.num_replicas]
    assert list(iter(sampler)) == expected

    # Epoch shift should reshuffle deterministically, changing the global order
    # and therefore this rank's shard.
    # Example: epoch=3 uses seed+3, so rank 0 sees a different shard than epoch 0.
    sampler.set_epoch(3)
    expected = _expected_shuffled_indices(valid_indices, 10)
    expected = expected[: sampler.total_size]
    expected = expected[0 : sampler.total_size : sampler.num_replicas]
    assert list(iter(sampler)) == expected


class _TinyDataset:
    def __init__(self, samples: list[dict]):
        self._samples = samples

    def __getitem__(self, index):
        return self._samples[index]

    def __len__(self):
        return len(self._samples)


@dataclasses.dataclass(frozen=True)
class _ValidIndicesSegment:
    start_index: int
    end_index: int
    mode: str


@dataclasses.dataclass(frozen=True)
class _ValidIndicesOutcomeInstance:
    outcomes: dict[int, str]


@dataclasses.dataclass(frozen=True)
class _ValidIndicesControlInstance:
    episode_modes: dict[int, list[_ValidIndicesSegment]]


@dataclasses.dataclass(frozen=True)
class _ValidIndicesSubDataset:
    repo_id: str
    episode_data_index: dict[str, list[int]]


class _ValidIndicesDataset:
    def __init__(self):
        self._samples = [{} for _ in range(6)]
        self._datasets = [
            _ValidIndicesSubDataset(
                repo_id="fake/repo",
                episode_data_index={"from": [0, 3], "to": [3, 6]},
            )
        ]
        self._cumulative_lengths = [0]
        self._index_maps = [None]
        self._plugin_instances = [[
            _ValidIndicesOutcomeInstance({0: "success", 1: "failure"}),
            _ValidIndicesControlInstance({0: [_ValidIndicesSegment(1, 2, "policy")], 1: [_ValidIndicesSegment(0, 2, "policy")]}),
        ]]

    def __getitem__(self, index):
        return self._samples[index]

    def __len__(self):
        return len(self._samples)


def test_create_data_loader_writes_valid_indices_when_missing(tmp_path, monkeypatch):
    model_config = pi0_config.Pi0Config(action_dim=2, action_horizon=2, max_token_len=4)
    config = _config.TrainConfig(
        name="test_valid_indices",
        exp_name="test",
        model=model_config,
        data=_config.LeRobotBinPackDataConfig(repo_id="repo"),
        assets_dir=str(tmp_path),
        batch_size=2,
    )
    dataset = _ValidIndicesDataset()

    monkeypatch.setattr(_data_loader, "create_torch_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(_data_loader, "transform_dataset", lambda dataset, data_config, *, skip_norm_stats=False: dataset)

    class _DummyTorchDataLoader:
        def __init__(self, dataset, **kwargs):
            self.dataset = dataset
            self.kwargs = kwargs

        def __iter__(self):
            return iter(())

    monkeypatch.setattr(_data_loader, "TorchDataLoader", _DummyTorchDataLoader)

    _data_loader.create_data_loader(config, num_batches=1, skip_norm_stats=True)

    assert (tmp_path / _data_loader.VALID_INDICES_FILENAME).read_text() == "0"


def test_create_data_loader_fails_when_auto_valid_indices_missing_outcomes(tmp_path, monkeypatch):
    model_config = pi0_config.Pi0Config(action_dim=2, action_horizon=2, max_token_len=4)
    config = _config.TrainConfig(
        name="test_valid_indices",
        exp_name="test",
        model=model_config,
        data=_config.LeRobotBinPackDataConfig(repo_id="repo"),
        assets_dir=str(tmp_path),
        batch_size=2,
    )
    dataset = _ValidIndicesDataset()
    dataset._plugin_instances = [[
        _ValidIndicesOutcomeInstance({}),
        _ValidIndicesControlInstance({0: [_ValidIndicesSegment(0, 2, "human")]}),
    ]]

    monkeypatch.setattr(_data_loader, "create_torch_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(_data_loader, "transform_dataset", lambda dataset, data_config, *, skip_norm_stats=False: dataset)

    class _DummyTorchDataLoader:
        def __init__(self, dataset, **kwargs):
            self.dataset = dataset
            self.kwargs = kwargs

        def __iter__(self):
            return iter(())

    monkeypatch.setattr(_data_loader, "TorchDataLoader", _DummyTorchDataLoader)

    with pytest.raises(ValueError, match="Missing outcome metadata"):
        _data_loader.create_data_loader(config, num_batches=1, skip_norm_stats=True)


def test_create_torch_data_loader_computes_missing_per_timestep_action_stats(tmp_path, monkeypatch):
    model_config = pi0_config.Pi0Config(action_dim=2, action_horizon=2, max_token_len=4)
    samples = [
        {
            "state": np.array([0.0, 1.0], dtype=np.float32),
            "actions": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        },
        {
            "state": np.array([1.0, 2.0], dtype=np.float32),
            "actions": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        },
    ]
    dataset = _TinyDataset(samples)
    norm_stats = {
        "state": _normalize.NormStats(mean=np.zeros(2), std=np.ones(2)),
        "actions": _normalize.NormStats(mean=np.zeros(2), std=np.ones(2)),
    }
    data_config = _config.DataConfig(
        repo_id="repo",
        asset_id="asset",
        norm_stats=norm_stats,
        use_per_timestep_action_norm=True,
        per_timestep_action_norm_stats=None,
    )

    monkeypatch.setattr(_data_loader, "create_torch_dataset", lambda *args, **kwargs: dataset)

    loader = _data_loader.create_torch_data_loader(
        data_config,
        model_config=model_config,
        action_horizon=model_config.action_horizon,
        batch_size=2,
        num_batches=1,
        assets_dir=tmp_path,
    )

    computed_stats = loader.data_config().per_timestep_action_norm_stats
    assert computed_stats is not None
    assert computed_stats.mean.shape == (2, 2)
    assert np.allclose(computed_stats.mean, np.array([[3.0, 4.0], [5.0, 6.0]]))
    assert (tmp_path / "asset" / "norm_stats_actions_per_timestep.json").exists()
