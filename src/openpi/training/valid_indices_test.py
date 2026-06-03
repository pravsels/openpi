import dataclasses
import json
import pathlib

import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
import openpi.training.valid_indices as _valid_indices


class TestRangeHelpers:
    def test_indices_to_ranges_empty(self):
        assert _valid_indices.indices_to_ranges([]) == []

    def test_indices_to_ranges_single(self):
        assert _valid_indices.indices_to_ranges([5]) == [[5, 5]]

    def test_indices_to_ranges_contiguous(self):
        assert _valid_indices.indices_to_ranges([0, 1, 2, 3, 4]) == [[0, 4]]

    def test_indices_to_ranges_gaps(self):
        assert _valid_indices.indices_to_ranges([0, 1, 2, 5, 6, 10]) == [[0, 2], [5, 6], [10, 10]]

    def test_ranges_to_indices_empty(self):
        assert _valid_indices.ranges_to_indices([]) == []

    def test_ranges_to_indices_single_range(self):
        assert _valid_indices.ranges_to_indices([[0, 4]]) == [0, 1, 2, 3, 4]

    def test_ranges_to_indices_multiple(self):
        assert _valid_indices.ranges_to_indices([[0, 2], [5, 6], [10, 10]]) == [0, 1, 2, 5, 6, 10]

    def test_roundtrip(self):
        original = [0, 1, 2, 5, 6, 10, 100, 101, 102, 103]
        ranges = _valid_indices.indices_to_ranges(original)
        assert _valid_indices.ranges_to_indices(ranges) == original

    def test_all_valid_produces_single_range(self):
        indices = list(range(34443))
        ranges = _valid_indices.indices_to_ranges(indices)
        assert ranges == [[0, 34442]]


class TestSaveLoadValidIndices:
    def test_save_and_load_roundtrip(self, tmp_path: pathlib.Path):
        indices = [0, 1, 2, 5, 6, 10]
        path = tmp_path / "valid_indices.json"
        _valid_indices.save_valid_indices(indices, path)

        raw = json.loads(path.read_text())
        assert raw == [[0, 2], [5, 6], [10, 10]]

        loaded = _valid_indices.load_valid_indices(path)
        assert loaded == indices

    def test_save_all_valid(self, tmp_path: pathlib.Path):
        path = tmp_path / "valid_indices.json"
        indices = list(range(1000))
        _valid_indices.save_valid_indices(indices, path)

        raw = json.loads(path.read_text())
        assert raw == [[0, 999]]

    def test_load_empty_ranges(self, tmp_path: pathlib.Path):
        path = tmp_path / "valid_indices.json"
        path.write_text("[]")
        assert _valid_indices.load_valid_indices(path) == []


@dataclasses.dataclass(frozen=True)
class _Segment:
    start_index: int
    end_index: int
    mode: str


@dataclasses.dataclass(frozen=True)
class _OutcomeInstance:
    outcomes: dict[int, str]


@dataclasses.dataclass(frozen=True)
class _ControlInstance:
    episode_modes: dict[int, list[_Segment]]


@dataclasses.dataclass(frozen=True)
class _SubDataset:
    repo_id: str
    episode_data_index: dict[str, list[int]]


class _WrappedDataset:
    def __init__(self, *, outcomes: dict[int, str], episode_modes: dict[int, list[_Segment]]):
        self._datasets = [
            _SubDataset(
                repo_id="fake/repo",
                episode_data_index={"from": [0, 3], "to": [3, 6]},
            )
        ]
        self._cumulative_lengths = [0]
        self._index_maps = [None]
        self._plugin_instances = [[_OutcomeInstance(outcomes), _ControlInstance(episode_modes)]]
        self._len = 6

    def __len__(self):
        return self._len


def test_policy_from_train_config_defaults_to_successful_human_only():
    config = _config.TrainConfig(
        name="test",
        exp_name="test",
        model=pi0_config.Pi0Config(action_dim=2, action_horizon=2, max_token_len=4),
        data=_config.LeRobotBinPackDataConfig(repo_id="repo"),
    )

    policy = _valid_indices.policy_from_train_config(config)

    assert policy.mode == "positive_only"
    assert policy.require_outcomes is True


def test_compute_valid_indices_positive_only_keeps_successful_human_frames():
    dataset = _WrappedDataset(
        outcomes={0: "success", 1: "failure"},
        episode_modes={0: [_Segment(1, 2, "policy")], 1: [_Segment(0, 2, "policy")]},
    )

    valid = _valid_indices.compute_valid_indices(
        dataset,
        _valid_indices.ValidIndicesPolicy(mode="positive_only"),
    )

    assert valid == [0]


def test_compute_valid_indices_mixed_keeps_negative_policy_frames_from_failures():
    dataset = _WrappedDataset(
        outcomes={0: "success", 1: "failure"},
        episode_modes={0: [_Segment(1, 2, "policy")], 1: [_Segment(0, 2, "policy")]},
    )

    valid = _valid_indices.compute_valid_indices(
        dataset,
        _valid_indices.ValidIndicesPolicy(mode="mixed"),
    )

    assert valid == [0, 1, 2, 3, 4, 5]


def test_compute_valid_indices_missing_control_mode_treats_frames_as_human():
    dataset = _WrappedDataset(
        outcomes={0: "success", 1: "failure"},
        episode_modes={},
    )

    valid = _valid_indices.compute_valid_indices(
        dataset,
        _valid_indices.ValidIndicesPolicy(mode="mixed"),
    )

    assert valid == [0, 1, 2]


def test_compute_valid_indices_raises_when_outcomes_missing():
    dataset = _WrappedDataset(
        outcomes={0: "success"},
        episode_modes={0: [_Segment(0, 2, "human")]},
    )

    with pytest.raises(ValueError, match="Missing outcome metadata"):
        _valid_indices.compute_valid_indices(
            dataset,
            _valid_indices.ValidIndicesPolicy(mode="positive_only"),
        )


def test_ensure_valid_indices_file_writes_once_and_reuses_existing(tmp_path: pathlib.Path):
    dataset = _WrappedDataset(
        outcomes={0: "success", 1: "failure"},
        episode_modes={0: [_Segment(1, 2, "policy")], 1: [_Segment(0, 2, "policy")]},
    )
    output_path = tmp_path / "valid_indices.json"
    policy = _valid_indices.ValidIndicesPolicy(mode="positive_only")

    written_path = _valid_indices.ensure_valid_indices_file(dataset, output_path, policy)
    assert written_path == output_path
    assert json.loads(output_path.read_text()) == [[0, 0]]

    output_path.write_text("[[7, 9]]")
    reused_path = _valid_indices.ensure_valid_indices_file(dataset, output_path, policy)
    assert reused_path == output_path
    assert json.loads(output_path.read_text()) == [[7, 9]]


def test_ensure_valid_indices_file_falls_back_to_all_valid_without_outcomes(tmp_path: pathlib.Path):
    dataset = _WrappedDataset(
        outcomes={},
        episode_modes={},
    )
    dataset._len = 6
    output_path = tmp_path / "valid_indices.json"
    policy = _valid_indices.ValidIndicesPolicy(mode="positive_only")

    written_path = _valid_indices.ensure_valid_indices_file(dataset, output_path, policy)
    assert written_path == output_path
    assert json.loads(output_path.read_text()) == [[0, 5]]
