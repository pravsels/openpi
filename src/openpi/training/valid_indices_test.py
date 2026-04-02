import dataclasses
import pathlib

import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
import openpi.training.valid_indices as _valid_indices


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
    output_path = tmp_path / "valid_indices.txt"
    policy = _valid_indices.ValidIndicesPolicy(mode="positive_only")

    written_path = _valid_indices.ensure_valid_indices_file(dataset, output_path, policy)
    assert written_path == output_path
    assert output_path.read_text() == "0"

    output_path.write_text("7,8,9")
    reused_path = _valid_indices.ensure_valid_indices_file(dataset, output_path, policy)
    assert reused_path == output_path
    assert output_path.read_text() == "7,8,9"
