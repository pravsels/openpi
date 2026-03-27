from pathlib import Path

from openpi.training import dataset_split


class _FakeSubDataset:
    def __init__(self, rows):
        self.hf_dataset = rows


class _FakeWrappedDataset:
    def __init__(self, groups):
        self._datasets = [_FakeSubDataset(rows) for rows in groups]
        self._dataset_lengths = [len(rows) for rows in groups]

    def __len__(self):
        return sum(self._dataset_lengths)

    def __getitem__(self, index):
        raise AssertionError("split logic should not materialize full dataset samples")


def test_episode_split_is_deterministic():
    episode_ids = [f"ep_{i:03d}" for i in range(200)]

    split_a = dataset_split.compute_episode_split(episode_ids, val_ratio=0.1, seed=42)
    split_b = dataset_split.compute_episode_split(episode_ids, val_ratio=0.1, seed=42)

    assert split_a == split_b
    assert len(split_a.train_episode_ids) == 180
    assert len(split_a.val_episode_ids) == 20
    assert set(split_a.train_episode_ids).isdisjoint(split_a.val_episode_ids)
    assert set(split_a.train_episode_ids) | set(split_a.val_episode_ids) == set(episode_ids)


def test_episode_split_round_trip(tmp_path: Path):
    split = dataset_split.compute_episode_split(
        [f"episode_{i}" for i in range(20)],
        val_ratio=0.1,
        seed=7,
    )

    dataset_split.save_episode_split(tmp_path, split)
    loaded = dataset_split.load_episode_split(tmp_path)

    assert loaded == split


def test_filter_items_by_episode_split():
    items = [
        {"episode_id": "ep_a"},
        {"episode_id": "ep_a"},
        {"episode_id": "ep_b"},
        {"episode_id": "ep_c"},
        {"episode_id": "ep_c"},
    ]
    split = dataset_split.EpisodeSplit(
        train_episode_ids=("ep_a", "ep_c"),
        val_episode_ids=("ep_b",),
        val_ratio=0.1,
        seed=42,
    )

    train_indices = dataset_split.filter_indices_by_episode_split(items, split, split_name="train")
    val_indices = dataset_split.filter_indices_by_episode_split(items, split, split_name="val")

    assert train_indices == [0, 1, 3, 4]
    assert val_indices == [2]


def test_wrapped_dataset_split_uses_hf_metadata_not_full_samples():
    dataset = _FakeWrappedDataset(
        [
            [
                {"episode_index": 10},
                {"episode_index": 10},
                {"episode_index": 11},
            ],
            [
                {"episode_index": 12},
                {"episode_index": 12},
            ],
        ]
    )
    split = dataset_split.EpisodeSplit(
        train_episode_ids=("10", "12"),
        val_episode_ids=("11",),
        val_ratio=0.1,
        seed=42,
    )

    episode_ids = dataset_split.get_episode_ids_from_dataset(dataset)
    train_indices = dataset_split.filter_dataset_indices_by_episode_split(dataset, split, split_name="train")
    val_indices = dataset_split.filter_dataset_indices_by_episode_split(dataset, split, split_name="val")

    assert episode_ids == ["10", "10", "11", "12", "12"]
    assert train_indices == [0, 1, 3, 4]
    assert val_indices == [2]
