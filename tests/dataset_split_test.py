from pathlib import Path

from openpi.training import dataset_split


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
