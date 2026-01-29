import numpy as np

import openpi.shared.normalize as normalize


def test_normalize_update():
    arr = np.arange(12).reshape(4, 3)  # 4 vectors of length 3

    stats = normalize.RunningStats()
    for i in range(len(arr)):
        stats.update(arr[i : i + 1])  # Update with one vector at a time
    results = stats.get_statistics()

    assert np.allclose(results.mean, np.mean(arr, axis=0))
    assert np.allclose(results.std, np.std(arr, axis=0))


def test_serialize_deserialize():
    stats = normalize.RunningStats()
    stats.update(np.arange(12).reshape(4, 3))  # 4 vectors of length 3

    norm_stats = {"test": stats.get_statistics()}
    norm_stats2 = normalize.deserialize_json(normalize.serialize_json(norm_stats))
    assert np.allclose(norm_stats["test"].mean, norm_stats2["test"].mean)
    assert np.allclose(norm_stats["test"].std, norm_stats2["test"].std)


def test_multiple_batch_dimensions():
    # Test with multiple batch dimensions: (2, 3, 4) where 4 is vector dimension
    batch_shape = (2, 3, 4)
    arr = np.random.rand(*batch_shape)

    stats = normalize.RunningStats()
    stats.update(arr)  # Should handle (2, 3, 4) -> reshape to (6, 4)
    results = stats.get_statistics()

    # Flatten batch dimensions and compute expected stats
    flattened = arr.reshape(-1, arr.shape[-1])  # (6, 4)
    expected_mean = np.mean(flattened, axis=0)
    expected_std = np.std(flattened, axis=0)

    assert np.allclose(results.mean, expected_mean)
    assert np.allclose(results.std, expected_std)


def test_actions_per_timestep_roundtrip(tmp_path):
    stats = normalize.NormStats(
        mean=np.zeros((2, 3)),
        std=np.ones((2, 3)),
        q01=np.zeros((2, 3)),
        q99=np.ones((2, 3)) * 2.0,
    )
    normalize.save_actions_per_timestep(tmp_path, stats)
    loaded = normalize.load_actions_per_timestep(tmp_path)
    assert np.allclose(stats.mean, loaded.mean)
    assert np.allclose(stats.std, loaded.std)
    assert np.allclose(stats.q01, loaded.q01)
    assert np.allclose(stats.q99, loaded.q99)


def test_merge_action_norm_stats_disabled():
    base = {
        "actions": normalize.NormStats(mean=np.zeros(2), std=np.ones(2)),
        "state": normalize.NormStats(mean=np.zeros(2), std=np.ones(2)),
    }
    per_timestep = normalize.NormStats(mean=np.ones((2, 2)), std=np.ones((2, 2)) * 2.0)
    merged = normalize.merge_action_norm_stats(
        base, per_timestep_action_stats=per_timestep, use_per_timestep_action_norm=False
    )
    assert np.allclose(merged["actions"].mean, base["actions"].mean)


def test_merge_action_norm_stats_enabled_missing():
    base = {
        "actions": normalize.NormStats(mean=np.zeros(2), std=np.ones(2)),
        "state": normalize.NormStats(mean=np.zeros(2), std=np.ones(2)),
    }
    merged = normalize.merge_action_norm_stats(
        base, per_timestep_action_stats=None, use_per_timestep_action_norm=True
    )
    assert np.allclose(merged["actions"].mean, base["actions"].mean)


def test_merge_action_norm_stats_enabled():
    base = {
        "actions": normalize.NormStats(mean=np.zeros(2), std=np.ones(2)),
        "state": normalize.NormStats(mean=np.zeros(2), std=np.ones(2)),
    }
    per_timestep = normalize.NormStats(mean=np.ones((2, 2)), std=np.ones((2, 2)) * 2.0)
    merged = normalize.merge_action_norm_stats(
        base, per_timestep_action_stats=per_timestep, use_per_timestep_action_norm=True
    )
    assert np.allclose(merged["actions"].mean, per_timestep.mean)
