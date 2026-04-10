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


# ---------------------------------------------------------------------------
# action_correlation_cholesky field
# ---------------------------------------------------------------------------

def test_norm_stats_correlation_cholesky_roundtrip():
    """action_correlation_cholesky serializes and deserializes correctly."""
    flat_dim = 6
    L = np.eye(flat_dim, dtype=np.float64) * 2.0
    stats = normalize.NormStats(
        mean=np.zeros(3),
        std=np.ones(3),
        action_correlation_cholesky=L,
    )
    norm_stats = {"actions": stats}
    loaded = normalize.deserialize_json(normalize.serialize_json(norm_stats))
    assert loaded["actions"].action_correlation_cholesky is not None
    assert loaded["actions"].action_correlation_cholesky.shape == (flat_dim, flat_dim)
    assert np.allclose(loaded["actions"].action_correlation_cholesky, L)


def test_norm_stats_correlation_cholesky_none_by_default():
    """action_correlation_cholesky is None when not provided."""
    stats = normalize.NormStats(mean=np.zeros(3), std=np.ones(3))
    assert stats.action_correlation_cholesky is None
    norm_stats = {"actions": stats}
    loaded = normalize.deserialize_json(normalize.serialize_json(norm_stats))
    assert loaded["actions"].action_correlation_cholesky is None


def test_compute_action_correlation_cholesky():
    """compute_action_correlation_cholesky produces a valid lower-triangular Cholesky factor."""
    rng = np.random.default_rng(42)
    n_samples, action_horizon, action_dim = 200, 4, 3
    flat_dim = action_horizon * action_dim
    actions = rng.standard_normal((n_samples, action_horizon, action_dim))

    mean = np.mean(actions.reshape(-1, action_dim), axis=0)  # (action_dim,)
    std = np.std(actions.reshape(-1, action_dim), axis=0)    # (action_dim,)

    L = normalize.compute_action_correlation_cholesky(actions, mean, std)
    assert L.shape == (flat_dim, flat_dim)
    # L should be lower triangular
    assert np.allclose(L, np.tril(L))
    # L L^T should be positive definite (all eigenvalues > 0)
    reconstructed = L @ L.T
    eigenvalues = np.linalg.eigvalsh(reconstructed)
    assert np.all(eigenvalues > 0)


def test_compute_action_correlation_cholesky_identity_for_iid():
    """When dimensions are independent with unit variance, L is close to identity."""
    rng = np.random.default_rng(7)
    n_samples, action_horizon, action_dim = 10000, 2, 2
    flat_dim = action_horizon * action_dim
    actions = rng.standard_normal((n_samples, action_horizon, action_dim))

    mean = np.zeros(action_dim)
    std = np.ones(action_dim)

    L = normalize.compute_action_correlation_cholesky(actions, mean, std)
    assert np.allclose(L, np.eye(flat_dim), atol=0.1)
