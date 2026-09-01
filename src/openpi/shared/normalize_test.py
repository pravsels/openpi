import numpy as np

import openpi.shared.normalize as normalize
import openpi.transforms as transforms


def _sparse_gripper_stats(horizon: int = 4, collapsed_until: int = 2):
    """Per-timestep and global stats for an arm joint plus a late-opening gripper.

    Channel 0 moves throughout. Channel 1 only opens near the end of the horizon, so
    its early per-timestep quantiles land on the same value, exactly as a real gripper
    that stays shut until the last moment.
    """
    q01 = np.zeros((horizon, 2), dtype=np.float32)
    q99 = np.zeros((horizon, 2), dtype=np.float32)
    q99[:, 0] = np.linspace(10.0, 30.0, horizon)  # healthy channel, widening with t
    q99[collapsed_until:, 1] = 50.0  # gripper: no spread before collapsed_until
    per_timestep = normalize.NormStats(mean=np.zeros((horizon, 2)), std=q99 / 4.0, q01=q01, q99=q99)
    overall = normalize.NormStats(
        mean=np.zeros(2),
        std=np.array([30.0, 50.0]) / 4.0,
        q01=np.zeros(2),
        q99=np.array([30.0, 50.0], dtype=np.float32),
    )
    return per_timestep, overall


def test_normalize_update():
    arr = np.arange(12).reshape(4, 3)  # 4 vectors of length 3

    stats = normalize.RunningStats()
    for i in range(len(arr)):
        stats.update(arr[i : i + 1])  # Update with one vector at a time
    results = stats.get_statistics()

    assert np.allclose(results.mean, np.mean(arr, axis=0))
    assert np.allclose(results.std, np.std(arr, axis=0))


def test_running_stats_uses_one_and_ninety_ninth_percentiles_by_default():
    arr = np.array([[0.0, -10.0], [1.0, 0.0], [100.0, 10.0]])
    stats = normalize.RunningStats()
    stats.update(arr)

    results = stats.get_statistics()

    np.testing.assert_allclose(results.q01, np.array([-1e-10, -10.0000000001]))
    np.testing.assert_allclose(results.q99, np.array([99.98, 9.996]))


def test_running_stats_can_use_true_min_max_bounds():
    arr = np.array([[3.0, -4.0], [1.0, 8.0], [7.0, 2.0]])
    stats = normalize.RunningStats()
    stats.update(arr)

    results = stats.get_statistics(use_min_max=True)

    np.testing.assert_array_equal(results.q01, np.min(arr, axis=0))
    np.testing.assert_array_equal(results.q99, np.max(arr, axis=0))


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


def test_a_sparse_channel_is_normalized_globally_across_the_whole_horizon():
    per_timestep, overall = _sparse_gripper_stats()

    filled = normalize.backfill_collapsed_timesteps(per_timestep, overall)

    # The gripper takes its global range at every step, not just the collapsed ones:
    # the steps flanking a collapse are near-collapsed too.
    assert np.allclose(filled.q99[:, 1], overall.q99[1])
    assert np.allclose(filled.q01[:, 1], overall.q01[1])
    # The arm joint keeps its own per-step scale, including where it is narrowest.
    assert np.allclose(filled.q99[:, 0], per_timestep.q99[:, 0])


def test_a_channel_that_never_collapses_is_left_alone():
    per_timestep, overall = _sparse_gripper_stats(collapsed_until=0)

    filled = normalize.backfill_collapsed_timesteps(per_timestep, overall)

    assert np.allclose(filled.q01, per_timestep.q01)
    assert np.allclose(filled.q99, per_timestep.q99)
    assert np.allclose(filled.std, per_timestep.std)


def test_a_globally_constant_channel_is_left_to_the_scale_guard():
    # A channel pinned for the whole dataset has no wider scale to borrow, so there is
    # nothing to backfill; transforms' degenerate-scale guard keeps it harmless.
    per_timestep = normalize.NormStats(
        mean=np.zeros((3, 2)), std=np.zeros((3, 2)), q01=np.zeros((3, 2)), q99=np.zeros((3, 2))
    )
    overall = normalize.NormStats(mean=np.zeros(2), std=np.zeros(2), q01=np.zeros(2), q99=np.zeros(2))

    filled = normalize.backfill_collapsed_timesteps(per_timestep, overall)

    assert np.allclose(filled.q99, per_timestep.q99)


def test_backfill_brings_a_sparse_gripper_target_back_into_range():
    per_timestep, overall = _sparse_gripper_stats()
    # A chunk whose gripper is wide open at the first step -- rare in training, which
    # is exactly why that step's quantiles missed it.
    actions = np.zeros((1, 4, 2), dtype=np.float32)
    actions[0, 0, 1] = 40.0

    def worst(stats):
        norm = transforms.Normalize({"actions": stats}, use_quantiles=True)
        return np.abs(norm({"actions": actions})["actions"]).max()

    assert worst(per_timestep) > 50.0  # ill-conditioned: target far outside [-1, 1]
    assert worst(normalize.backfill_collapsed_timesteps(per_timestep, overall)) <= 1.0


def test_backfill_uses_std_when_the_stats_have_no_quantiles():
    per_timestep = normalize.NormStats(mean=np.zeros((3, 2)), std=np.array([[1.0, 0.0], [1.0, 4.0], [1.0, 4.0]]))
    overall = normalize.NormStats(mean=np.zeros(2), std=np.array([1.0, 4.0]))

    filled = normalize.backfill_collapsed_timesteps(per_timestep, overall)

    assert np.allclose(filled.std[:, 1], 4.0)
    assert np.allclose(filled.std[:, 0], 1.0)
    assert filled.q01 is None
