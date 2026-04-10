"""Tests for action inpainting helpers (shared between Pi0 and Pi05)."""

import jax
import jax.numpy as jnp
import pytest

from openpi.models import action_inpainting


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def test_build_inpainting_indices_basic():
    """O covers first k timesteps × first d dims; U covers everything else."""
    action_horizon, action_dim = 4, 8
    num_initial_steps, input_action_dim = 2, 8

    O, U = action_inpainting.build_inpainting_indices(
        num_initial_steps, input_action_dim, action_horizon, action_dim,
    )
    # O should contain all indices for first 2 timesteps (2*8=16 indices)
    assert O.shape == (16,)
    # U should contain the remaining (4*8 - 16 = 16 indices)
    assert U.shape == (16,)
    # O and U should be disjoint and cover the full flat space
    all_indices = jnp.sort(jnp.concatenate([O, U]))
    assert jnp.array_equal(all_indices, jnp.arange(action_horizon * action_dim))


def test_build_inpainting_indices_partial_dim():
    """When input_action_dim < action_dim, only those dims per timestep are in O."""
    action_horizon, action_dim = 4, 8
    num_initial_steps, input_action_dim = 2, 5

    O, U = action_inpainting.build_inpainting_indices(
        num_initial_steps, input_action_dim, action_horizon, action_dim,
    )
    assert O.shape == (2 * 5,)
    assert U.shape == (4 * 8 - 10,)
    # First O index should be 0, second should be 1, ..., 5th should be 4
    # (first timestep, dims 0-4), then 8, 9, 10, 11, 12 (second timestep, dims 0-4)
    expected_O = jnp.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12])
    assert jnp.array_equal(O, expected_O)


def test_build_inpainting_indices_disjoint():
    """O and U must be disjoint and cover all indices."""
    for h, d, k, id_ in [(6, 10, 3, 7), (10, 32, 5, 23), (4, 8, 1, 8)]:
        O, U = action_inpainting.build_inpainting_indices(k, id_, h, d)
        all_idx = jnp.sort(jnp.concatenate([O, U]))
        assert jnp.array_equal(all_idx, jnp.arange(h * d)), f"Failed for h={h}, d={d}, k={k}, id_={id_}"


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------

def test_pad_initial_actions_no_padding_needed():
    ia = jnp.ones((1, 3, 8))
    padded = action_inpainting.pad_initial_actions(ia, action_horizon=4, action_dim=8)
    assert padded.shape == (1, 4, 8)
    assert jnp.allclose(padded[:, :3, :], ia)
    assert jnp.allclose(padded[:, 3:, :], 0.0)


def test_pad_initial_actions_dim_padding():
    ia = jnp.ones((1, 2, 5))
    padded = action_inpainting.pad_initial_actions(ia, action_horizon=4, action_dim=8)
    assert padded.shape == (1, 4, 8)
    assert jnp.allclose(padded[:, :2, :5], 1.0)
    assert jnp.allclose(padded[:, :2, 5:], 0.0)


def test_pad_initial_actions_truncates_excess_dims():
    ia = jnp.ones((1, 2, 10))
    padded = action_inpainting.pad_initial_actions(ia, action_horizon=4, action_dim=8)
    assert padded.shape == (1, 4, 8)


# ---------------------------------------------------------------------------
# Hard inpainting constraint
# ---------------------------------------------------------------------------

def test_apply_hard_inpainting_overwrites_constrained_coords():
    """After applying hard inpainting, O coordinates match the flow path."""
    batch_size, action_horizon, action_dim = 1, 4, 8
    num_initial_steps, input_action_dim = 2, 8

    x0_flat = jnp.zeros((batch_size, action_horizon * action_dim))
    z_flat = jnp.ones((batch_size, action_horizon * action_dim))
    # Start with x_t = all 5s (wrong value)
    x_t_flat = jnp.full((batch_size, action_horizon * action_dim), 5.0)

    O, U = action_inpainting.build_inpainting_indices(
        num_initial_steps, input_action_dim, action_horizon, action_dim,
    )
    time_new = 0.7  # somewhere in the middle of denoising

    result_flat = action_inpainting.apply_hard_inpainting(
        x_t_flat, x0_flat, z_flat, O, time_new,
    )

    # O coordinates should be (1-t)*x0 + t*z = 0.3*0 + 0.7*1 = 0.7
    expected_O = (1.0 - time_new) * x0_flat[:, O] + time_new * z_flat[:, O]
    assert jnp.allclose(result_flat[:, O], expected_O)

    # U coordinates should be unchanged (still 5.0)
    assert jnp.allclose(result_flat[:, U], 5.0)


def test_apply_hard_inpainting_at_time_zero():
    """At t=0 (clean), constrained coords should equal x0."""
    batch_size, flat_dim = 1, 32
    x0_flat = jnp.full((batch_size, flat_dim), 3.0)
    z_flat = jnp.full((batch_size, flat_dim), 7.0)
    x_t_flat = jnp.full((batch_size, flat_dim), 99.0)
    O = jnp.arange(16)

    result = action_inpainting.apply_hard_inpainting(x_t_flat, x0_flat, z_flat, O, time_new=0.0)
    assert jnp.allclose(result[:, O], 3.0)


def test_apply_hard_inpainting_at_time_one():
    """At t=1 (noise), constrained coords should equal z."""
    batch_size, flat_dim = 1, 32
    x0_flat = jnp.full((batch_size, flat_dim), 3.0)
    z_flat = jnp.full((batch_size, flat_dim), 7.0)
    x_t_flat = jnp.full((batch_size, flat_dim), 99.0)
    O = jnp.arange(16)

    result = action_inpainting.apply_hard_inpainting(x_t_flat, x0_flat, z_flat, O, time_new=1.0)
    assert jnp.allclose(result[:, O], 7.0)


# ---------------------------------------------------------------------------
# prepare_inpainting_state
# ---------------------------------------------------------------------------

def test_prepare_inpainting_state_shapes():
    batch_size, action_horizon, action_dim = 1, 6, 10
    ia = jnp.ones((batch_size, 3, 7))
    noise = jnp.zeros((batch_size, action_horizon, action_dim))

    state = action_inpainting.prepare_inpainting_state(
        ia, noise, action_horizon, action_dim,
    )
    assert state["O_indices"].shape == (3 * 7,)
    assert state["U_indices"].shape == (6 * 10 - 21,)
    assert state["x0_flat"].shape == (1, 60)
    assert state["z_flat"].shape == (1, 60)
    assert state["num_initial_steps"] == 3
    assert state["input_action_dim"] == 7


# ---------------------------------------------------------------------------
# Soft / time-thresholded inpainting
# ---------------------------------------------------------------------------

def test_should_apply_inpainting_above_threshold():
    assert action_inpainting.should_apply_inpainting(time_new=0.5, threshold=0.3) is True


def test_should_apply_inpainting_below_threshold():
    assert action_inpainting.should_apply_inpainting(time_new=0.2, threshold=0.3) is False


def test_should_apply_inpainting_at_threshold():
    """At exactly the threshold, constraint should NOT be applied (strictly greater)."""
    assert action_inpainting.should_apply_inpainting(time_new=0.3, threshold=0.3) is False


def test_should_apply_inpainting_zero_threshold_always_applies():
    """Threshold of 0.0 means always apply (any positive time is above it)."""
    assert action_inpainting.should_apply_inpainting(time_new=0.05, threshold=0.0) is True
    assert action_inpainting.should_apply_inpainting(time_new=0.0, threshold=0.0) is False


# ---------------------------------------------------------------------------
# Correlation loading & validation
# ---------------------------------------------------------------------------

def test_load_correlation_cholesky_valid():
    """Loads and returns a matching Cholesky factor."""
    import numpy as np
    from openpi.shared.normalize import NormStats

    action_horizon, action_dim = 4, 3
    flat_dim = action_horizon * action_dim
    L = np.eye(flat_dim, dtype=np.float64) * 2.0
    norm_stats = {
        "actions": NormStats(
            mean=np.zeros(action_dim),
            std=np.ones(action_dim),
            action_correlation_cholesky=L,
        ),
    }
    loaded = action_inpainting.load_correlation_cholesky(norm_stats, action_horizon, action_dim)
    assert loaded.shape == (flat_dim, flat_dim)
    assert jnp.allclose(loaded, jnp.array(L))


def test_load_correlation_cholesky_missing_raises():
    """Raises ValueError when Cholesky is None but caller requested it."""
    import numpy as np
    from openpi.shared.normalize import NormStats

    norm_stats = {
        "actions": NormStats(mean=np.zeros(3), std=np.ones(3)),
    }
    with pytest.raises(ValueError, match="action_correlation_cholesky"):
        action_inpainting.load_correlation_cholesky(norm_stats, 4, 3)


def test_load_correlation_cholesky_dimension_mismatch_raises():
    """Raises ValueError when stored Cholesky has wrong dimensions."""
    import numpy as np
    from openpi.shared.normalize import NormStats

    wrong_dim = 5
    L = np.eye(wrong_dim)
    norm_stats = {
        "actions": NormStats(
            mean=np.zeros(3),
            std=np.ones(3),
            action_correlation_cholesky=L,
        ),
    }
    with pytest.raises(ValueError, match="dimension mismatch"):
        action_inpainting.load_correlation_cholesky(norm_stats, 4, 3)


def test_load_correlation_cholesky_no_actions_key_raises():
    """Raises ValueError when norm_stats has no 'actions' key."""
    with pytest.raises(ValueError, match="actions"):
        action_inpainting.load_correlation_cholesky({}, 4, 3)


# ---------------------------------------------------------------------------
# Correlation-aware correction matrix
# ---------------------------------------------------------------------------

def test_precompute_correction_matrix_shape():
    """Correction matrix has shape [|U|, |O|]."""
    action_horizon, action_dim = 4, 3
    flat_dim = action_horizon * action_dim
    L = jnp.eye(flat_dim)
    O, U = action_inpainting.build_inpainting_indices(2, 3, action_horizon, action_dim)

    M = action_inpainting.precompute_correction_matrix(L, O, U)
    assert M.shape == (U.shape[0], O.shape[0])


def test_precompute_correction_matrix_identity_covariance():
    """With identity covariance (independent dims), correction is all zeros."""
    action_horizon, action_dim = 4, 3
    flat_dim = action_horizon * action_dim
    L = jnp.eye(flat_dim)
    O, U = action_inpainting.build_inpainting_indices(2, 3, action_horizon, action_dim)

    M = action_inpainting.precompute_correction_matrix(L, O, U)
    assert jnp.allclose(M, 0.0, atol=1e-5)


def test_precompute_correction_matrix_correlated():
    """Non-trivial correlation produces a non-zero correction matrix."""
    import numpy as np
    action_horizon, action_dim = 4, 3
    flat_dim = action_horizon * action_dim
    rng = np.random.default_rng(42)
    A = rng.standard_normal((flat_dim, flat_dim))
    cov = A @ A.T + 0.1 * np.eye(flat_dim)
    L = jnp.array(np.linalg.cholesky(cov))

    O, U = action_inpainting.build_inpainting_indices(2, 3, action_horizon, action_dim)
    M = action_inpainting.precompute_correction_matrix(L, O, U)

    assert M.shape == (U.shape[0], O.shape[0])
    assert not jnp.allclose(M, 0.0, atol=1e-3)


def test_apply_correlation_correction():
    """Correlation correction modifies U coordinates based on O delta."""
    import numpy as np
    action_horizon, action_dim = 4, 3
    flat_dim = action_horizon * action_dim
    batch_size = 2

    rng = np.random.default_rng(42)
    A = rng.standard_normal((flat_dim, flat_dim))
    cov = A @ A.T + 0.1 * np.eye(flat_dim)
    L = jnp.array(np.linalg.cholesky(cov))

    O, U = action_inpainting.build_inpainting_indices(2, 3, action_horizon, action_dim)
    M = action_inpainting.precompute_correction_matrix(L, O, U)

    x_t_flat = jnp.ones((batch_size, flat_dim)) * 5.0
    x0_flat = jnp.zeros((batch_size, flat_dim))
    z_flat = jnp.ones((batch_size, flat_dim))
    time_new = 0.7
    beta = 0.5

    result = action_inpainting.apply_correlated_inpainting(
        x_t_flat, x0_flat, z_flat, O, U, M, time_new, beta,
    )
    # O coords should be exactly the flow path value
    expected_O = (1.0 - time_new) * x0_flat[:, O] + time_new * z_flat[:, O]
    assert jnp.allclose(result[:, O], expected_O)
    # U coords should differ from x_t_flat (corrected)
    assert not jnp.allclose(result[:, U], x_t_flat[:, U])


def test_apply_correlation_correction_beta_zero_is_hard_only():
    """With beta=0, correlated inpainting reduces to hard inpainting."""
    import numpy as np
    action_horizon, action_dim = 4, 3
    flat_dim = action_horizon * action_dim
    batch_size = 1

    rng = np.random.default_rng(42)
    A = rng.standard_normal((flat_dim, flat_dim))
    cov = A @ A.T + 0.1 * np.eye(flat_dim)
    L = jnp.array(np.linalg.cholesky(cov))

    O, U = action_inpainting.build_inpainting_indices(2, 3, action_horizon, action_dim)
    M = action_inpainting.precompute_correction_matrix(L, O, U)

    x_t_flat = jnp.ones((batch_size, flat_dim)) * 5.0
    x0_flat = jnp.zeros((batch_size, flat_dim))
    z_flat = jnp.ones((batch_size, flat_dim))
    time_new = 0.7

    result = action_inpainting.apply_correlated_inpainting(
        x_t_flat, x0_flat, z_flat, O, U, M, time_new, beta=0.0,
    )
    hard_result = action_inpainting.apply_hard_inpainting(
        x_t_flat, x0_flat, z_flat, O, time_new,
    )
    assert jnp.allclose(result, hard_result)
