"""Shared action inpainting helpers for flow-matching denoising loops.

Used by both Pi0 and Pi05 to constrain a prefix of the action sequence during
denoising, so that previously predicted actions are held fixed while the model
generates the continuation.

Index convention: flatten (action_horizon, action_dim) in row-major order,
so flat index = timestep * action_dim + dim.  This matches the covariance
layout used by the correlation-aware correction (Tasks 7-8).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def build_inpainting_indices(
    num_initial_steps: int,
    input_action_dim: int,
    action_horizon: int,
    action_dim: int,
) -> tuple[jax.Array, jax.Array]:
    """Build flat index arrays for constrained (O) and free (U) coordinates.

    O covers the first ``num_initial_steps`` timesteps and the first
    ``input_action_dim`` action dimensions at each of those timesteps.
    U covers everything else.

    Returns:
        (O_indices, U_indices) as int32 JAX arrays.
    """
    flat_dim = action_horizon * action_dim
    O_list = [
        t * action_dim + d
        for t in range(num_initial_steps)
        for d in range(input_action_dim)
    ]
    O_set = set(O_list)
    U_list = [i for i in range(flat_dim) if i not in O_set]
    return jnp.array(O_list, dtype=jnp.int32), jnp.array(U_list, dtype=jnp.int32)


def pad_initial_actions(
    initial_actions: jax.Array,
    action_horizon: int,
    action_dim: int,
) -> jax.Array:
    """Pad (or truncate) initial_actions to (batch, action_horizon, action_dim).

    The input may have fewer timesteps or fewer action dimensions than the
    model expects.  Missing entries are zero-filled; excess dims are clipped.
    """
    batch_size = initial_actions.shape[0]
    num_steps = initial_actions.shape[1]
    in_dim = initial_actions.shape[2]

    ia = initial_actions[:, :, :action_dim]

    if in_dim < action_dim:
        ia = jnp.concatenate(
            [ia, jnp.zeros((batch_size, num_steps, action_dim - in_dim))],
            axis=2,
        )

    if num_steps < action_horizon:
        ia = jnp.concatenate(
            [ia, jnp.zeros((batch_size, action_horizon - num_steps, action_dim))],
            axis=1,
        )
    elif num_steps > action_horizon:
        ia = ia[:, :action_horizon, :]

    return ia


def should_apply_inpainting(time_new: float, threshold: float) -> bool:
    """Return True if inpainting should be enforced at this denoising time.

    The constraint is applied while `time_new > threshold`.  Once the flow
    drops to or below the threshold the model runs freely, producing a
    smoother blend between constrained and unconstrained regions.

    With ``threshold=0.0`` the constraint is applied at every step except
    the very last (t=0).
    """
    return time_new > threshold


def apply_hard_inpainting(
    x_t_flat: jax.Array,
    x0_flat: jax.Array,
    z_flat: jax.Array,
    O_indices: jax.Array,
    time_new: float | jax.Array,
) -> jax.Array:
    """Overwrite constrained coordinates so they follow the flow path.

    At denoising time ``time_new`` (where 1=noise, 0=clean), the desired value
    on constrained coordinates is::

        x_desired_O = (1 - time_new) * x0_O + time_new * z_O

    Args:
        x_t_flat: Current denoised state, shape ``(batch, flat_dim)``.
        x0_flat: Target clean actions (padded), shape ``(batch, flat_dim)``.
        z_flat: Fixed noise at constrained coords, shape ``(batch, flat_dim)``.
        O_indices: Flat indices of constrained coordinates.
        time_new: Current flow time after the Euler step.

    Returns:
        Updated ``x_t_flat`` with constrained coordinates overwritten.
    """
    x_desired_O = (1.0 - time_new) * x0_flat[:, O_indices] + time_new * z_flat[:, O_indices]
    return x_t_flat.at[:, O_indices].set(x_desired_O)


def load_correlation_cholesky(
    norm_stats: dict,
    action_horizon: int,
    action_dim: int,
) -> jax.Array:
    """Extract and validate the action-correlation Cholesky factor from norm stats.

    Raises ``ValueError`` on missing data or dimension mismatches.

    Returns:
        The Cholesky factor as a JAX array, shape
        ``(action_horizon * action_dim, action_horizon * action_dim)``.
    """
    if "actions" not in norm_stats:
        raise ValueError("norm_stats must contain an 'actions' key.")

    action_stats = norm_stats["actions"]
    L = getattr(action_stats, "action_correlation_cholesky", None)
    if L is None:
        raise ValueError(
            "action_correlation_cholesky is None in the action norm stats. "
            "Run compute_norm_stats.py with --compute-action-correlation to generate it."
        )

    flat_dim = action_horizon * action_dim
    L_np = np.asarray(L)
    if L_np.shape != (flat_dim, flat_dim):
        raise ValueError(
            f"action_correlation_cholesky dimension mismatch: expected "
            f"({flat_dim}, {flat_dim}) but got {L_np.shape}."
        )

    return jnp.array(L_np)


def precompute_correction_matrix(
    L: jax.Array,
    O_indices: jax.Array,
    U_indices: jax.Array,
    reg_eps: float = 1e-6,
) -> jax.Array:
    """Build the correlation-aware correction matrix M of shape ``(|U|, |O|)``.

    Given the Cholesky factor L of the action covariance Sigma = L L^T::

        Sigma_OO = Sigma[O, O]
        Sigma_UO = Sigma[U, O]
        M = Sigma_UO @ inv(Sigma_OO)

    During denoising, the correction for unconstrained dims is::

        delta_U = beta * M @ delta_O

    where delta_O is the change applied to constrained coordinates.
    """
    Sigma = L @ L.T
    Sigma_OO = Sigma[jnp.ix_(O_indices, O_indices)]
    Sigma_UO = Sigma[jnp.ix_(U_indices, O_indices)]

    Sigma_OO_reg = Sigma_OO + reg_eps * jnp.eye(Sigma_OO.shape[0])
    # M = Sigma_UO @ inv(Sigma_OO) — using solve for stability.
    # solve(A, B) gives inv(A) @ B, so solve(Sigma_OO^T, Sigma_UO^T) gives
    # inv(Sigma_OO^T) @ Sigma_UO^T = (Sigma_UO @ inv(Sigma_OO))^T
    M = jax.scipy.linalg.solve(Sigma_OO_reg, Sigma_UO.T, assume_a="pos").T
    return M


def apply_correlated_inpainting(
    x_t_flat: jax.Array,
    x0_flat: jax.Array,
    z_flat: jax.Array,
    O_indices: jax.Array,
    U_indices: jax.Array,
    correction_matrix: jax.Array,
    time_new: float | jax.Array,
    beta: float,
) -> jax.Array:
    """Apply hard inpainting on O and correlation correction on U.

    1. Compute the desired O value on the flow path.
    2. Compute delta_O = desired_O - x_t_O (the displacement applied to O).
    3. Overwrite O coordinates (hard constraint).
    4. Correct U coordinates: x_t_U += beta * M @ delta_O.
    """
    x_desired_O = (1.0 - time_new) * x0_flat[:, O_indices] + time_new * z_flat[:, O_indices]
    delta_O = x_desired_O - x_t_flat[:, O_indices]

    # Hard-set constrained dims.
    result = x_t_flat.at[:, O_indices].set(x_desired_O)

    # Correlation correction on unconstrained dims.
    if beta > 0.0:
        # delta_O: (batch, |O|), correction_matrix: (|U|, |O|)
        delta_U = beta * (delta_O @ correction_matrix.T)
        result = result.at[:, U_indices].add(delta_U)

    return result


def prepare_inpainting_state(
    initial_actions: jax.Array,
    noise: jax.Array,
    action_horizon: int,
    action_dim: int,
) -> dict:
    """Pre-compute all inpainting state needed by the denoising loop.

    Call this once before the loop begins.  Returns a dict with keys:
    ``O_indices``, ``U_indices``, ``x0_flat``, ``z_flat``,
    ``num_initial_steps``, ``input_action_dim``.
    """
    batch_size = initial_actions.shape[0]
    num_initial_steps = initial_actions.shape[1]
    input_action_dim = initial_actions.shape[2]

    padded = pad_initial_actions(initial_actions, action_horizon, action_dim)
    flat_dim = action_horizon * action_dim
    x0_flat = padded.reshape(batch_size, flat_dim)
    z_flat = noise.reshape(batch_size, flat_dim)

    O_indices, U_indices = build_inpainting_indices(
        num_initial_steps, input_action_dim, action_horizon, action_dim,
    )

    return {
        "O_indices": O_indices,
        "U_indices": U_indices,
        "x0_flat": x0_flat,
        "z_flat": z_flat,
        "num_initial_steps": num_initial_steps,
        "input_action_dim": input_action_dim,
    }
