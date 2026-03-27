import math

import jax.numpy as jnp

from openpi.models import pi0_rl


def _token_only_decoder(
    rl_token: jnp.ndarray,
    target_embeddings: jnp.ndarray,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    del mask
    return jnp.broadcast_to(rl_token[:, None, :], target_embeddings.shape)


def test_reconstruction_ablation_metrics_detects_zero_and_shuffled_tokens():
    rl_token = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    target_embeddings = jnp.broadcast_to(rl_token[:, None, :], (2, 2, 2))
    mask = jnp.ones((2, 2), dtype=jnp.bool_)

    metrics = pi0_rl.compute_reconstruction_ablation_metrics(
        _token_only_decoder,
        rl_token,
        target_embeddings,
        mask,
        shuffle_perm=jnp.array([1, 0]),
    )

    assert math.isclose(metrics["real_recon_loss"], 0.0)
    assert math.isclose(metrics["zero_recon_loss"], 15.0)
    assert math.isclose(metrics["shuffled_recon_loss"], 8.0)
    assert math.isclose(metrics["zero_recon_gap"], 15.0)
    assert math.isclose(metrics["shuffled_recon_gap"], 8.0)
