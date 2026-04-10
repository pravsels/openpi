import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


# ---------------------------------------------------------------------------
# initial_actions plumbing tests for Pi0
# ---------------------------------------------------------------------------

def _make_pi0_config():
    return _pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,
        action_horizon=4,
        max_token_len=16,
    )


def test_pi0_sample_actions_forwards_initial_actions_to_denoise():
    config = _make_pi0_config()
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=1)
    noise = jnp.full((1, config.action_horizon, config.action_dim), -1.0)
    ia = jnp.ones((1, 2, config.action_dim), dtype=jnp.float32)
    expected_actions = jnp.full((1, config.action_horizon, config.action_dim), 2.0)

    captured = {}

    def fake_denoise(observation, prefix_mask, kv_cache, *, num_steps, noise, velocity_fn=None, initial_actions=None):
        captured["initial_actions"] = initial_actions
        return expected_actions

    model._denoise_actions = fake_denoise
    model.build_prefix_cache = lambda obs: ("cache", jnp.array([[True, True]]))

    result = model.sample_actions(jax.random.key(1), obs, num_steps=7, noise=noise, initial_actions=ia)
    assert captured["initial_actions"] is not None
    assert captured["initial_actions"].shape == (1, 2, config.action_dim)


def test_pi0_sample_actions_passes_none_when_no_initial_actions():
    config = _make_pi0_config()
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=1)
    noise = jnp.full((1, config.action_horizon, config.action_dim), -1.0)
    expected_actions = jnp.full((1, config.action_horizon, config.action_dim), 2.0)

    captured = {}

    def fake_denoise(observation, prefix_mask, kv_cache, *, num_steps, noise, velocity_fn=None, initial_actions=None):
        captured["initial_actions"] = initial_actions
        return expected_actions

    model._denoise_actions = fake_denoise
    model.build_prefix_cache = lambda obs: ("cache", jnp.array([[True, True]]))

    model.sample_actions(jax.random.key(1), obs, num_steps=7, noise=noise)
    assert captured["initial_actions"] is None


def test_pi0_validates_initial_actions_dim():
    config = _make_pi0_config()
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=1)
    noise = jnp.full((1, config.action_horizon, config.action_dim), -1.0)

    wrong_dim_ia = jnp.ones((1, 2, config.action_dim + 5), dtype=jnp.float32)
    with pytest.raises(ValueError, match="action_dim"):
        model.sample_actions(jax.random.key(1), obs, noise=noise, initial_actions=wrong_dim_ia)


def test_pi0_validates_initial_actions_horizon():
    config = _make_pi0_config()
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=1)
    noise = jnp.full((1, config.action_horizon, config.action_dim), -1.0)

    too_long_ia = jnp.ones((1, config.action_horizon + 1, config.action_dim), dtype=jnp.float32)
    with pytest.raises(ValueError, match="action_horizon"):
        model.sample_actions(jax.random.key(1), obs, noise=noise, initial_actions=too_long_ia)


def test_pi0_sample_actions_cfg_with_initial_actions():
    config = _make_pi0_config()
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=1)
    ia = jnp.ones((1, 2, config.action_dim), dtype=jnp.float32)

    result = model.sample_actions_cfg(
        jax.random.key(1), obs, obs, guidance_scale=2.0, initial_actions=ia,
    )
    assert result.shape == (1, config.action_horizon, config.action_dim)


def test_pi0_denoise_actions_end_to_end_with_initial_actions():
    """Run _denoise_actions with initial_actions through the full while_loop.

    Verifies that:
    1. The JAX tracing succeeds (no ConcretizationTypeError)
    2. Constrained coordinates converge toward x0 at the end
    """
    config = _make_pi0_config()
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=1)

    noise = jax.random.normal(jax.random.key(42), (1, config.action_horizon, config.action_dim))
    ia = jnp.ones((1, 2, config.action_dim), dtype=jnp.float32) * 3.0

    kv_cache, prefix_mask = model.build_prefix_cache(obs)
    result = model._denoise_actions(
        obs, prefix_mask, kv_cache,
        num_steps=10, noise=noise, initial_actions=ia,
    )
    assert result.shape == (1, config.action_horizon, config.action_dim)
    # The constrained timesteps (first 2) should be close to the
    # initial_actions target (3.0) at t=0.  With threshold=0.0 the
    # constraint releases at the very last step, so there may be a
    # small deviation O(dt), but it should be much closer to 3.0 than
    # to the noise.
    constrained_region = result[0, :2, :]
    assert jnp.allclose(constrained_region, 3.0, atol=0.5)
