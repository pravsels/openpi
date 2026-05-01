import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from openpi.models.pi05 import Pi05
import openpi.models.pi05_config as _pi05_config
import openpi.shared.nnx_utils as nnx_utils


def _state_l2_norm(state: nnx.State) -> jax.Array:
    squared_norms = []
    for leaf in state.flat_state().values():
        value = leaf.value if hasattr(leaf, "value") else leaf
        squared_norms.append(jnp.sum(jnp.square(value)))
    if not squared_norms:
        return jnp.array(0.0)
    return jnp.sqrt(jnp.sum(jnp.stack(squared_norms)))


def _compute_flow_grad_norms(config: _pi05_config.Pi05Config) -> tuple[jax.Array, jax.Array]:
    model = config.create(jax.random.key(0))

    obs, _ = config.fake_obs(batch_size=1), config.fake_act(batch_size=1)
    obs = obs.replace(  # avoid degenerate all-ones token inputs
        tokenized_prompt=jax.random.randint(jax.random.key(1), obs.tokenized_prompt.shape, minval=0, maxval=128)
    )
    actions = jax.random.normal(jax.random.key(2), (1, config.action_horizon, config.action_dim))

    def loss_fn(model):
        return jnp.mean(model.compute_loss(jax.random.key(3), obs, actions, train=True))

    diff_state = nnx.DiffState(0, nnx.Param)
    _, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model)

    non_action_expert_filter = nnx.All(
        nnx_utils.PathRegex(".*llm.*"),
        nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")),
    )
    action_expert_filter = nnx_utils.PathRegex(".*llm.*_1.*")

    non_action_expert_grad_norm = _state_l2_norm(grads.filter(non_action_expert_filter))
    action_expert_grad_norm = _state_l2_norm(grads.filter(action_expert_filter))
    return non_action_expert_grad_norm, action_expert_grad_norm


def test_pi05_flow_prefix_stop_gradient():
    base_config = _pi05_config.Pi05Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,
        action_horizon=4,
        max_token_len=16,
        subtask_loss_weight=0.0,
        fast_token_loss_weight=0.0,
        flow_matching_loss_weight=1.0,
        stop_gradient_flow_to_prefix=False,
    )
    insulated_config = dataclasses.replace(base_config, stop_gradient_flow_to_prefix=True)

    non_action_grad, action_grad = _compute_flow_grad_norms(base_config)
    insulated_non_action_grad, insulated_action_grad = _compute_flow_grad_norms(insulated_config)

    assert non_action_grad > 0
    assert insulated_non_action_grad == 0
    assert action_grad > 0
    assert insulated_action_grad > 0


def test_pi05_sample_actions_cfg_guidance_scale_one_delegates():
    config = _pi05_config.Pi05Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,
        action_horizon=4,
        max_token_len=16,
        subtask_loss_weight=0.0,
        fast_token_loss_weight=0.0,
        flow_matching_loss_weight=1.0,
    )
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=1)

    expected_actions = jnp.ones((1, config.action_horizon, config.action_dim))
    expected_tokens = jnp.array([[1, 2, 0]], dtype=jnp.int32)

    model.sample_actions = lambda rng, observation, *, num_steps=10, noise=None: (expected_actions, expected_tokens)

    actions, tokens = model.sample_actions_cfg(jax.random.key(1), obs, obs, guidance_scale=1.0)

    assert jnp.array_equal(actions, expected_actions)
    assert jnp.array_equal(tokens, expected_tokens)


def test_pi05_sample_actions_rebuilds_action_prefix_cache_from_output_tokens():
    config = _pi05_config.Pi05Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,
        action_horizon=4,
        max_token_len=16,
        subtask_loss_weight=0.0,
        fast_token_loss_weight=0.0,
        flow_matching_loss_weight=1.0,
    )
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=1)

    expected_tokens = jnp.array([[7, 3, 0]], dtype=jnp.int32)
    rebuilt_prefix_mask = jnp.array([[True, True, True, False]])
    expected_actions = jnp.full((1, config.action_horizon, config.action_dim), 2.0)
    noise = jnp.full((1, config.action_horizon, config.action_dim), -1.0)

    model.sample_low_level_task = (
        lambda rng, observation, *, max_decoding_steps, paligemma_eos_token, temperature: (
            expected_tokens,
            "stale-cache",
            jnp.array([[True, False, False, False]]),
            None,
        )
    )

    def rebuild_cache(observation, output_tokens, *, append_action_prompt=True):
        assert append_action_prompt is True
        assert jnp.array_equal(output_tokens, expected_tokens)
        return "rebuilt-cache", rebuilt_prefix_mask

    model.build_prefix_cache_with_generated_subtask = rebuild_cache

    def sample_with_cache(observation, kv_cache, prefix_mask, *, num_steps, noise):
        assert kv_cache == "rebuilt-cache"
        assert jnp.array_equal(prefix_mask, rebuilt_prefix_mask)
        assert num_steps == 7
        assert jnp.array_equal(noise, jnp.full((1, config.action_horizon, config.action_dim), -1.0))
        return expected_actions

    model._sample_actions_with_prefix_cache = sample_with_cache

    actions, tokens = model.sample_actions(jax.random.key(1), obs, num_steps=7, noise=noise)

    assert jnp.array_equal(actions, expected_actions)
    assert jnp.array_equal(tokens, expected_tokens)


def test_pi05_cfg_velocity_combination():
    v_cond = jnp.array([[2.0, 4.0]])
    v_uncond = jnp.array([[1.0, 3.0]])

    result = Pi05._combine_cfg_velocity(v_cond, v_uncond, 2.5)

    assert jnp.allclose(result, jnp.array([[3.5, 5.5]]))


def test_pi05_sample_actions_cfg_reuses_conditional_subtask_tokens():
    config = _pi05_config.Pi05Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,
        action_horizon=4,
        max_token_len=16,
        subtask_loss_weight=0.0,
        fast_token_loss_weight=0.0,
        flow_matching_loss_weight=1.0,
    )
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=1)
    uncond_obs = config.fake_obs(batch_size=1).replace(
        tokenized_prompt=obs.tokenized_prompt,
        tokenized_prompt_mask=obs.tokenized_prompt_mask,
        action_tokenized_prompt=jnp.full_like(obs.action_tokenized_prompt, 9),
    )

    expected_tokens = jnp.array([[7, 3, 0]], dtype=jnp.int32)
    expected_actions = jnp.full((1, config.action_horizon, config.action_dim), 2.0)
    cond_prefix_mask = jnp.array([[True, True, True, False]])
    uncond_prefix_mask = jnp.array([[True, False, False, False]])
    noise = jnp.full((1, config.action_horizon, config.action_dim), -1.0)

    model.sample_low_level_task = (
        lambda rng, observation, *, max_decoding_steps, paligemma_eos_token, temperature: (
            expected_tokens,
            "stale-cond-cache",
            jnp.array([[True, False, False, False]]),
            None,
        )
    )

    def build_cache(observation, output_tokens, *, append_action_prompt=True):
        assert append_action_prompt is True
        assert jnp.array_equal(output_tokens, expected_tokens)
        if jnp.array_equal(observation.action_tokenized_prompt, obs.action_tokenized_prompt):
            return "cond-cache", cond_prefix_mask
        assert jnp.array_equal(observation.action_tokenized_prompt, uncond_obs.action_tokenized_prompt)
        return "uncond-cache", uncond_prefix_mask

    model.build_prefix_cache_with_generated_subtask = build_cache

    def sample_cfg_from_caches(
        observation,
        cond_kv_cache,
        cond_prefix_mask_arg,
        uncond_observation,
        uncond_kv_cache,
        uncond_prefix_mask_arg,
        *,
        guidance_scale,
        num_steps,
        noise: jax.Array,
    ):
        assert cond_kv_cache == "cond-cache"
        assert uncond_kv_cache == "uncond-cache"
        assert jnp.array_equal(cond_prefix_mask_arg, cond_prefix_mask)
        assert jnp.array_equal(uncond_prefix_mask_arg, uncond_prefix_mask)
        assert guidance_scale == 2.0
        assert num_steps == 7
        assert jnp.array_equal(noise, jnp.full((1, config.action_horizon, config.action_dim), -1.0))
        return expected_actions

    model._sample_actions_cfg_with_prefix_caches = sample_cfg_from_caches

    actions, tokens = model.sample_actions_cfg(
        jax.random.key(1), obs, uncond_obs, guidance_scale=2.0, num_steps=7, noise=noise
    )

    assert jnp.array_equal(actions, expected_actions)
    assert jnp.array_equal(tokens, expected_tokens)


def test_pi05_sample_actions_cfg_requires_post_subtask_action_prompts():
    config = _pi05_config.Pi05Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,
        action_horizon=4,
        max_token_len=16,
        subtask_loss_weight=0.0,
        fast_token_loss_weight=0.0,
        flow_matching_loss_weight=1.0,
    )
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=1).replace(
        action_tokenized_prompt=None,
        action_tokenized_prompt_mask=None,
    )

    model.sample_low_level_task = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not decode"))

    with pytest.raises(ValueError, match="action_tokenized_prompt"):
        model.sample_actions_cfg(jax.random.key(1), obs, obs, guidance_scale=2.0)


def test_pi05_sample_actions_cfg_requires_shared_prefix_prompt():
    config = _pi05_config.Pi05Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,
        action_horizon=4,
        max_token_len=16,
        subtask_loss_weight=0.0,
        fast_token_loss_weight=0.0,
        flow_matching_loss_weight=1.0,
    )
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=1)
    uncond_obs = obs.replace(tokenized_prompt=jnp.full_like(obs.tokenized_prompt, 9))

    model.sample_low_level_task = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not decode"))

    with pytest.raises(ValueError, match="share the same tokenized_prompt"):
        model.sample_actions_cfg(jax.random.key(1), obs, uncond_obs, guidance_scale=2.0)
