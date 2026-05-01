import flax.nnx as nnx
import jax
import jax.numpy as jnp
from openpi.models.pi0 import Pi0
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


def test_pi0_cpg_velocity_combination():
    v_positive = jnp.array([[2.0, 4.0]])
    v_negative = jnp.array([[1.0, 3.0]])

    result = Pi0._combine_cpg_velocity(v_positive, v_negative, 2.5)

    assert jnp.allclose(result, jnp.array([[3.5, 5.5]]))


def test_pi0_sample_actions_cpg_uses_positive_and_negative_prompts():
    config = _pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,
        action_horizon=4,
        max_token_len=16,
    )
    model = config.create(jax.random.key(0))
    positive_obs = config.fake_obs(batch_size=1)
    negative_obs = positive_obs.replace(tokenized_prompt=jnp.full_like(positive_obs.tokenized_prompt, 9))
    positive_prefix_mask = jnp.array([[True, True, True]])
    negative_prefix_mask = jnp.array([[True, False, False]])
    expected_actions = jnp.full((1, config.action_horizon, config.action_dim), 2.0)

    def build_prefix_cache(observation):
        if jnp.array_equal(observation.tokenized_prompt, positive_obs.tokenized_prompt):
            return "positive-cache", positive_prefix_mask
        assert jnp.array_equal(observation.tokenized_prompt, negative_obs.tokenized_prompt)
        return "negative-cache", negative_prefix_mask

    model.build_prefix_cache = build_prefix_cache

    def denoise_actions(observation, prefix_mask, kv_cache, *, num_steps, noise, velocity_fn):
        assert jnp.array_equal(observation.tokenized_prompt, positive_obs.tokenized_prompt)
        assert kv_cache == "positive-cache"
        assert jnp.array_equal(prefix_mask, positive_prefix_mask)
        assert num_steps == 7
        assert noise.shape == (1, config.action_horizon, config.action_dim)
        assert velocity_fn is not None
        return expected_actions

    model._denoise_actions = denoise_actions

    actions = model.sample_actions_cpg(jax.random.key(1), positive_obs, negative_obs, guidance_scale=2.0, num_steps=7)

    assert jnp.array_equal(actions, expected_actions)


