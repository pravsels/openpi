import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model as _model
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


def test_pi0_inputs_spec_uses_custom_image_keys():
    keys = ("base_0_rgb", "left_wrist_0_rgb", "base_1_rgb")
    observation_spec, _ = _pi0_config.Pi0Config(image_keys=keys).inputs_spec()
    assert tuple(observation_spec.images) == keys
    assert tuple(observation_spec.image_masks) == keys


def test_configured_image_keys_survive_jax_pytree_sort():
    keys = ("base_0_rgb", "left_wrist_0_rgb", "base_1_rgb")
    height, width = _model.IMAGE_RESOLUTION
    observation = _model.Observation(
        images={key: jnp.zeros((1, height, width, 3), dtype=jnp.float32) for key in keys},
        image_masks={key: jnp.ones((1,), dtype=jnp.bool_) for key in keys},
        state=jnp.zeros((1, 32), dtype=jnp.float32),
    )
    sorted_observation = jax.tree.map(lambda value: value, observation)
    assert tuple(sorted_observation.images) == ("base_0_rgb", "base_1_rgb", "left_wrist_0_rgb")
    assert _model.configured_image_keys(sorted_observation, keys) == list(keys)
    processed = _model.preprocess_observation(
        None, sorted_observation, train=False, image_keys=_model.configured_image_keys(sorted_observation, keys)
    )
    assert tuple(processed.images) == keys


def test_pi0_stores_config_image_keys():
    keys = ("base_0_rgb", "left_wrist_0_rgb", "base_1_rgb")
    model = nnx.eval_shape(_pi0_config.Pi0Config(image_keys=keys).create, jax.random.key(0))
    assert tuple(model.image_keys) == keys
