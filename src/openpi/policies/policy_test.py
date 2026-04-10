from unittest import mock

from openpi_client import action_chunk_broker
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


# ---------------------------------------------------------------------------
# Unit tests for initial_actions forwarding through Policy.infer
# ---------------------------------------------------------------------------

def _make_fake_policy(action_horizon=4, action_dim=8):
    """Build a Policy with a fake model whose sample_actions captures kwargs."""
    captured = {}

    class FakeModel(_model.BaseModel):
        supports_initial_actions = True

        def compute_loss(self, rng, observation, actions, *, train=False):
            raise NotImplementedError

        def sample_actions(self, rng, observation, *, num_steps=10, noise=None, initial_actions=None):
            captured["noise"] = noise
            captured["initial_actions"] = initial_actions
            return jnp.zeros((1, action_horizon, action_dim))

    model = FakeModel(action_dim=action_dim, action_horizon=action_horizon, max_token_len=16)
    # Bypass JIT for testing: wire _sample_actions directly
    policy = _policy.Policy.__new__(_policy.Policy)
    policy._model = model
    policy._input_transform = lambda x: x
    policy._output_transform = lambda x: x
    policy._sample_kwargs = {}
    policy._metadata = {}
    policy._is_pytorch_model = False
    policy._pytorch_device = "cpu"
    policy._sample_actions = model.sample_actions
    policy._sample_actions_cfg = None
    policy._delta_actions_transform = None
    policy._action_normalizer = None
    import jax
    policy._rng = jax.random.key(0)
    return policy, captured


def test_policy_infer_forwards_initial_actions():
    policy, captured = _make_fake_policy(action_horizon=4, action_dim=8)
    obs = {
        "image": {"base_0_rgb": np.zeros((224, 224, 3), dtype=np.float32)},
        "image_mask": {"base_0_rgb": np.array(True)},
        "state": np.zeros(8, dtype=np.float32),
    }
    policy.infer(obs, initial_actions=np.ones((2, 8), dtype=np.float32))
    assert "initial_actions" in captured
    assert captured["initial_actions"] is not None
    assert captured["initial_actions"].shape == (1, 2, 8)


def test_policy_infer_batches_2d_initial_actions():
    policy, captured = _make_fake_policy(action_horizon=4, action_dim=8)
    obs = {
        "image": {"base_0_rgb": np.zeros((224, 224, 3), dtype=np.float32)},
        "image_mask": {"base_0_rgb": np.array(True)},
        "state": np.zeros(8, dtype=np.float32),
    }
    ia = np.ones((3, 8), dtype=np.float32)
    policy.infer(obs, initial_actions=ia)
    assert captured["initial_actions"].shape == (1, 3, 8)


def test_policy_infer_passes_3d_initial_actions_unchanged():
    policy, captured = _make_fake_policy(action_horizon=4, action_dim=8)
    obs = {
        "image": {"base_0_rgb": np.zeros((224, 224, 3), dtype=np.float32)},
        "image_mask": {"base_0_rgb": np.array(True)},
        "state": np.zeros(8, dtype=np.float32),
    }
    ia = np.ones((1, 3, 8), dtype=np.float32)
    policy.infer(obs, initial_actions=ia)
    assert captured["initial_actions"].shape == (1, 3, 8)


def test_policy_infer_initial_actions_and_noise_coexist():
    policy, captured = _make_fake_policy(action_horizon=4, action_dim=8)
    obs = {
        "image": {"base_0_rgb": np.zeros((224, 224, 3), dtype=np.float32)},
        "image_mask": {"base_0_rgb": np.array(True)},
        "state": np.zeros(8, dtype=np.float32),
    }
    ia = np.ones((2, 8), dtype=np.float32)
    noise = np.zeros((4, 8), dtype=np.float32)
    policy.infer(obs, initial_actions=ia, noise=noise)
    assert captured["initial_actions"] is not None
    assert captured["noise"] is not None


def test_policy_infer_no_initial_actions_passes_none():
    policy, captured = _make_fake_policy(action_horizon=4, action_dim=8)
    obs = {
        "image": {"base_0_rgb": np.zeros((224, 224, 3), dtype=np.float32)},
        "image_mask": {"base_0_rgb": np.array(True)},
        "state": np.zeros(8, dtype=np.float32),
    }
    policy.infer(obs)
    assert captured.get("initial_actions") is None


def test_policy_infer_rejects_initial_actions_on_unsupported_model():
    """Models without supports_initial_actions=True should raise ValueError."""

    class UnsupportedModel(_model.BaseModel):
        def compute_loss(self, rng, observation, actions, *, train=False):
            raise NotImplementedError

        def sample_actions(self, rng, observation, **kwargs):
            return jnp.zeros((1, 4, 8))

    model = UnsupportedModel(action_dim=8, action_horizon=4, max_token_len=16)
    policy = _policy.Policy.__new__(_policy.Policy)
    policy._model = model
    policy._input_transform = lambda x: x
    policy._output_transform = lambda x: x
    policy._sample_kwargs = {}
    policy._metadata = {}
    policy._is_pytorch_model = False
    policy._pytorch_device = "cpu"
    policy._sample_actions = model.sample_actions
    policy._sample_actions_cfg = None
    policy._delta_actions_transform = None
    policy._action_normalizer = None
    import jax
    policy._rng = jax.random.key(0)

    obs = {
        "image": {"base_0_rgb": np.zeros((224, 224, 3), dtype=np.float32)},
        "image_mask": {"base_0_rgb": np.array(True)},
        "state": np.zeros(8, dtype=np.float32),
    }
    with pytest.raises(ValueError, match="does not support initial_actions"):
        policy.infer(obs, initial_actions=np.ones((2, 8), dtype=np.float32))


def test_policy_infer_applies_delta_and_normalize_to_initial_actions():
    """initial_actions in output space should be delta-converted then normalized."""
    from openpi import transforms as _transforms
    from openpi.shared.normalize import NormStats

    action_horizon, action_dim = 4, 6
    captured = {}

    class FakeModel(_model.BaseModel):
        supports_initial_actions = True

        def compute_loss(self, rng, observation, actions, *, train=False):
            raise NotImplementedError

        def sample_actions(self, rng, observation, *, num_steps=10, noise=None, initial_actions=None):
            captured["initial_actions"] = initial_actions
            return jnp.zeros((1, action_horizon, action_dim))

    model = FakeModel(action_dim=action_dim, action_horizon=action_horizon, max_token_len=16)

    delta_mask = [True, True, True, True, True, False]
    norm_stats = {
        "actions": NormStats(mean=np.zeros(action_dim), std=np.ones(action_dim) * 2.0),
    }
    policy = _policy.Policy.__new__(_policy.Policy)
    policy._model = model
    policy._input_transform = lambda x: x
    policy._output_transform = lambda x: x
    policy._sample_kwargs = {}
    policy._metadata = {}
    policy._is_pytorch_model = False
    policy._pytorch_device = "cpu"
    policy._sample_actions = model.sample_actions
    policy._sample_actions_cfg = None
    policy._delta_actions_transform = _transforms.DeltaActions(delta_mask)
    policy._action_normalizer = _transforms.Normalize(norm_stats)
    import jax
    policy._rng = jax.random.key(0)

    state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    obs = {
        "image": {"base_0_rgb": np.zeros((224, 224, 3), dtype=np.float32)},
        "image_mask": {"base_0_rgb": np.array(True)},
        "state": state,
    }
    # initial_actions: absolute positions, 2 timesteps
    ia_absolute = np.array([
        [3.0, 4.0, 5.0, 6.0, 7.0, 10.0],
        [5.0, 6.0, 7.0, 8.0, 9.0, 20.0],
    ], dtype=np.float32)

    policy.infer(obs, initial_actions=ia_absolute)
    ia_model = np.asarray(captured["initial_actions"])

    # Expected: delta = absolute - state (for masked dims), then normalized / 2.0
    # Dim 0: (3-1)/2=1.0, dim 1: (4-2)/2=1.0, ..., dim 4: (7-5)/2=1.0
    # Dim 5 (not masked): (10-0)/2=5.0 (no delta, just normalize)
    expected_t0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 5.0])
    expected_t1 = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 10.0])

    assert ia_model.shape == (1, 2, 6)
    np.testing.assert_allclose(ia_model[0, 0], expected_t0, atol=1e-5)
    np.testing.assert_allclose(ia_model[0, 1], expected_t1, atol=1e-5)


def test_policy_infer_handles_per_timestep_norm_with_fewer_initial_steps():
    """Per-timestep stats (2D mean/std) must be sliced to match initial_actions timesteps."""
    from openpi import transforms as _transforms
    from openpi.shared.normalize import NormStats

    action_horizon, action_dim = 4, 3
    captured = {}

    class FakeModel(_model.BaseModel):
        supports_initial_actions = True

        def compute_loss(self, rng, observation, actions, *, train=False):
            raise NotImplementedError

        def sample_actions(self, rng, observation, *, num_steps=10, noise=None, initial_actions=None):
            captured["initial_actions"] = initial_actions
            return jnp.zeros((1, action_horizon, action_dim))

    model = FakeModel(action_dim=action_dim, action_horizon=action_horizon, max_token_len=16)

    # Per-timestep stats: each timestep has different mean/std
    per_ts_mean = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
    ], dtype=np.float32)
    per_ts_std = np.ones((action_horizon, action_dim), dtype=np.float32) * 2.0

    norm_stats = {
        "actions": NormStats(mean=per_ts_mean, std=per_ts_std),
    }
    policy = _policy.Policy.__new__(_policy.Policy)
    policy._model = model
    policy._input_transform = lambda x: x
    policy._output_transform = lambda x: x
    policy._sample_kwargs = {}
    policy._metadata = {}
    policy._is_pytorch_model = False
    policy._pytorch_device = "cpu"
    policy._sample_actions = model.sample_actions
    policy._sample_actions_cfg = None
    policy._delta_actions_transform = None
    policy._action_normalizer = _transforms.Normalize(norm_stats)
    import jax
    policy._rng = jax.random.key(0)

    obs = {
        "image": {"base_0_rgb": np.zeros((224, 224, 3), dtype=np.float32)},
        "image_mask": {"base_0_rgb": np.array(True)},
        "state": np.zeros(action_dim, dtype=np.float32),
    }
    # 2 timesteps out of action_horizon=4
    ia = np.array([
        [4.0, 4.0, 4.0],
        [5.0, 5.0, 5.0],
    ], dtype=np.float32)

    policy.infer(obs, initial_actions=ia)
    ia_model = np.asarray(captured["initial_actions"])

    assert ia_model.shape == (1, 2, 3)
    # t=0: (4 - 0) / 2 = 2.0, t=1: (5 - 1) / 2 = 2.0
    np.testing.assert_allclose(ia_model[0, 0], [2.0, 2.0, 2.0], atol=1e-5)
    np.testing.assert_allclose(ia_model[0, 1], [2.0, 2.0, 2.0], atol=1e-5)


# ---------------------------------------------------------------------------
# Manual integration tests (require checkpoints)
# ---------------------------------------------------------------------------

@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
