from openpi_client import action_chunk_broker
import numpy as np
import pytest
import torch

from openpi.policies import aloha_policy
from openpi.policies import policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def _minimal_obs(prompt_token: int = 1):
    image = np.zeros((2, 2, 3), dtype=np.float32)
    return {
        "image": {
            "base_0_rgb": image,
            "left_wrist_0_rgb": image,
            "right_wrist_0_rgb": image,
        },
        "image_mask": {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.False_,
        },
        "state": np.zeros((3,), dtype=np.float32),
        "tokenized_prompt": np.array([prompt_token, 0], dtype=np.int32),
        "tokenized_prompt_mask": np.array([True, False]),
    }


def test_policy_routes_cpg_separately_from_cfg():
    class _Model:
        def to(self, device):
            return self

        def eval(self):
            pass

        def sample_actions(self, *args, **kwargs):
            raise AssertionError("plain sampling should not be used")

        def sample_actions_cfg(self, *args, **kwargs):
            raise AssertionError("CFG should not be used for CPG")

        def sample_actions_cpg(self, device, observation, negative_observation, *, guidance_scale, **kwargs):
            assert device == "cpu"
            assert guidance_scale == 2.0
            assert not torch.equal(observation.tokenized_prompt, negative_observation.tokenized_prompt)
            return torch.ones((1, 2, 3))

    p = policy.Policy(_Model(), is_pytorch=True)

    result = p.infer(_minimal_obs(1), negative_obs=_minimal_obs(2), cpg_guidance_scale=2.0)

    assert result["actions"].shape == (2, 3)


def test_policy_requires_complete_cpg_arguments():
    class _Model:
        def to(self, device):
            return self

        def eval(self):
            pass

        def sample_actions(self, *args, **kwargs):
            raise AssertionError("plain sampling should not be used")

    p = policy.Policy(_Model(), is_pytorch=True)

    with pytest.raises(ValueError, match="CPG requires both"):
        p.infer(_minimal_obs(1), negative_obs=_minimal_obs(2))


def test_policy_requires_distinct_cpg_prompts():
    class _Model:
        def to(self, device):
            return self

        def eval(self):
            pass

        def sample_actions(self, *args, **kwargs):
            raise AssertionError("plain sampling should not be used")

        def sample_actions_cpg(self, *args, **kwargs):
            raise AssertionError("CPG should not run with identical prompts")

    p = policy.Policy(_Model(), is_pytorch=True)

    with pytest.raises(ValueError, match="distinct positive and negative prompts"):
        p.infer(_minimal_obs(1), negative_obs=_minimal_obs(1), cpg_guidance_scale=2.0)


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
