from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy

# Schedules for the RTC soft overlap mask, matching LeRobot's RTCAttentionSchedule.
RTC_PREFIX_SCHEDULES = ("linear", "exp", "ones", "zeros")


def get_rtc_prefix_weights(start: int, end: int, total: int, schedule: str) -> np.ndarray:
    """Build the RTC soft overlap mask over the action chunk.

    Direct numpy port of LeRobot's `RTCProcessor.get_prefix_weights`. `start` is the
    inference delay (full-weight region), `end` is the execution horizon (weights
    decay to zero by here), and `total` is the action horizon. Computed host-side so
    the jitted denoiser takes a fixed-shape float array instead of dynamic ints.
    """
    start = min(start, end)

    if schedule == "zeros":
        weights = np.zeros(total, dtype=np.float32)
        weights[:start] = 1.0
        return weights
    if schedule == "ones":
        weights = np.ones(total, dtype=np.float32)
        weights[end:] = 0.0
        return weights
    if schedule not in ("linear", "exp"):
        raise ValueError(f"Unknown RTC prefix_attention_schedule: {schedule!r}")

    # linear / exp share the same linear ramp; exp reshapes it.
    skip_steps_at_end = max(total - end, 0)
    linspace_steps = total - skip_steps_at_end - start
    if end <= start or linspace_steps <= 0:
        lin = np.empty(0, dtype=np.float32)
    else:
        lin = np.linspace(1.0, 0.0, linspace_steps + 2, dtype=np.float32)[1:-1]
        if schedule == "exp":
            lin = lin * np.expm1(lin) / (np.e - 1.0)

    trailing = max(total - end, 0)
    if trailing > 0:
        lin = np.concatenate([lin, np.zeros(trailing, dtype=np.float32)])
    leading = min(start, total)
    if leading > 0:
        lin = np.concatenate([np.ones(leading, dtype=np.float32), lin])
    return lin.astype(np.float32)


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            self._sample_actions_cfg = getattr(model, "sample_actions_cfg", None)
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            _rtc_fn = getattr(model, "sample_actions_rtc", None)
            self._sample_actions_rtc = nnx_utils.module_jit(_rtc_fn) if _rtc_fn is not None else None
            _cfg_fn = getattr(model, "sample_actions_cfg", None)
            self._sample_actions_cfg = nnx_utils.module_jit(_cfg_fn) if _cfg_fn is not None else None
            self._rng = rng or jax.random.key(0)

    def _prepare_inputs(self, obs: dict):
        """Transform, batch, and prepare an observation dict for the model."""
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        else:
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
        return inputs

    @override
    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        uncond_obs: dict | None = None,
        guidance_scale: float | None = None,
    ) -> dict:  # type: ignore[misc]
        inputs = self._prepare_inputs(obs)

        if not self._is_pytorch_model:
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            sample_rng_or_pytorch_device = self._pytorch_device

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)
            if noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()

        use_cfg = uncond_obs is not None and guidance_scale is not None
        if use_cfg:
            if self._sample_actions_cfg is None:
                raise ValueError("Model does not support sample_actions_cfg")
            uncond_inputs = self._prepare_inputs(uncond_obs)
            uncond_observation = _model.Observation.from_dict(uncond_inputs)
            action_output = self._sample_actions_cfg(
                sample_rng_or_pytorch_device,
                observation,
                uncond_observation,
                guidance_scale=guidance_scale,
                **sample_kwargs,
            )
        else:
            action_output = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)

        if isinstance(action_output, tuple):
            actions = action_output[0]
            output_tokens = action_output[1]
            tokenizer = _tokenizer.PaligemmaTokenizer(max_len=50)
            output_tokens = jnp.array(output_tokens, dtype=int)
            subtask_text = tokenizer.detokenize(output_tokens[0])
            print(f"\n{'#' * 60}\n###  GENERATED SUBTASK: {subtask_text}\n{'#' * 60}")
        else:
            actions = action_output
        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def infer_rtc(
        self,
        obs: dict,
        *,
        prefix_actions: np.ndarray | None,
        inference_delay: int,
        prefix_attention_horizon: int,
        max_guidance_weight: float,
        prefix_attention_schedule: str = "exp",
    ) -> dict:  # type: ignore[misc]
        """Real-Time Chunking inference (JAX models only).

        `prefix_actions` is the previous chunk's unexecuted tail, **already in this
        model's normalized action space** (i.e. the raw `actions` array returned
        under the "actions_raw" key by a previous call). Shape `(T_left, action_dim)`.
        Pass `None` for the very first chunk: this falls back to plain sampling but
        still returns "actions_raw" so the caller can seed its queue.

        Returns the usual transformed (robot-space) "actions" plus "actions_raw":
        the un-transformed normalized chunk to feed back as the next prefix.
        """
        if self._is_pytorch_model:
            raise NotImplementedError("infer_rtc is only implemented for JAX models")
        if prefix_actions is not None and self._sample_actions_rtc is None:
            raise ValueError("Model does not support sample_actions_rtc (RTC requires a pi0.5 model)")

        inputs = self._prepare_inputs(obs)
        observation = _model.Observation.from_dict(inputs)

        start_time = time.monotonic()
        self._rng, sample_rng = jax.random.split(self._rng)

        if prefix_actions is None:
            # First chunk: no previous tail to blend with -> plain sampling.
            action_output = self._sample_actions(sample_rng, observation)
        else:
            action_horizon = int(self._model.action_horizon)
            action_dim = int(self._model.action_dim)
            prefix = np.asarray(prefix_actions, dtype=np.float32)
            t_left = prefix.shape[0]
            # Don't guide further than we have previous actions to blend with.
            horizon = min(prefix_attention_horizon, t_left)
            weights = get_rtc_prefix_weights(
                inference_delay, horizon, action_horizon, prefix_attention_schedule
            )
            # Pad/truncate the prefix to the full action horizon (zeros are masked
            # out by the weights beyond the overlap region).
            padded = np.zeros((action_horizon, action_dim), dtype=np.float32)
            usable = min(t_left, action_horizon)
            padded[:usable] = prefix[:usable, :action_dim]
            action_output = self._sample_actions_rtc(
                sample_rng,
                observation,
                jnp.asarray(padded)[jnp.newaxis, ...],
                jnp.asarray(weights),
                jnp.asarray(max_guidance_weight, dtype=jnp.float32),
            )

        if isinstance(action_output, tuple):
            actions = action_output[0]
            output_tokens = action_output[1]
            tokenizer = _tokenizer.PaligemmaTokenizer(max_len=50)
            output_tokens = jnp.array(output_tokens, dtype=int)
            subtask_text = tokenizer.detokenize(output_tokens[0])
            print(f"\n{'#' * 60}\n###  GENERATED SUBTASK: {subtask_text}\n{'#' * 60}")
        else:
            actions = action_output

        # Keep the raw (normalized, full action_dim) chunk to feed back as the next
        # prefix; the output transform below would un-normalize and slice it.
        actions_raw = np.asarray(actions[0, ...])

        outputs = {"state": inputs["state"], "actions": actions}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        model_time = time.monotonic() - start_time

        outputs = self._output_transform(outputs)
        outputs["actions_raw"] = actions_raw
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
