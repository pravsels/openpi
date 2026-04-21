"""Data transforms for build_block_tower datasets (LeRobot v2.1 format).

Handles the v2.1 key layout where state and actions are both 7D joint-space:
  - observation.state (7D joints)
  - action (7D joints)
  - observation.images.front, observation.images.wrist

State and action dimensions match so delta actions (action - state) are valid.
"""

import dataclasses

import numpy as np

from openpi import transforms
from openpi.policies import bin_pack_policy

_LOGGED_PROMPT = False
_RAW_DIM = 7
_CANONICAL_DIM = 17
_CANONICAL_MASK = np.array([True] * _RAW_DIM + [False] * (_CANONICAL_DIM - _RAW_DIM), dtype=bool)
_SEMANTIC_MASK = np.ones(_CANONICAL_DIM, dtype=bool)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        import einops
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _get_key(data: dict, *keys: str):
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"Missing keys: {keys}")


def _to_canonical_17d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] == _CANONICAL_DIM:
        return values
    if values.shape[-1] != _RAW_DIM:
        raise ValueError(f"Expected last dim {_RAW_DIM} or {_CANONICAL_DIM}, got {values.shape[-1]}")
    return transforms.pad_to_dim(values, _CANONICAL_DIM).astype(np.float32)


def _try_get_key(data: dict, *keys: str):
    for key in keys:
        if key in data:
            return data[key]
    return None


def _copy_passthrough_metadata(data: dict, inputs: dict) -> None:
    for key in ("control_mode", "advantage_label"):
        if key in data:
            inputs[key] = data[key]


def _append_gripper(eef_pose: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    eef_pose = np.asarray(eef_pose, dtype=np.float32)
    if eef_pose.shape[-1] == 7:
        return eef_pose
    if eef_pose.shape[-1] != 6:
        raise ValueError(f"Expected eef pose last dim 6 or 7, got {eef_pose.shape[-1]}")
    return np.concatenate([eef_pose, np.asarray(gripper, dtype=np.float32)], axis=-1).astype(np.float32)


def _semantic_state_and_actions(state: np.ndarray, actions: np.ndarray, data: dict) -> tuple[np.ndarray, np.ndarray] | None:
    state_eef = _try_get_key(
        data,
        "observation/state/eef_pose",
        "observation.state.eef_pose",
        "observation/eef_6d_pose",
        "observation.eef_6d_pose",
    )
    action_eef = _try_get_key(data, "action/eef_pose", "action.eef_pose")

    if state_eef is None or action_eef is None:
        return None

    state_eef = _append_gripper(state_eef, state[..., 6:7])
    action_eef = _append_gripper(action_eef, actions[..., 6:7])

    semantic_state = np.concatenate([state, bin_pack_policy._eef_pose_rpy_to_rot6d(state_eef)], axis=-1).astype(np.float32)
    semantic_actions = np.concatenate([actions, bin_pack_policy._eef_pose_rpy_to_rot6d(action_eef)], axis=-1).astype(np.float32)
    return semantic_state, semantic_actions


@dataclasses.dataclass(frozen=True)
class BlockTowerInputs(transforms.DataTransformFn):
    """Inputs for build_block_tower datasets.

    Raw state/actions are 7D joint-space. When EEF pose fields are available,
    they are lifted into the repo-standard semantic 17D layout:
    joints(7) + xyz(3) + rot6d(6) + gripper(1). Otherwise the data falls back
    to a padded 17D compatibility path with a dimension mask.

    When ``joints_only=True``, the EEF semantic lift is bypassed regardless of
    whether EEF pose fields are present: state/actions are padded to 17D, the
    EEF channels of state are zeroed, and ``action_dim_mask`` is forced to the
    joints-only mask so the flow-matching loss never touches the EEF dims.
    """

    default_prompt: str = "build a block tower"
    model_type: object | None = None
    joints_only: bool = False

    def __call__(self, data: dict) -> dict:
        front = _parse_image(_get_key(data, "observation/images/front", "observation.images.front"))
        wrist = _parse_image(_get_key(data, "observation/images/wrist", "observation.images.wrist"))

        state = _to_canonical_17d(
            _get_key(data, "observation/state", "observation.state", "observation/state/pos", "observation.state.pos")
        )

        actions = _to_canonical_17d(_get_key(data, "action", "actions"))
        if self.joints_only:
            # In joints-only mode, keep the 17D shape but zero all non-joint channels.
            state = np.array(state, copy=True)
            state[..., _RAW_DIM:] = 0.0
            actions = np.array(actions, copy=True)
            actions[..., _RAW_DIM:] = 0.0
            # Train loss only on canonical joint dims.
            action_dim_mask = np.array(_CANONICAL_MASK, copy=True)
        else:
            # Prefer semantic 17D lift (joints + xyz + rot6d + gripper) when fields exist.
            semantic = _semantic_state_and_actions(state[:_RAW_DIM], actions[..., :_RAW_DIM], data)
            if semantic is not None:
                state, actions = semantic
                # Semantic layout marks valid EEF channels explicitly.
                action_dim_mask = np.array(_SEMANTIC_MASK, copy=True)
            else:
                # Fallback: padded canonical layout (joints valid, remaining dims ignored in loss).
                action_dim_mask = np.array(_CANONICAL_MASK, copy=True)

        inputs = {
            "state": state,
            "actions": actions,
            "action_dim_mask": action_dim_mask,
            "image": {
                "base_0_rgb": front,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(front),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        _copy_passthrough_metadata(data, inputs)

        for pad_key in ("action_is_pad", "action.pos_is_pad", "action/pos_is_pad"):
            if pad_key in data:
                inputs["action_is_pad"] = np.asarray(data[pad_key]).astype(bool)
                break

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        elif "task" in data:
            inputs["prompt"] = str(data["task"])
        else:
            inputs["prompt"] = self.default_prompt

        global _LOGGED_PROMPT
        if not _LOGGED_PROMPT:
            print(f"[block_tower] prompt: {inputs['prompt']}, state_dim: {state.shape[-1]}, action_dim: {actions.shape[-1]}")
            _LOGGED_PROMPT = True

        return inputs


@dataclasses.dataclass(frozen=True)
class BlockTowerSubtaskInputs(transforms.DataTransformFn):
    """Hierarchical inputs for build_block_tower datasets."""

    default_prompt: str = "build a block tower"
    joints_only: bool = False

    def __call__(self, data: dict) -> dict:
        inputs = BlockTowerInputs(default_prompt=self.default_prompt, joints_only=self.joints_only)(data)

        if "prompt" in data:
            inputs["high_prompt"] = data["prompt"]
        elif "task" in data:
            inputs["high_prompt"] = str(data["task"])
        else:
            inputs["high_prompt"] = self.default_prompt

        subtask = _get_key(data, "subtask")
        if isinstance(subtask, bytes):
            subtask = subtask.decode("utf-8")
        elif hasattr(subtask, "item"):
            subtask = subtask.item()
        inputs["low_prompt"] = str(subtask)
        inputs.pop("prompt", None)
        return inputs


@dataclasses.dataclass(frozen=True)
class BlockTowerOutputs(transforms.DataTransformFn):
    """Outputs for build_block_tower datasets."""

    action_dim: int = _CANONICAL_DIM

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        actions = actions[:, :self.action_dim]
        return {"actions": actions.astype(np.float32)}
