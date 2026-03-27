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

_LOGGED_PROMPT = False


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


@dataclasses.dataclass(frozen=True)
class BlockTowerInputs(transforms.DataTransformFn):
    """Inputs for build_block_tower datasets.

    Both state and actions are 7D joint-space, enabling delta action computation.
    """

    default_prompt: str = "build a block tower"
    model_type: object | None = None

    def __call__(self, data: dict) -> dict:
        front = _parse_image(_get_key(data, "observation/images/front", "observation.images.front"))
        wrist = _parse_image(_get_key(data, "observation/images/wrist", "observation.images.wrist"))

        state = np.asarray(
            _get_key(data, "observation/state", "observation.state", "observation/state/pos", "observation.state.pos")
        ).astype(np.float32)

        actions = np.asarray(_get_key(data, "action", "actions")).astype(np.float32)

        inputs = {
            "state": state,
            "actions": actions,
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
class BlockTowerOutputs(transforms.DataTransformFn):
    """Outputs for build_block_tower datasets."""

    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        actions = actions[:, :self.action_dim]
        return {"actions": actions.astype(np.float32)}
