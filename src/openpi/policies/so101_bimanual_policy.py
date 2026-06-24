"""Data transforms for bimanual SO101 (dual-arm, LeRobot v3 format).

Two 5-DOF arms + grippers = 12D joint-space:
  - observation.state: [left_shoulder_pan, left_shoulder_lift, left_elbow_flex,
      left_wrist_flex, left_wrist_roll, left_gripper,
      right_shoulder_pan, right_shoulder_lift, right_elbow_flex,
      right_wrist_flex, right_wrist_roll, right_gripper] (12D)
  - action: same 12D joint positions
  - observation.images.top, observation.images.left_wrist, observation.images.right_wrist

Camera -> model slot mapping:
  top -> base_0_rgb, left_wrist -> left_wrist_0_rgb, right_wrist -> right_wrist_0_rgb
"""

import dataclasses

import numpy as np

from openpi import transforms


_SO101_BIMANUAL_ACTION_DIM = 12


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
class SO101BimanualInputs(transforms.DataTransformFn):
    """Transforms bimanual SO101 dataset observations into model input format.

    Handles both training (LeRobot keys with dots/slashes) and inference
    (keys as sent by the robot driver).
    """

    default_prompt: str = "complete the task"

    def __call__(self, data: dict) -> dict:
        top = _parse_image(
            _get_key(data, "observation.images.top", "observation/images/top", "image", "images.top")
        )
        try:
            left_wrist = _parse_image(
                _get_key(
                    data,
                    "observation.images.left_wrist",
                    "observation/images/left_wrist",
                    "left_wrist_image",
                    "images.left_wrist",
                )
            )
        except KeyError:
            left_wrist = np.zeros_like(top)
        try:
            right_wrist = _parse_image(
                _get_key(
                    data,
                    "observation.images.right_wrist",
                    "observation/images/right_wrist",
                    "right_wrist_image",
                    "images.right_wrist",
                )
            )
        except KeyError:
            right_wrist = np.zeros_like(top)

        state = np.asarray(
            _get_key(data, "observation.state", "observation/state", "state"),
            dtype=np.float32,
        )

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": top,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
        elif "task" in data:
            task = data["task"]
            inputs["prompt"] = task.decode("utf-8") if isinstance(task, bytes) else str(task)
        else:
            inputs["prompt"] = self.default_prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class SO101BimanualOutputs(transforms.DataTransformFn):
    """Slices model output back to bimanual SO101's native 12D action space."""

    action_dim: int = _SO101_BIMANUAL_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        return {"actions": actions[:, : self.action_dim].astype(np.float32)}
