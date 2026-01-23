import dataclasses

import einops
import numpy as np

from openpi import transforms

_LOGGED_PROMPT = False


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _get_key(data: dict, *keys: str):
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"Missing keys: {keys}")


@dataclasses.dataclass(frozen=True)
class BinPackInputs(transforms.DataTransformFn):
    """Inputs for the bin_pack_coffee_capsules dataset."""

    default_prompt: str = "pack coffee capsules into the cardboard bin container"

    # Determines which model will be used (unused in this transform).
    model_type: object | None = None

    def __call__(self, data: dict) -> dict:
        front = _parse_image(_get_key(data, "observation/images/front", "observation.images.front"))
        wrist = _parse_image(_get_key(data, "observation/images/wrist", "observation.images.wrist"))

        state_pos = np.asarray(_get_key(data, "observation/state/pos", "observation.state.pos"))
        state_eef = np.asarray(_get_key(data, "observation/state/eef_pose", "observation.state.eef_pose"))
        state = np.concatenate([state_pos, state_eef], axis=-1)

        inputs = {
            "state": state,
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

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        else:
            action_pos = _get_key(data, "action/pos", "action.pos")
            action_eef = _get_key(data, "action/eef_pose", "action.eef_pose")
            inputs["actions"] = np.concatenate([np.asarray(action_pos), np.asarray(action_eef)], axis=-1)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        elif "task" in data:
            task = str(data["task"])
            task = task.replace("pick a single coffee capsule", "pick coffee capsules")
            inputs["prompt"] = task
        else:
            inputs["prompt"] = self.default_prompt

        global _LOGGED_PROMPT
        if not _LOGGED_PROMPT:
            print(f"[bin_pack] prompt: {inputs['prompt']}")
            _LOGGED_PROMPT = True

        return inputs


@dataclasses.dataclass(frozen=True)
class BinPackOutputs(transforms.DataTransformFn):
    """Outputs for the bin_pack_coffee_capsules dataset."""

    action_dim: int | None = None

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        action_dim = self.action_dim or actions.shape[-1]
        return {"actions": actions[:, : action_dim]}

