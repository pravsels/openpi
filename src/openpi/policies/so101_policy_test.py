import numpy as np

from openpi.policies import so101_policy


def _rgb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(low=0, high=255, size=(48, 64, 3), dtype=np.uint8)


def test_two_cam_inputs_mask_right_wrist():
    front = _rgb(0)
    wrist = _rgb(1)
    raw = {
        "observation.images.front": front,
        "observation.images.wrist": wrist,
        "observation.state": np.arange(6, dtype=np.float32),
        "task": "stack the rings",
    }

    out = so101_policy.SO101Inputs()(raw)

    assert set(out["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    np.testing.assert_array_equal(out["image"]["base_0_rgb"], front)
    np.testing.assert_array_equal(out["image"]["left_wrist_0_rgb"], wrist)
    assert out["image_mask"]["base_0_rgb"]
    assert out["image_mask"]["left_wrist_0_rgb"]
    assert not out["image_mask"]["right_wrist_0_rgb"]


def test_three_cam_inputs_map_front_to_base_1():
    top = _rgb(2)
    wrist = _rgb(3)
    front = _rgb(4)
    raw = {
        "observation.images.top": top,
        "observation.images.wrist": wrist,
        "observation.images.front": front,
        "observation.state": np.arange(6, dtype=np.float32),
        "actions": np.ones((30, 6), dtype=np.float32),
        "task": "push the green button",
    }

    out = so101_policy.SO101ThreeCamInputs(default_prompt="push the green button")(raw)

    np.testing.assert_array_equal(out["image"]["base_0_rgb"], top)
    np.testing.assert_array_equal(out["image"]["left_wrist_0_rgb"], wrist)
    np.testing.assert_array_equal(out["image"]["base_1_rgb"], front)
    assert "right_wrist_0_rgb" not in out["image"]
    assert out["image_mask"]["base_0_rgb"]
    assert out["image_mask"]["left_wrist_0_rgb"]
    assert out["image_mask"]["base_1_rgb"]
    assert out["prompt"] == "push the green button"
    np.testing.assert_array_equal(out["actions"], raw["actions"])
    np.testing.assert_array_equal(out["state"], raw["observation.state"])
