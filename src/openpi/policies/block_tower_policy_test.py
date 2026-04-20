import numpy as np

from openpi.policies import block_tower_policy
from openpi.policies import bin_pack_policy
from openpi import transforms


def test_block_tower_inputs_fallback_pad_7d_without_eef_pose():
    rng = np.random.default_rng(0)
    front = rng.integers(low=0, high=255, size=(224, 224, 3), dtype=np.uint8)
    wrist = rng.integers(low=0, high=255, size=(224, 224, 3), dtype=np.uint8)
    state = np.arange(7, dtype=np.float32)
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)

    x = block_tower_policy.BlockTowerInputs()(
        {
            "observation/images/front": front,
            "observation/images/wrist": wrist,
            "observation/state": state,
            "action": actions,
            "task": "build a block tower where the purple block forms the base",
        }
    )

    assert x["state"].shape == (17,)
    assert x["actions"].shape == (3, 17)
    assert np.array_equal(x["state"][:7], state)
    assert np.array_equal(x["actions"][:, :7], actions)
    assert np.array_equal(x["action_dim_mask"], np.array([True] * 7 + [False] * 10))


def test_block_tower_inputs_use_eef_pose_for_semantic_17d():
    rng = np.random.default_rng(1)
    front = rng.integers(low=0, high=255, size=(224, 224, 3), dtype=np.uint8)
    wrist = rng.integers(low=0, high=255, size=(224, 224, 3), dtype=np.uint8)
    state = rng.normal(size=(7,)).astype(np.float32)
    state_eef_xyz_rpy = np.array([0.1, -0.2, 0.3, 0.2, -0.1, 0.05], dtype=np.float32)
    actions = rng.normal(size=(3, 7)).astype(np.float32)
    action_eef_rpy = np.tile(np.concatenate([state_eef_xyz_rpy, state[6:7]]), (3, 1)).astype(np.float32)

    x = block_tower_policy.BlockTowerInputs()(
        {
            "observation/images/front": front,
            "observation/images/wrist": wrist,
            "observation/state": state,
            "observation/eef_6d_pose": state_eef_xyz_rpy,
            "action": actions,
            "action/eef_pose": action_eef_rpy,
            "task": "build a block tower where the purple block forms the base",
        }
    )

    assert x["state"].shape == (17,)
    assert x["actions"].shape == (3, 17)
    assert np.allclose(x["state"][:7], state)
    assert np.allclose(
        x["state"][7:],
        bin_pack_policy._eef_pose_rpy_to_rot6d(np.concatenate([state_eef_xyz_rpy, state[6:7]])),
    )
    assert np.allclose(x["actions"][:, :7], actions)
    assert np.allclose(x["actions"][:, 7:], bin_pack_policy._eef_pose_rpy_to_rot6d(action_eef_rpy))
    assert np.array_equal(x["action_dim_mask"], np.ones(17, dtype=bool))


def test_block_tower_inputs_joints_only_without_eef_data_pads_and_masks():
    """joints_only without EEF keys: should pad to 17D, zero EE, use joints mask."""
    rng = np.random.default_rng(3)
    front = rng.integers(low=0, high=255, size=(224, 224, 3), dtype=np.uint8)
    wrist = rng.integers(low=0, high=255, size=(224, 224, 3), dtype=np.uint8)
    state = rng.normal(size=(7,)).astype(np.float32)
    actions = rng.normal(size=(3, 7)).astype(np.float32)

    x = block_tower_policy.BlockTowerInputs(joints_only=True)(
        {
            "observation/images/front": front,
            "observation/images/wrist": wrist,
            "observation/state": state,
            "action": actions,
            "task": "build a block tower",
        }
    )

    assert x["state"].shape == (17,)
    assert x["actions"].shape == (3, 17)
    assert np.array_equal(x["state"][:7], state)
    assert np.array_equal(x["state"][7:], np.zeros(10, dtype=np.float32))
    assert np.array_equal(x["actions"][:, :7], actions)
    assert np.array_equal(x["actions"][:, 7:], np.zeros((3, 10), dtype=np.float32))
    assert np.array_equal(x["action_dim_mask"], np.array([True] * 7 + [False] * 10))


def test_block_tower_inputs_joints_only_zeros_eef_and_forces_joints_mask():
    rng = np.random.default_rng(2)
    front = rng.integers(low=0, high=255, size=(224, 224, 3), dtype=np.uint8)
    wrist = rng.integers(low=0, high=255, size=(224, 224, 3), dtype=np.uint8)
    state = rng.normal(size=(7,)).astype(np.float32)
    state_eef_xyz_rpy = np.array([0.1, -0.2, 0.3, 0.2, -0.1, 0.05], dtype=np.float32)
    actions = rng.normal(size=(3, 7)).astype(np.float32)
    action_eef_rpy = np.tile(np.concatenate([state_eef_xyz_rpy, state[6:7]]), (3, 1)).astype(np.float32)

    x = block_tower_policy.BlockTowerInputs(joints_only=True)(
        {
            "observation/images/front": front,
            "observation/images/wrist": wrist,
            "observation/state": state,
            "observation/eef_6d_pose": state_eef_xyz_rpy,
            "action": actions,
            "action/eef_pose": action_eef_rpy,
            "task": "build a block tower where the purple block forms the base",
        }
    )

    assert x["state"].shape == (17,)
    assert x["actions"].shape == (3, 17)
    assert np.array_equal(x["state"][:7], state)
    assert np.array_equal(x["state"][7:], np.zeros(10, dtype=np.float32))
    assert np.array_equal(x["actions"][:, :7], actions)
    assert np.array_equal(x["actions"][:, 7:], np.zeros((3, 10), dtype=np.float32))
    assert np.array_equal(x["action_dim_mask"], np.array([True] * 7 + [False] * 10))


def test_pad_states_and_actions_preserves_existing_action_dim_mask():
    data = {
        "state": np.zeros(17, dtype=np.float32),
        "actions": np.zeros((2, 17), dtype=np.float32),
        "action_dim_mask": np.array([True] * 7 + [False] * 10),
    }

    padded = transforms.PadStatesAndActions(model_action_dim=32)(data)

    assert padded["state"].shape == (32,)
    assert padded["actions"].shape == (2, 32)
    assert padded["action_dim_mask"].shape == (32,)
    assert np.array_equal(padded["action_dim_mask"][:17], np.array([True] * 7 + [False] * 10))
    assert np.array_equal(padded["action_dim_mask"][17:], np.zeros(15, dtype=bool))
