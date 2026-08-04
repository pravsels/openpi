import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_tokenize_high_prompt_emits_structured_action_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=64)
    transform = _transforms.TokenizeHighPrompt(tokenizer)

    data = transform({"prompt": "Stack the blocks", "advantage_label": "positive"})

    decoded_main = tokenizer.detokenize(data["tokenized_prompt"])
    decoded_action_prefix = tokenizer.detokenize(
        data["action_tokenized_prompt"][data["action_tokenized_prompt_mask"]]
    )

    assert "Task: stack the blocks. Subtask:" in decoded_main
    assert "Advantage:" not in decoded_main
    assert decoded_action_prefix == "\nAdvantage: positive;\nAction: "


def test_tokenize_high_prompt_with_discrete_state_keeps_state_in_main_prefix():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=128)
    transform = _transforms.TokenizeHighPrompt(tokenizer, discrete_state_input=True)

    data = transform(
        {
            "prompt": "Stack the blocks",
            "state": np.zeros(4, dtype=np.float32),
            "advantage_label": "positive",
        }
    )

    decoded_main = tokenizer.detokenize(data["tokenized_prompt"])
    decoded_action_prefix = tokenizer.detokenize(
        data["action_tokenized_prompt"][data["action_tokenized_prompt_mask"]]
    )

    assert "Task: stack the blocks." in decoded_main
    assert "State:" in decoded_main
    assert "Subtask:" in decoded_main
    assert "Advantage:" not in decoded_main
    assert decoded_action_prefix == "\nAdvantage: positive;\nAction: "


def test_tokenize_high_low_prompt_emits_structured_action_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=128)
    transform = _transforms.TokenizeHighLowPrompt(tokenizer)

    data = transform(
        {
            "high_prompt": "Stack the blocks",
            "low_prompt": "Place the red block",
            "state": np.zeros(4, dtype=np.float32),
            "advantage_label": "negative",
        }
    )

    decoded_main = tokenizer.detokenize(data["tokenized_prompt"][data["tokenized_prompt_mask"]])
    decoded_action_prefix = tokenizer.detokenize(
        data["action_tokenized_prompt"][data["action_tokenized_prompt_mask"]]
    )

    assert "Subtask: place the red block." in decoded_main
    assert "Advantage:" not in decoded_main
    assert "Action:" not in decoded_main
    assert decoded_action_prefix == "\nAdvantage: negative;\nAction: "


def test_set_advantage_label_from_control_mode_marks_policy_negative_on_low_prompt():
    transform = _transforms.SetAdvantageLabelFromControlMode()

    data = transform({"low_prompt": np.asarray("pick up the fork"), "control_mode": np.asarray("policy")})

    assert data["low_prompt"] == "pick up the fork."
    assert data["advantage_label"] == "negative"


def test_set_advantage_label_from_control_mode_treats_unknown_as_positive_on_low_prompt():
    transform = _transforms.SetAdvantageLabelFromControlMode()

    data = transform({"low_prompt": "pick up the fork", "control_mode": "unknown"})

    assert data["low_prompt"] == "pick up the fork."
    assert data["advantage_label"] == "positive"


def test_set_advantage_label_from_control_mode_positive_only_skips_policy():
    transform = _transforms.SetAdvantageLabelFromControlMode(mode="positive_only")

    result = transform({"low_prompt": "pick up the fork", "control_mode": "policy"})

    assert result is None


def test_set_advantage_label_from_control_mode_positive_only_keeps_human_on_low_prompt():
    transform = _transforms.SetAdvantageLabelFromControlMode(mode="positive_only")

    data = transform({"low_prompt": "pick up the fork", "control_mode": "human"})

    assert data["low_prompt"] == "pick up the fork."
    assert data["advantage_label"] == "positive"


def test_set_advantage_label_from_control_mode_positive_only_missing_control_mode_uses_low_prompt():
    transform = _transforms.SetAdvantageLabelFromControlMode(mode="positive_only")

    data = transform({"low_prompt": "pick up the fork"})

    assert data["low_prompt"] == "pick up the fork."
    assert data["advantage_label"] == "positive"


def test_set_advantage_label_from_control_mode_targets_low_prompt():
    transform = _transforms.SetAdvantageLabelFromControlMode()

    data = transform({"low_prompt": "move arm to the cup", "control_mode": "policy"})

    assert data["low_prompt"] == "move arm to the cup."
    assert data["advantage_label"] == "negative"


def test_set_advantage_label_from_control_mode_dropout_omits_suffix_for_kept_low_prompt(monkeypatch):
    transform = _transforms.SetAdvantageLabelFromControlMode(dropout_rate=0.3)
    monkeypatch.setattr(np.random, "random", lambda: 0.1)

    data = transform({"low_prompt": "pick up the fork", "control_mode": "human"})

    assert data["low_prompt"] == "pick up the fork."
    assert "advantage_label" not in data


def test_set_advantage_label_from_control_mode_positive_only_still_skips_policy_before_dropout(monkeypatch):
    transform = _transforms.SetAdvantageLabelFromControlMode(mode="positive_only", dropout_rate=0.3)
    monkeypatch.setattr(np.random, "random", lambda: 0.1)

    result = transform({"low_prompt": "pick up the fork", "control_mode": "policy"})

    assert result is None


def test_set_advantage_label_from_control_mode_requires_low_prompt():
    transform = _transforms.SetAdvantageLabelFromControlMode()

    with pytest.raises(ValueError, match="low_prompt or prompt"):
        transform({"control_mode": "policy"})


def test_set_advantage_label_from_control_mode_marks_policy_negative_on_prompt():
    transform = _transforms.SetAdvantageLabelFromControlMode()

    data = transform({"prompt": "pick up the fork", "control_mode": "policy"})

    assert data["prompt"] == "pick up the fork."
    assert data["advantage_label"] == "negative"


def test_block_tower_subtask_inputs_emit_high_and_low_prompts():
    from openpi.policies import block_tower_policy

    transform = block_tower_policy.BlockTowerSubtaskInputs()
    data = transform(
        {
            "observation.images.front": np.zeros((3, 8, 8), dtype=np.uint8),
            "observation.images.wrist": np.zeros((3, 8, 8), dtype=np.uint8),
            "observation.state": np.zeros(7, dtype=np.float32),
            "action": np.zeros((2, 7), dtype=np.float32),
            "task": "build a block tower",
            "subtask": "pick up the red block",
        }
    )

    assert data["high_prompt"] == "build a block tower"
    assert data["low_prompt"] == "pick up the red block"


def test_block_tower_inputs_preserve_advantage_metadata():
    from openpi.policies import block_tower_policy

    transform = block_tower_policy.BlockTowerInputs()
    data = transform(
        {
            "observation.images.front": np.zeros((3, 8, 8), dtype=np.uint8),
            "observation.images.wrist": np.zeros((3, 8, 8), dtype=np.uint8),
            "observation.state": np.zeros(7, dtype=np.float32),
            "action": np.zeros((2, 7), dtype=np.float32),
            "task": "build a block tower",
            "control_mode": "policy",
            "advantage_label": "negative",
        }
    )

    assert data["control_mode"] == "policy"
    assert data["advantage_label"] == "negative"


def test_block_tower_subtask_inputs_preserve_control_mode_for_advantage_labeling():
    from openpi.policies import block_tower_policy

    transform = block_tower_policy.BlockTowerSubtaskInputs()
    data = transform(
        {
            "observation.images.front": np.zeros((3, 8, 8), dtype=np.uint8),
            "observation.images.wrist": np.zeros((3, 8, 8), dtype=np.uint8),
            "observation.state": np.zeros(7, dtype=np.float32),
            "action": np.zeros((2, 7), dtype=np.float32),
            "task": "build a block tower",
            "subtask": "pick up the red block",
            "control_mode": "policy",
        }
    )

    assert data["control_mode"] == "policy"


def test_block_tower_subtask_inputs_chain_into_advantage_labeling():
    from openpi.policies import block_tower_policy

    input_transform = block_tower_policy.BlockTowerSubtaskInputs()
    advantage_transform = _transforms.SetAdvantageLabelFromControlMode()

    data = advantage_transform(
        input_transform(
            {
                "observation.images.front": np.zeros((3, 8, 8), dtype=np.uint8),
                "observation.images.wrist": np.zeros((3, 8, 8), dtype=np.uint8),
                "observation.state": np.zeros(7, dtype=np.float32),
                "action": np.zeros((2, 7), dtype=np.float32),
                "task": "build a block tower",
                "subtask": "pick up the red block",
                "control_mode": "policy",
            }
        )
    )

    assert data["high_prompt"] == "build a block tower"
    assert data["low_prompt"] == "pick up the red block."
    assert data["advantage_label"] == "negative"


def test_block_tower_subtask_inputs_requires_subtask():
    from openpi.policies import block_tower_policy

    transform = block_tower_policy.BlockTowerSubtaskInputs()

    with pytest.raises(KeyError, match="subtask"):
        transform(
            {
                "observation.images.front": np.zeros((3, 8, 8), dtype=np.uint8),
                "observation.images.wrist": np.zeros((3, 8, 8), dtype=np.uint8),
                "observation.state": np.zeros(7, dtype=np.float32),
                "action": np.zeros((2, 7), dtype=np.float32),
                "task": "build a block tower",
            }
        )


def test_set_advantage_label_from_control_mode_dropout_omits_suffix_for_kept_prompt(monkeypatch):
    transform = _transforms.SetAdvantageLabelFromControlMode(dropout_rate=0.3)
    monkeypatch.setattr(np.random, "random", lambda: 0.1)

    data = transform({"prompt": "pick up the fork", "control_mode": "human"})

    assert data["prompt"] == "pick up the fork."
    assert "advantage_label" not in data


def test_quantile_normalize_uses_q01_q99_without_extra_clipping():
    stats = {
        "actions": _transforms.NormStats(
            mean=np.zeros(1),
            std=np.ones(1),
            q01=np.array([0.0]),
            q99=np.array([10.0]),
        )
    }
    transform = _transforms.Normalize(stats, use_quantiles=True)

    data = transform({"actions": np.array([[-10.0], [5.0], [20.0]])})

    assert np.allclose(data["actions"], np.array([[-3.0], [0.0], [3.0]]), atol=1e-6)


def test_quantile_normalize_does_not_explode_a_constant_channel():
    # A gripper that stays shut for 99% of an episode has q01 == q99, so the spread
    # is zero. Dividing by `spread + 1e-6` would scale the rare frame where it does
    # move by 10^6 and hand the model a target in the millions.
    stats = {
        "actions": _transforms.NormStats(mean=np.zeros(1), std=np.zeros(1), q01=np.array([0.0]), q99=np.array([0.0]))
    }
    transform = _transforms.Normalize(stats, use_quantiles=True)

    data = transform({"actions": np.array([[0.0], [52.87]])})

    # The value that equals the training constant sits at the bottom of the range,
    # and the outlier stays on the same scale as the data rather than 10^6 times it.
    assert np.allclose(data["actions"], np.array([[-1.0], [2.0 * 52.87 - 1.0]]), atol=1e-6)


def test_quantile_normalize_keeps_a_constant_channel_from_swamping_the_others():
    # The failure this guards against is not local to the bad channel: because the
    # loss sums over dimensions, one channel with a 10^6 gain dominates the gradient
    # for every other joint. Check the healthy channels are untouched by its presence.
    stats = {
        "actions": _transforms.NormStats(
            mean=np.zeros(3),
            std=np.ones(3),
            q01=np.array([0.0, -1.0, 0.0]),
            q99=np.array([10.0, 1.0, 0.0]),
        )
    }
    transform = _transforms.Normalize(stats, use_quantiles=True)

    data = transform({"actions": np.array([[5.0, 0.0, 3.0]])})

    assert np.allclose(data["actions"][0, :2], np.array([0.0, 0.0]), atol=1e-6)
    assert abs(data["actions"][0, 2]) < 10.0


def test_quantile_normalize_is_unchanged_for_well_conditioned_channels():
    q01, q99 = np.array([-31.994, 0.606]), np.array([76.764, 41.2])
    stats = {"actions": _transforms.NormStats(mean=np.zeros(2), std=np.ones(2), q01=q01, q99=q99)}
    x = np.array([[12.5, 3.75], [-20.0, 40.0]])

    data = _transforms.Normalize(stats, use_quantiles=True)({"actions": x.copy()})

    # Byte-for-byte the historical formula, so existing checkpoints are unaffected.
    assert np.array_equal(data["actions"], (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0)


def test_quantile_unnormalize_inverts_normalize_including_a_constant_channel():
    stats = {
        "actions": _transforms.NormStats(
            mean=np.zeros(2), std=np.ones(2), q01=np.array([-5.0, 0.0]), q99=np.array([5.0, 0.0])
        )
    }
    x = np.array([[2.5, 0.0]])

    normalized = _transforms.Normalize(stats, use_quantiles=True)({"actions": x.copy()})
    restored = _transforms.Unnormalize(stats, use_quantiles=True)({"actions": normalized["actions"]})

    assert np.allclose(restored["actions"], x, atol=1e-5)


def test_zscore_normalize_does_not_explode_a_constant_channel():
    stats = {"actions": _transforms.NormStats(mean=np.array([2.0]), std=np.array([0.0]))}

    normalized = _transforms.Normalize(stats)({"actions": np.array([[3.0]])})
    restored = _transforms.Unnormalize(stats)({"actions": normalized["actions"]})

    assert np.allclose(normalized["actions"], np.array([[1.0]]), atol=1e-6)
    assert np.allclose(restored["actions"], np.array([[3.0]]), atol=1e-6)


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})
