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


def test_inject_advantage_prompt_marks_policy_negative():
    transform = _transforms.InjectAdvantagePrompt()

    data = transform({"prompt": np.asarray("pick up the fork"), "control_mode": np.asarray("policy")})

    assert data["prompt"] == "pick up the fork. Advantage: negative"


def test_inject_advantage_prompt_treats_unknown_as_positive():
    transform = _transforms.InjectAdvantagePrompt()

    data = transform({"prompt": "pick up the fork", "control_mode": "unknown"})

    assert data["prompt"] == "pick up the fork. Advantage: positive"


def test_inject_advantage_prompt_positive_only_skips_policy():
    transform = _transforms.InjectAdvantagePrompt(mode="positive_only")

    result = transform({"prompt": "pick up the fork", "control_mode": "policy"})

    assert result is None


def test_inject_advantage_prompt_positive_only_keeps_human():
    transform = _transforms.InjectAdvantagePrompt(mode="positive_only")

    data = transform({"prompt": "pick up the fork", "control_mode": "human"})

    assert data["prompt"] == "pick up the fork. Advantage: positive"


def test_inject_advantage_prompt_positive_only_missing_control_mode():
    transform = _transforms.InjectAdvantagePrompt(mode="positive_only")

    data = transform({"prompt": "pick up the fork"})

    assert data["prompt"] == "pick up the fork. Advantage: positive"


def test_inject_advantage_prompt_dropout_omits_suffix_for_kept_example(monkeypatch):
    transform = _transforms.InjectAdvantagePrompt(dropout_rate=0.3)
    monkeypatch.setattr(np.random, "random", lambda: 0.1)

    data = transform({"prompt": "pick up the fork", "control_mode": "human"})

    assert data["prompt"] == "pick up the fork."


def test_inject_advantage_prompt_positive_only_still_skips_policy_before_dropout(monkeypatch):
    transform = _transforms.InjectAdvantagePrompt(mode="positive_only", dropout_rate=0.3)
    monkeypatch.setattr(np.random, "random", lambda: 0.1)

    result = transform({"prompt": "pick up the fork", "control_mode": "policy"})

    assert result is None


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
