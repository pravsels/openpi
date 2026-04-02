import numpy as np

from openpi.models import tokenizer as _tokenizer


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


def test_pi05_tokenize_moves_advantage_after_state_before_action():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=64)
    state = np.zeros(4, dtype=np.float32)

    tokens, masks = tokenizer.tokenize("Pick up the cup. Advantage: positive", state)

    decoded = tokenizer.detokenize(tokens[masks])
    assert "Task: Pick up the cup." in decoded
    assert "State:" in decoded
    assert "Advantage: positive;" in decoded
    assert "Action:" in decoded
    assert decoded.index("State:") < decoded.index("Advantage: positive;") < decoded.index("Action:")


def test_hierarchical_tokenize_separates_advantage_into_action_prefix():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=128)
    state = np.zeros(4, dtype=np.float32)

    (
        tokens,
        token_masks,
        _ar_mask,
        loss_mask,
        subtask_region_mask,
        action_region_mask,
        action_prompt_tokens,
        action_prompt_mask,
    ) = tokenizer.tokenize_high_low_prompt(
        "Stack the blocks",
        "Place the red block. Advantage: negative",
        state,
    )

    decoded_main = tokenizer.detokenize(tokens[token_masks])
    decoded_action_prefix = tokenizer.detokenize(action_prompt_tokens[action_prompt_mask])

    assert "Subtask: place the red block." in decoded_main
    assert "Advantage:" not in decoded_main
    assert "Action:" not in decoded_main
    assert "Advantage: negative;" in decoded_action_prefix
    assert "Action:" in decoded_action_prefix
    assert np.any(loss_mask)
    assert np.any(subtask_region_mask)
    assert not np.any(action_region_mask)


def test_fast_tokenizer():
    prompt = "Hello, world!"
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(prompt, state, action)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)
