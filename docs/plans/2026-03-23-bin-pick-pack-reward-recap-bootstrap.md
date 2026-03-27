# Bin Pick Pack Reward RECAP Bootstrap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the minimal `reward_recap` bootstrap for `bin_pick_pack` with two run modes: positive-only conditioning and mixed positive/negative conditioning from plugin-provided `control_mode`.

**Architecture:** Keep `pi05` unchanged and inject the condition entirely in the data pipeline. A new transform will rewrite the prompt from per-sample `control_mode`, and `LeRobotBinPackDataConfig` will opt into that transform with a small flag or mode enum so we can run both a positive-only baseline and a mixed positive/negative run without changing model code.

**Tech Stack:** Python, JAX data pipeline, LeRobot / RoboCandyWrapper dataset wrapping, pytest

---

### Task 1: Prompt Injection Transform

**Files:**
- Modify: `src/openpi/transforms.py`
- Test: `src/openpi/transforms_test.py`

**Step 1: Write the failing test**

```python
def test_inject_advantage_prompt_marks_policy_negative():
    transform = _transforms.InjectAdvantagePrompt()
    out = transform({"prompt": "pick up the fork", "control_mode": "policy"})
    assert out["prompt"] == "pick up the fork. Advantage: negative"
```

**Step 2: Run test to verify it fails**

Run: `pytest src/openpi/transforms_test.py -k advantage_prompt -v`
Expected: FAIL because `InjectAdvantagePrompt` does not exist yet.

**Step 3: Write minimal implementation**

```python
@dataclasses.dataclass(frozen=True)
class InjectAdvantagePrompt(DataTransformFn):
    mode: str = "mixed"

    def __call__(self, data: DataDict) -> DataDict:
        ...
```

**Step 4: Run test to verify it passes**

Run: `pytest src/openpi/transforms_test.py -k advantage_prompt -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/openpi/transforms.py src/openpi/transforms_test.py
git commit -m "feat: add control-mode advantage prompt injection"
```

### Task 2: Bin Pack Config Plumbing

**Files:**
- Modify: `src/openpi/training/config.py`
- Test: `src/openpi/transforms_test.py`

**Step 1: Write the failing test**

```python
def test_bin_pack_reward_recap_transform_adds_advantage_prompt():
    cfg = _config.LeRobotBinPackDataConfig(
        repo_id="fake",
        use_control_mode_advantage_prompt=True,
    )
    data_cfg = cfg.create(pathlib.Path("."), pi0_config.Pi0Config(pi05=True))
    assert any(isinstance(t, _transforms.InjectAdvantagePrompt) for t in data_cfg.data_transforms.inputs)
```

**Step 2: Run test to verify it fails**

Run: `pytest src/openpi/transforms_test.py -k bin_pack_reward_recap -v`
Expected: FAIL because the config flag/plumbing does not exist yet.

**Step 3: Write minimal implementation**

```python
@dataclasses.dataclass(frozen=True)
class LeRobotBinPackDataConfig(DataConfigFactory):
    use_control_mode_advantage_prompt: bool = False
    advantage_prompt_mode: str = "mixed"
```

and append the transform when enabled. The final implementation should support:

- positive-only: all examples get `Advantage: positive`
- mixed: `policy` examples become negative

**Step 4: Run test to verify it passes**

Run: `pytest src/openpi/transforms_test.py -k "bin_pack_reward_recap or advantage_prompt" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/openpi/training/config.py src/openpi/transforms_test.py
git commit -m "feat: wire bin pack reward recap bootstrap"
```

### Task 3: Two-Run Verification

**Files:**
- Test: `src/openpi/transforms_test.py`
- Test: `src/openpi/training/config_test.py`

**Step 1: Add a positive-only test**

```python
def test_inject_advantage_prompt_can_force_all_positive():
    transform = _transforms.InjectAdvantagePrompt(mode="positive_only")
    out = transform({"prompt": "pick up the fork", "control_mode": "policy"})
    assert out["prompt"] == "pick up the fork. Advantage: positive"
```

**Step 2: Run targeted tests**

Run: `pytest src/openpi/transforms_test.py -k "advantage_prompt or bin_pack_reward_recap" -v`
Expected: PASS

**Step 3: Run broader transform/config sanity checks**

Run: `pytest src/openpi/transforms_test.py src/openpi/training/config_test.py -v`
Expected: PASS

**Step 4: Lint / diagnostics**

Run IDE diagnostics on:
- `src/openpi/transforms.py`
- `src/openpi/training/config.py`
- `src/openpi/transforms_test.py`
- `src/openpi/training/config_test.py`

Expected: no new issues introduced.
