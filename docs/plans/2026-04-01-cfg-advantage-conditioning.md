# Classifier-Free Guidance for Advantage Conditioning

**Date:** 2026-04-01
**Status:** Partially implemented
**Prereqs:** Existing `InjectAdvantagePrompt` transform, reward recap configs

## Problem

Our current advantage conditioning implementation is missing classifier-free guidance (CFG),
which is a core part of how the paper makes advantage conditioning work.

### What the paper does (π\*0.6 / RECAP, Appendix F + Section V-B)

1. **Training**: Randomly drops the advantage indicator 30% of the time. This trains the
   model to produce actions both **with** and **without** the advantage signal:
   - `π(a | I_t, o, ℓ)` — conditional (sees "Advantage: positive" or "Advantage: negative")
   - `π(a | o, ℓ)` — unconditional (no advantage text at all)

   The dropout replaces the loss multiplier α from Equation 3. The paper states:
   > "We employ this dropout so that we can directly sample from either the conditional or
   > unconditional policy during inference time and use CFG for test-time policy improvement;
   > and it effectively replaces the loss multiplier α."

2. **Inference**: Two options depending on guidance scale β:
   - β = 1: Just sample from `π(a | I=positive, o, ℓ)` directly.
   - β > 1 (CFG): Combine conditional and unconditional via:
     ```
     v_guided = v_uncond + β · (v_cond − v_uncond)
     ```
     This sharpens the action distribution toward the conditional signal.

### What we started with

- **Training**: Every example always gets either `"Advantage: positive"` or
  `"Advantage: negative"`. No dropout. The model never learns `π(a | o, ℓ)`.
- **Inference**: Fixed `"Advantage: positive"` suffix. No CFG.

This was explicitly scoped out in `docs/reward_recap_design.md` as a v1 simplification.

### Why this matters

Without dropout during training, the model never learns a meaningful unconditional baseline.
The conditioning text just becomes part of the prompt noise — the model can't contrast
"advantage-conditioned" vs "unconditioned" because it's never seen the unconditioned case.
Even at β = 1, the advantage text carries less signal because the model hasn't been trained to
treat its presence/absence as informative.

## Scope

Both tasks that currently use `InjectAdvantagePrompt`:
- bin-pack (`LeRobotBinPackDataConfig`)
- block tower (`LeRobotBlockTowerDataConfig`)

## Implementation status

Implemented already:
- Training-side advantage dropout in `InjectAdvantagePrompt`
- `advantage_dropout_rate` wiring in both LeRobot data configs
- Dropout-enabled training configs with `advantage_dropout_rate=0.3`
- Core `Pi05.sample_actions_cfg()` implementation and unit tests

Still pending:
- Rollout / evaluation integration that constructs conditional and unconditional observations
- Retraining checkpoints that were originally trained without dropout
- End-to-end CFG evaluation and guidance-scale sweeps

## Design

### Part 1: Training — advantage conditioning dropout (implemented)

**File:** `src/openpi/transforms.py`, class `InjectAdvantagePrompt`

Add a `dropout_rate: float = 0.0` parameter. When dropout fires, the prompt is returned
**without** any `"Advantage: ..."` suffix — this is the unconditional training example.

Behavior by mode:
- `mixed` mode: each example has probability `dropout_rate` of getting no advantage suffix.
  When not dropped, the example gets `"positive"` or `"negative"` based on `control_mode`
  as before.
- `positive_only` mode: same — dropout means no suffix, otherwise `"positive"` only (and
  `None` for policy samples, as before).

The dropout decision must be **per-example** and **random**, not deterministic. Since
`InjectAdvantagePrompt.__call__` currently receives a single data dict (not a batch), we need
a source of randomness. Options:

1. **Use numpy RNG** — simplest. `InjectAdvantagePrompt` already uses numpy. Add an
   `np.random.default_rng()` instance or just use `np.random.random()`. Since transforms
   run in the data loader (not in JAX), numpy randomness is fine and doesn't affect
   reproducibility of the JAX training loop.

2. **Hash-based deterministic dropout** — hash the example index / episode ID to get a
   deterministic but uniform dropout decision. More reproducible but harder to implement
   cleanly since we don't always have a stable example index.

Recommendation: **option 1** (numpy RNG). The paper uses random dropout and doesn't claim
it needs to be deterministic.

Paper's dropout rate: **30%**. We should default to `0.3` for new configs but keep `0.0` as the
class default so existing configs are unaffected.

#### Transform changes

```python
@dataclasses.dataclass(frozen=True)
class InjectAdvantagePrompt(DataTransformFn):
    mode: Literal["positive_only", "mixed"] = "mixed"
    default_prompt: str | None = None
    negative_control_modes: tuple[str, ...] = ("policy",)
    dropout_rate: float = 0.0  # NEW: probability of omitting the advantage suffix

    def __call__(self, data: DataDict) -> DataDict | None:
        prompt = self._extract_prompt(data)
        if prompt is None:
            return data

        # NEW: with probability dropout_rate, return prompt without advantage suffix
        if self.dropout_rate > 0 and np.random.random() < self.dropout_rate:
            prompt = prompt.strip()
            if prompt and prompt[-1] not in ".!?":
                prompt += "."
            return {**data, "prompt": np.asarray(prompt)}

        # ... existing logic for positive_only / mixed ...
```

#### Config changes

**File:** `src/openpi/training/config.py`

In `LeRobotBinPackDataConfig` and `LeRobotBlockTowerDataConfig`, add
`advantage_dropout_rate: float = 0.0` and pass it through to `InjectAdvantagePrompt`.

New training configs (or updated existing ones) should set `advantage_dropout_rate=0.3`.

Status: implemented in both data configs, with tests covering the transform behavior and config
wiring.

### Part 2: Inference — classifier-free guidance (core model support implemented)

CFG at inference requires two forward passes per denoising step: one with the advantage
prompt (conditional) and one without (unconditional). The two velocity predictions are
combined:

```
v_guided = v_uncond + β · (v_cond − v_uncond)
```

#### Architecture constraint: there are two different "Pi05" paths in this repo

There are two materially different ways Pi05-style models show up in this codebase:

1. **Current reward-recap path**: configs use `pi0_config.Pi0Config(pi05=True)` together with the
   simple high-level prompt path. In this setup, the advantage text is currently injected into the
   plain `prompt` string before tokenization.
2. **True hierarchical Pi05 path**: configs use `Pi05Config` and the hierarchical
   `tokenize_high_low_prompt()` format, which explicitly separates
   `[task + state] + [subtask] + [actions]`.

The paper's literal placement rule is: the advantage indicator should appear **after the subtask
text but before the action prediction** so that it modulates the action distribution without
changing subtask generation.

##### Consequence for the current reward-recap path (non-hierarchical)

In the non-hierarchical path, there is no teacher-forced "post-subtask, pre-action" slot in the
training tokenizer. The prompt is just the high-level task prefix. That means exact paper-aligned
placement is awkward during training.

The **closest approximation** in this path is:

1. Keep the original high-level prompt prefix unchanged.
2. Run `sample_low_level_task()` to generate subtask tokens.
3. Append advantage tokens **after those generated subtask tokens**.
4. Build the action-conditioning KV cache from:
   - high-level prefix tokens
   - generated subtask tokens
   - advantage-indicator tokens

This keeps the advantage signal out of the initial subtask generation and makes it affect only the
action denoising stage, which is much closer to the paper than appending it to the original prompt.

##### Consequence for the true hierarchical Pi05 path

In the hierarchical path, the paper-aligned slot exists naturally in the tokenizer sequence:

`[task + state] -> [subtask] -> [advantage indicator] -> [actions]`

That is the cleanest place to put the indicator if we later migrate the reward-recap training path
to the true hierarchical Pi05 setup.

##### CFG inference flow for the current Pi05 implementation

1. Build **two** observations:
   - `obs_cond`: conditional branch
   - `obs_uncond`: unconditional branch

2. Run `sample_low_level_task()` **once** to get subtask tokens.

3. Build **two** action-conditioning prefix caches:
   - `kv_cond`: high-level prefix + subtask tokens + conditional advantage tokens
   - `kv_uncond`: high-level prefix + same subtask tokens + unconditional/no advantage tokens

4. Denoising loop: each step runs the action expert **twice**:
   ```
   v_cond  = suffix_forward(x_t, time, kv_cond)
   v_uncond = suffix_forward(x_t, time, kv_uncond)
   v_t = v_uncond + β · (v_cond − v_uncond)
   x_{t+dt} = x_t + dt · v_t
   ```

5. When β = 1, this reduces to `v_t = v_cond` (no guidance). The unconditional branch
   is not needed and can be skipped for efficiency.

##### Implementation approach

Add a new method `sample_actions_cfg` to `Pi05` (and/or `Pi0`) rather than modifying the
existing `sample_actions`. This keeps the default inference path unchanged and makes it easy
to A/B test.

Status: implemented for `Pi05`, including the `guidance_scale == 1` short-circuit and unit
tests for CFG combination and cache reuse. The remaining work is caller-side integration.

**File:** `src/openpi/models/pi05.py`

```python
def sample_actions_cfg(
    self,
    rng: at.KeyArrayLike,
    observation_cond: _model.Observation,
    observation_uncond: _model.Observation,
    *,
    guidance_scale: float = 1.5,
    num_steps: int | at.Int[at.Array, ""] = 10,
    noise: at.Float[at.Array, "b ah ad"] | None = None,
) -> _model.Actions:
    """Sample actions with classifier-free guidance.

    Requires two observations: one with advantage conditioning text and one without.
    """
    ...
```

The caller (policy server / evaluation script) is responsible for constructing the two
observations with the appropriate prompts.

**File:** `src/openpi/models/pi0.py` — same pattern for pi0 if we use it with advantage
conditioning.

##### Policy server / evaluation integration

The policy server or evaluation harness needs a way to:
1. Know the guidance scale β
2. Construct both conditional and unconditional observations
3. Call `sample_actions_cfg` instead of `sample_actions`

Options:
- Add `guidance_scale` as a serve config parameter. When > 1, the server strips the
  `"Advantage: ..."` suffix to build the unconditional observation and calls
  `sample_actions_cfg`.
- OR: handle this in the evaluation script / rollout code rather than the policy server.

Recommendation: keep it in the evaluation/rollout layer for now, not in the policy server.
This is easier to control and doesn't complicate the serving infrastructure. The evaluation
script already constructs the prompt, so it can construct both variants.

#### Computational cost

CFG doubles the action-expert forward passes per denoising step (10 steps × 2 = 20 forward
passes instead of 10). The prefix embedding is computed twice but cached, so the extra cost
per step is one suffix forward pass.

The paper notes they use moderate β ∈ [1.5, 2.5] and primarily rely on the training-time
threshold ε_ℓ rather than aggressive inference-time CFG.

For β = 1, the unconditional branch is not needed. We can short-circuit to the standard
`sample_actions` path.

### Part 3: new training configs (implemented, with final names differing slightly)

Add new config variants with dropout enabled. Original proposed naming convention:

```
pi05_bin_pack_coffee_capsules_reward_recap_mixed_cfg
pi05_build_block_tower_mixed_cfg
```

These are identical to the existing `_mixed` configs but with `advantage_dropout_rate=0.3`.

Keep the existing configs unchanged so we can compare.

Status: dropout-enabled configs were added, but the exact checked-in names differ slightly from
the original proposal in this document.

### Part 4: retraining

Models trained without dropout cannot do CFG — the unconditional distribution was never
learned. Existing checkpoints need to be **retrained** with dropout enabled.

Options:
- Retrain from scratch (from base or from task-trained checkpoint).
- Resume from an existing reward-recap checkpoint and continue training with dropout. This
  might work if the model can quickly learn the unconditional distribution, but the early
  training without dropout may have already shaped the model's treatment of the advantage
  text in ways that are hard to undo.

Recommendation: **retrain from the task-trained checkpoint** (same starting point as before)
with dropout enabled. This is the cleanest comparison.

## Task list

Completed:
1. Add `dropout_rate` parameter to `InjectAdvantagePrompt` in `transforms.py`
2. Add `advantage_dropout_rate` parameter to `LeRobotBinPackDataConfig` and
   `LeRobotBlockTowerDataConfig` in `config.py`, wired through to the transform
3. Add dropout-enabled training configs with `advantage_dropout_rate=0.3`
4. Add/update tests for `InjectAdvantagePrompt` dropout behavior
5. Implement `sample_actions_cfg` on `Pi05`

Remaining:
6. Retrain models with dropout enabled
7. Wire up evaluation/rollout code to construct dual observations and call CFG sampling
8. Test with β = 1 (should match standard sampling) and β > 1 end-to-end
9. Compare retrained models (with dropout) vs existing models (without dropout) at β = 1
10. Sweep β ∈ {1.0, 1.5, 2.0, 2.5} on the retrained models

## Decisions

1. **Subtask interaction**: The paper-aligned placement is different from the current prompt-based
   bootstrap path. For the current reward-recap configs, the closest non-hierarchical approximation
   is to add advantage tokens after generated subtask tokens and before action denoising. The truly
   clean placement exists in the hierarchical `Pi05Config` path.

2. **Dropout rate**: Fixed at `0.3`, matching the paper. We are not planning a dropout-rate
   sweep for v1.

3. **Interaction with `positive_only` mode**: The mode-based filtering happens first.
   In `positive_only`, policy chunks are still dropped entirely via `None`. Dropout only
   applies to examples that survive mode-based filtering, deciding whether they keep or
   omit the advantage suffix.

4. **FAST token / subtask training configs**: Not immediately required for the current reward-recap
   configs, but they are still relevant conceptually because the true hierarchical Pi05 path is the
   place where the paper's intended post-subtask/pre-action placement exists naturally.
