# Classifier-Free Guidance for Advantage Conditioning

**Date:** 2026-04-01
**Status:** Plan
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

### What we did

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

## Design

### Part 1: Training — advantage conditioning dropout

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

### Part 2: Inference — classifier-free guidance

CFG at inference requires two forward passes per denoising step: one with the advantage
prompt (conditional) and one without (unconditional). The two velocity predictions are
combined:

```
v_guided = v_uncond + β · (v_cond − v_uncond)
```

#### Architecture constraint: advantage text is in the prefix

In our implementation, the advantage text is part of `tokenized_prompt`, which is embedded
in the **prefix**. This means conditional vs unconditional produce different prefix KV caches.

For pi05 specifically, `sample_actions` calls `sample_low_level_task` first for autoregressive
subtask generation. The subtask tokens become part of the KV cache.

##### CFG inference flow for pi05

1. Build **two** observations:
   - `obs_cond`: prompt ends with `"Advantage: positive"`
   - `obs_uncond`: prompt has no advantage suffix

2. Run `sample_low_level_task` **once** with `obs_cond` to get subtask tokens.
   (The paper says the advantage indicator only affects actions, not subtask prediction.
   We generate subtask from the conditional prompt and reuse the same subtask text for both
   branches.)

3. Build **two** prefix KV caches:
   - `kv_cond`: from `obs_cond` prefix + subtask tokens
   - `kv_uncond`: from `obs_uncond` prefix + same subtask tokens

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

### Part 3: new training configs

Add new config variants with dropout enabled. Naming convention:

```
pi05_bin_pack_coffee_capsules_reward_recap_mixed_cfg
pi05_build_block_tower_mixed_cfg
```

These are identical to the existing `_mixed` configs but with `advantage_dropout_rate=0.3`.

Keep the existing configs unchanged so we can compare.

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

### Training side (required)
1. Add `dropout_rate` parameter to `InjectAdvantagePrompt` in `transforms.py`
2. Add `advantage_dropout_rate` parameter to `LeRobotBinPackDataConfig` and
   `LeRobotBlockTowerDataConfig` in `config.py`, wired through to the transform
3. Add new `_cfg` training configs with `advantage_dropout_rate=0.3`
4. Add/update tests for `InjectAdvantagePrompt` dropout behavior
5. Retrain models with dropout enabled

### Inference side (can follow)
6. Implement `sample_actions_cfg` on `Pi05` (and `Pi0` if needed)
7. Wire up evaluation/rollout code to construct dual observations and call CFG sampling
8. Test with β = 1 (should match standard sampling) and β > 1

### Evaluation
9. Compare retrained models (with dropout) vs existing models (without dropout) at β = 1
10. Sweep β ∈ {1.0, 1.5, 2.0, 2.5} on the retrained models

## Decisions

1. **Subtask interaction**: Ignore this for now. We will keep the advantage text in the
   current prompt path and not redesign prompt placement as part of this change.

2. **Dropout rate**: Fixed at `0.3`, matching the paper. We are not planning a dropout-rate
   sweep for v1.

3. **Interaction with `positive_only` mode**: The mode-based filtering happens first.
   In `positive_only`, policy chunks are still dropped entirely via `None`. Dropout only
   applies to examples that survive mode-based filtering, deciding whether they keep or
   omit the advantage suffix.

4. **FAST token / subtask training configs**: Not relevant for this work. The affected
   reward-recap configs do not use FAST-token or subtask training, so CFG planning does
   not need to account for those interactions.
