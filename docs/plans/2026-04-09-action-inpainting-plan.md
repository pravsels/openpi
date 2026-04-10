# Action Inpainting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add BEHAVIOR-style action inpainting to `openpi` so a caller can feed back a short action prefix from the previous chunk, constrain the next chunk during denoising, and optionally propagate that constraint through a correlation-aware correction matrix.

**Architecture:** Extend the `Policy.infer` and `Pi05.sample_actions` interfaces with an optional `initial_actions` argument in action space. Normalize those actions into model space before sampling, then apply constrained denoising inside the `Pi05` sampler rather than adding new prompt tokens. Start with hard/soft constrained denoising, then add covariance-backed correction using action-correlation statistics loaded from normalization assets. The deployment owner for chunk reuse is `../alpha-robotics/hw_control/new_lerobot_integrations/deploy_policy.py` together with `../alpha-robotics/hw_control/new_lerobot_integrations/openpi_utils.py`, and the adapter boundary in `missiontracker/adapters/openpi_adapter.py` must be treated as an explicit integration surface, not an implicit passthrough.

**Tech Stack:** JAX, Flax NNX, existing `openpi` policy wrappers, `Pi05` sampler helpers, pytest, and the alpha-robotics OpenPI deployment runtime.

**Execution note:** Do not create intermediate commits while executing this plan. Only create a commit if explicitly requested after implementation and verification.

---

### Task 1: Document the target behavior and data flow

**Files:**
- Modify: `docs/behavior_challenge_solution_changes.md`
- Create: `docs/plans/2026-04-09-action-inpainting-plan.md`

**Step 1: Write a short behavior contract in the plan**

Add a section that defines:

- external API: `policy.infer(obs, initial_actions=...)`
- accepted shape: `(k, action_dim)` or batched `(1, k, action_dim)`
- expected semantics: constrain the first `k` future actions during denoising
- non-goal: do not add action-prefix tokens to the transformer prefix
- deployment control point: the runtime decides whether forwarding is enabled, how many actions from the tail are kept, and whether correlation-aware inpainting is requested
- action-space contract: `initial_actions` must be in the same space expected by `Policy.infer` for OpenPI, and the plan must explicitly define whether the broker stores pre- or post-adapter-transformed chunks
- compatibility rule: generic `Policy.infer` must not blindly pass `initial_actions` to models that do not support it

**Step 2: Record the model-space data flow**

Add a short note to the plan:

```text
caller actions -> policy input transforms -> normalized initial_actions -> Pi05.sample_actions(...)
-> denoising loop constraint -> final sampled chunk -> output transforms
```

Also record the deployment path:

```text
deploy_policy.py flags -> AdapterChunkBroker chunk bookkeeping -> adapter.predict(..., initial_actions=?)
-> OpenPI policy.infer(..., initial_actions=...) -> Pi05 sampler
```

Add a short note that covariance / correction matrices must be computed in the same action coordinates as the sampler state's `x_t`, not merely any serialized action-statistics space.

**Step 3: Verify no code change is needed yet**

Run: no command, documentation-only step.

**Step 4: Planning note**

This task is documentation-only.

### Task 2: Add failing tests for policy forwarding

**Files:**
- Modify: `src/openpi/policies/policy.py`
- Create or modify: a policy-focused test file near `src/openpi/policies/`
- Test: `src/openpi/models/pi05_test.py` only for Pi05-specific plumbing that truly belongs there

**Step 1: Write the failing tests**

Add tests that verify:

- `Policy.infer(..., initial_actions=arr)` forwards `initial_actions` into `model.sample_actions`
- a 2D `(k, d)` input is batched to `(1, k, d)`
- `noise` and `initial_actions` can both be passed in the same call
- unsupported model types either ignore `initial_actions` intentionally or raise a clear error at the policy boundary instead of a low-level `TypeError`

Suggested test shape:

```python
def test_policy_infer_forwards_initial_actions():
    captured = {}

    def fake_sample_actions(rng, observation, **kwargs):
        captured.update(kwargs)
        return jnp.zeros((1, 4, 8))

    ...
    policy.infer(obs_dict, initial_actions=np.ones((2, 8), dtype=np.float32))
    assert "initial_actions" in captured
    assert captured["initial_actions"].shape == (1, 2, 8)
```

**Step 2: Run test to verify it fails**

Run: `pytest src/openpi/models/pi05_test.py -k initial_actions -q`

Expected: FAIL because `Policy.infer` does not yet accept or forward `initial_actions`.

**Step 3: Write minimal implementation**

Update `Policy.infer` to accept:

```python
def infer(
    self,
    obs: dict,
    *,
    noise: np.ndarray | None = None,
    initial_actions: np.ndarray | None = None,
    uncond_obs: dict | None = None,
    guidance_scale: float | None = None,
) -> dict:
```

Then:

- convert `initial_actions` to JAX/PyTorch array
- add a batch dimension if needed
- place it in `sample_kwargs["initial_actions"]`
- add explicit feature detection or model capability gating so non-Pi05 models are not broken by the new generic `Policy.infer` kwarg

**Step 4: Run test to verify it passes**

Run: `pytest src/openpi/models/pi05_test.py -k initial_actions -q`

Expected: PASS

**Step 5: Checkpoint review**

Confirm:

- tests added in this task are green
- `Policy.infer` API change is minimal and actually backwards-compatible across supported model families

### Task 2.5: Plan the deployment/runtime integration points

**Files:**
- Modify: `../alpha-robotics/hw_control/new_lerobot_integrations/deploy_policy.py`
- Modify: `../alpha-robotics/hw_control/new_lerobot_integrations/openpi_utils.py`
- Modify: `../missiontracker/adapters/openpi_adapter.py`
- Modify: any adapter interface / base type if needed so new kwargs are explicit

**Step 1: Write down the runtime knobs**

Add these intended controls to the plan:

- `--actions-to-execute`: existing knob, keep using it
- `--openpi-forward-tail-actions`: bool flag, enables/disables feeding leftover actions into the next inference call
- `--openpi-tail-actions-to-keep`: int flag, number of leftover actions to pass back
- `--openpi-use-action-correlation`: bool flag, requests correlation-aware inpainting if supported by the model/checkpoint

**Step 2: Record ownership boundaries**

Document:

- `deploy_policy.py` owns CLI flags
- `AdapterChunkBroker` owns chunk execution, tail caching, and deciding when to call the adapter again
- `missiontracker` adapter owns the contract between broker/runtime tensors and `openpi` policy inputs
- `openpi` owns sampler semantics once `initial_actions` reaches `policy.infer`

**Step 3: Record the concrete rollout pattern**

Document the intended example:

```text
Infer 50 actions -> execute first 30 -> keep remaining 20 -> next call passes those 20 as initial_actions
```

Clarify that tail forwarding must be optional and controlled by a runtime flag.

Also clarify:

- whether the broker stores the raw adapter `action_chunk`
- whether that chunk is already converted from delta to absolute or otherwise transformed
- how that stored chunk is converted back into the exact action space expected by `Policy.infer(..., initial_actions=...)`

**Step 4: Planning note**

Keep the adapter/runtime ownership explicit in the plan.

#### Runtime integration notes (Task 2.5 output)

**Intended runtime flags** (all on `deploy_policy.py`):

- `--actions-to-execute` (existing, keep as-is)
- `--openpi-forward-tail-actions` (bool, enables feeding leftover actions into next inference)
- `--openpi-tail-actions-to-keep` (int, how many tail actions to cache)
- `--openpi-use-action-correlation` (bool, requests correlation-aware inpainting)

**Ownership boundaries:**

| Component | Owns |
|-----------|------|
| `deploy_policy.py` | CLI flags, argument validation, incompatible-combination rejection |
| `AdapterChunkBroker` (`openpi_utils.py`) | Chunk execution, tail caching, deciding when to re-infer |
| `OpenPiAdapter` (`missiontracker/adapters/openpi_adapter.py`) | Contract between broker tensors and `openpi` policy inputs |
| `openpi Policy.infer` | Normalization, batching, forwarding `initial_actions` into sampler |
| `Pi05` sampler | Constrained denoising semantics once `initial_actions` arrives |

**Concrete rollout pattern:**

```text
Infer 50 actions → execute first 30 → keep remaining 20
→ next call passes those 20 as initial_actions
→ Pi05 constrains first 20 timesteps during denoising
```

**Chunk space contract:**

- `AdapterChunkBroker` stores `action_chunk` from `OpenPiAdapter.predict()`, which is already in absolute joint-position space (post delta→absolute conversion if applicable).
- The tail segment (`chunk[actions_to_execute : actions_to_execute + tail_to_keep]`) is in the same space as `Policy.infer` output actions.
- When fed back as `initial_actions`, `Policy.infer` transforms them through input transforms (including normalization) before passing to the sampler, so the broker stores raw output-space actions and the policy handles coordinate conversion.
- If CFG + tail forwarding is requested and CFG inpainting is not yet supported, `deploy_policy.py` must reject the combination early.

---

### Task 3: Add failing tests for `Pi05.sample_actions` API and plumbing

**Files:**
- Modify: `src/openpi/models/pi05.py`
- Test: `src/openpi/models/pi05_test.py`

**Step 1: Write the failing tests**

Add tests that verify:

- `Pi05.sample_actions(..., initial_actions=...)` passes the argument into `_sample_actions_with_prefix_cache`
- `Pi05.sample_actions_cfg(..., initial_actions=...)` either forwards it symmetrically or explicitly rejects it with a clear error if CFG support is deferred
- shape validation rejects invalid `initial_actions`
- transform-aligned `initial_actions` arrive in the same model-space coordinates used by the sampler

Suggested test shape:

```python
def test_pi05_sample_actions_forwards_initial_actions_to_prefix_cache_sampler():
    captured = {}

    def sample_with_cache(observation, kv_cache, prefix_mask, *, num_steps, noise, initial_actions=None):
        captured["initial_actions"] = initial_actions
        return expected_actions
```

**Step 2: Run test to verify it fails**

Run: `pytest src/openpi/models/pi05_test.py -k "forwards_initial_actions or validates_initial_actions" -q`

Expected: FAIL because the sampler helpers do not accept the new kwarg.

**Step 3: Write minimal implementation**

Update:

- `Pi05.sample_actions(...)`
- `Pi05._sample_actions_with_prefix_cache(...)`
- optionally `Pi05.sample_actions_cfg(...)` and `_sample_actions_cfg_with_prefix_caches(...)`

Add the kwarg:

```python
initial_actions: jax.Array | None = None
```

If full CFG support is not implemented in the same task, raise:

```python
raise NotImplementedError("initial_actions is not yet supported with sample_actions_cfg")
```

**Step 4: Run test to verify it passes**

Run: `pytest src/openpi/models/pi05_test.py -k "forwards_initial_actions or validates_initial_actions" -q`

Expected: PASS

**Step 5: Checkpoint review**

Confirm:

- sampler plumbing is in place
- tests for forwarding and validation are green

### Task 4: Implement hard constrained denoising

**Files:**
- Modify: `src/openpi/models/pi05.py`
- Test: `src/openpi/models/pi05_test.py`

**Step 1: Write the failing test**

Add a unit test for a helper that:

- takes `initial_actions` of shape `(1, k, d)`
- pads them to `(1, action_horizon, action_dim)` if needed
- builds flat constrained indices for the first `k` timesteps and provided action dims

Add a sampler-level test that verifies constrained coordinates are changed by the inpainting helper before loop exit.

**Step 2: Run test to verify it fails**

Run: `pytest src/openpi/models/pi05_test.py -k "inpainting_indices or hard_inpainting" -q`

Expected: FAIL because helpers do not exist.

**Step 3: Write minimal implementation**

Add private helpers in `Pi05`, for example:

- `_prepare_initial_actions_for_inpainting(...)`
- `_build_inpainting_indices(...)`
- `_apply_hard_inpainting_constraint(...)`

Behavior:

- normalize expected shapes
- pad to horizon and action_dim
- build `O_indices`
- explicitly document flattening order for `(t, d)` indices and keep it consistent with the covariance layout used later
- at each denoising step compute desired constrained values:

```python
x_desired_O = (1.0 - time_new) * x0_O + time_new * z_O
```

- overwrite `x_flat[:, O_indices]` with `x_desired_O`

**Step 4: Run test to verify it passes**

Run: `pytest src/openpi/models/pi05_test.py -k "inpainting_indices or hard_inpainting" -q`

Expected: PASS

**Step 5: Checkpoint review**

Confirm:

- hard constrained denoising works in isolation
- helper naming and scope are still minimal

### Task 5: Add soft time-thresholded inpainting

**Files:**
- Modify: `src/openpi/models/pi05.py`
- Modify: `src/openpi/models/pi05_config.py`
- Test: `src/openpi/models/pi05_test.py`

**Step 1: Write the failing test**

Add tests for:

- new config field `time_threshold_inpaint`
- constraint is applied only while `time > threshold`
- once the threshold is crossed, denoising proceeds freely

**Step 2: Run test to verify it fails**

Run: `pytest src/openpi/models/pi05_test.py -k "time_threshold_inpaint or soft_inpainting" -q`

Expected: FAIL because config and branch logic are missing.

**Step 3: Write minimal implementation**

Add config field:

```python
time_threshold_inpaint: float = 0.3
```

In the denoising loop:

- compute `time_new = time + dt`
- only apply constrained overwrite if `time_new > self.time_threshold_inpaint`

**Step 4: Run test to verify it passes**

Run: `pytest src/openpi/models/pi05_test.py -k "time_threshold_inpaint or soft_inpainting" -q`

Expected: PASS

**Step 5: Checkpoint review**

Confirm:

- thresholded behavior matches the tests
- config additions are narrowly scoped

### Task 6: Extend norm stats with action-correlation Cholesky

**Files:**
- Modify: `src/openpi/shared/normalize.py`
- Modify: `scripts/compute_norm_stats.py`
- Test: create or extend a normalize-related test file if one exists nearby

**Step 1: Write the failing test**

Add a test that:

- serializes and deserializes `NormStats` with an `action_correlation_cholesky` field
- preserves shape and values

**Step 2: Run test to verify it fails**

Run: `pytest -q -k "norm_stats and correlation"`

Expected: FAIL because the field does not exist yet.

**Step 3: Write minimal implementation**

Add to `NormStats`:

```python
action_correlation_cholesky: numpydantic.NDArray | None = None
```

Then extend norm-stat computation so an optional flag computes:

- flattened action covariance over `(action_horizon * action_dim)`
- Cholesky decomposition
- saves it in the action stats block
- ensure the covariance is computed in the same action space that the sampler uses for `x_t`, and write that invariant down in code comments/tests

**Step 4: Run test to verify it passes**

Run: `pytest -q -k "norm_stats and correlation"`

Expected: PASS

**Step 5: Checkpoint review**

Confirm:

- norm stats schema changes serialize and deserialize correctly
- compute script behavior is explicit and documented

### Task 7: Load and validate action correlation in `Pi05`

**Files:**
- Modify: `src/openpi/models/pi05.py`
- Modify: `src/openpi/models/pi05_config.py`
- Test: `src/openpi/models/pi05_test.py`

**Step 1: Write the failing test**

Add tests that verify:

- `Pi05` can load a Cholesky factor from norm stats
- mismatched dimensions raise a clear error
- missing correlation stats raise a clear error when correlation-aware inpainting is enabled
- near-singular or badly conditioned correlation inputs fail loudly or are regularized in a defined way

**Step 2: Run test to verify it fails**

Run: `pytest src/openpi/models/pi05_test.py -k "correlation_matrix or correlated_inpainting" -q`

Expected: FAIL because no loading/validation helpers exist.

**Step 3: Write minimal implementation**

Add fields/config:

- `use_correlation_inpainting: bool = False`
- `correlation_beta: float = 0.5`

Add helpers:

- `load_correlation_matrix(norm_stats)`
- `generate_correlated_noise(...)` if you want the same machinery reusable later
- storage for `action_correlation_cholesky`

Keep this task focused on loading/validation, not yet on the correction matrix.

**Step 4: Run test to verify it passes**

Run: `pytest src/openpi/models/pi05_test.py -k "correlation_matrix or correlated_inpainting" -q`

Expected: PASS

**Step 5: Checkpoint review**

Confirm:

- loading and validation paths fail loudly on mismatches
- config flags are clear

### Task 8: Implement correlation-aware correction matrix

**Files:**
- Modify: `src/openpi/models/pi05.py`
- Test: `src/openpi/models/pi05_test.py`

**Step 1: Write the failing test**

Add helper-level tests for:

- `_precompute_correction_matrix(O_indices, U_indices)`
- correction matrix shape `[|U|, |O|]`
- correction leaves constrained coordinates exact
- unconstrained coordinates are modified when correlation is nontrivial

**Step 2: Run test to verify it fails**

Run: `pytest src/openpi/models/pi05_test.py -k "correction_matrix or correlation_aware_inpainting" -q`

Expected: FAIL because the helper does not exist.

**Step 3: Write minimal implementation**

Add helper in `Pi05`:

```python
def _precompute_correction_matrix(self, O_indices, U_indices):
    Sigma = L @ L.T
    Sigma_OO = Sigma[jnp.ix_(O_indices, O_indices)]
    Sigma_UO = Sigma[jnp.ix_(U_indices, O_indices)]
    correction_matrix = solve(Sigma_OO_reg, Sigma_UO.T).T
```

Cache by constrained prefix shape, e.g. `(num_initial_steps, constrained_action_dim)`.

In the denoising step:

- compute `delta_O`
- hard-set constrained dims
- if enabled, compute `delta_U = correction_matrix @ delta_O`
- add `delta_U` to unconstrained dims

**Step 4: Run test to verify it passes**

Run: `pytest src/openpi/models/pi05_test.py -k "correction_matrix or correlation_aware_inpainting" -q`

Expected: PASS

**Step 5: Checkpoint review**

Confirm:

- correction matrix caching is correct
- constrained and unconstrained updates behave as intended

### Task 8.5: Defer action-expert inspection until final review

**Files:**
- Create: `scripts/inspect_action_inpainting.py`
- Modify: `docs/plans/2026-04-09-action-inpainting-plan.md`
- Optional notes output: `run_logs/` or a markdown note under `docs/` if useful

**Step 1: Record the review requirement**

Do not stop implementation here for inspection work. The standalone inspection script and output study happen only in final verification.

**Step 2: Capture the deferred deliverable**

At final review time we still want:

- `scripts/inspect_action_inpainting.py`
- baseline vs forwarded-action output comparison
- forwarded-action vs correlation-aware output comparison
- a short written note on what changed

**Step 3: Checkpoint review**

Confirm:

- implementation should continue without waiting on the analysis script
- the analysis requirement remains part of final verification

### Task 9: Decide and implement CFG behavior

**Files:**
- Modify: `src/openpi/models/pi05.py`
- Modify: `src/openpi/policies/policy.py`
- Test: `src/openpi/models/pi05_test.py`

**Step 1: Write the failing test**

Choose one behavior and encode it in tests:

- either `sample_actions_cfg(..., initial_actions=...)` is supported
- or it raises a deliberate `NotImplementedError`

If supporting it, add a forwarding test that both CFG branches share the same inpainting constraint.

**Step 2: Run test to verify it fails**

Run: `pytest src/openpi/models/pi05_test.py -k "cfg and initial_actions" -q`

Expected: FAIL until the policy/model behavior is made explicit.

**Step 3: Write minimal implementation**

Recommended initial choice: explicitly reject CFG + inpainting in the first version unless there is an immediate need.

```python
if initial_actions is not None:
    raise NotImplementedError("initial_actions is not yet supported with sample_actions_cfg")
```

This keeps scope under control and avoids silent inconsistencies.

Also require a policy/runtime-facing guard:

- `deploy_policy.py` must reject incompatible combinations such as tail forwarding + CFG before entering the control loop
- the adapter should fail clearly if asked for an unsupported combination

**Step 4: Run test to verify it passes**

Run: `pytest src/openpi/models/pi05_test.py -k "cfg and initial_actions" -q`

Expected: PASS

**Step 5: Checkpoint review**

Confirm:

- CFG behavior is explicit, tested, and documented
- unsupported combinations fail clearly

### Task 10: Add an inference wrapper example and rollout notes

**Files:**
- Modify: `docs/remote_inference.md`
- Modify: `../alpha-robotics/hw_control/new_lerobot_integrations/deploy_policy.py`
- Modify: `../alpha-robotics/hw_control/new_lerobot_integrations/openpi_utils.py`
- Modify: `../missiontracker/adapters/openpi_adapter.py`
- Create or modify: a small policy wrapper/example if there is an appropriate location

**Step 1: Write the failing documentation task**

Document and implement:

- predict chunk
- execute first `N`
- save next `K`
- feed saved `K` back as `initial_actions` on the next call
- gate all of the above behind runtime flags instead of making it always on

Include one short code sample.

**Step 2: Verify docs gap exists**

Run: no command, docs-only verification.

**Step 3: Write minimal implementation**

Add a short example like:

```python
next_initial_actions = None
while True:
    output = policy.infer(obs, initial_actions=next_initial_actions)
    chunk = output["actions"]
    execute(chunk[:26])
    next_initial_actions = chunk[26:30]
```

Add caveats:

- action space must match policy output space
- values are transformed internally into model space
- this is sampler-level constrained generation, not prefix-token conditioning
- document explicitly whether the adapter returns absolute or delta-like chunks and how the broker converts them before reusing them as `initial_actions`

For the alpha-robotics runtime, update `AdapterChunkBroker` to support tail forwarding roughly like:

```python
if self._chunk is None or self._index >= self._actions_to_execute:
    output = self._adapter.predict(
        policy_obs,
        advantage_mode=self._advantage_mode,
        guidance_scale=self._guidance_scale,
        initial_actions=self._next_initial_actions if self._forward_tail_actions else None,
        use_action_correlation=self._use_action_correlation,
    )
    self._chunk = output.action_chunk.cpu().numpy()
    self._next_initial_actions = (
        self._chunk[self._actions_to_execute : self._actions_to_execute + self._tail_actions_to_keep].copy()
        if self._forward_tail_actions else None
    )
    self._index = 0
```

In `deploy_policy.py`, add CLI flags so the user can choose:

- whether forwarding happens at all
- how many tail actions are kept
- whether correlation-aware inference is requested
- and reject unsupported combinations, especially CFG + tail forwarding if CFG remains unsupported for inpainting

**Step 4: Run docs check**

Run: no command

Expected: doc is readable and consistent with the implementation.

**Step 5: Checkpoint review**

Confirm:

- docs match the implemented runtime behavior
- alpha-robotics flag names are consistent with the broker API

### Task 11: Run focused verification

**Files:**
- Modify: none
- Test: `src/openpi/models/pi05_test.py`
- Test: policy-layer tests near `src/openpi/policies/`
- Test: normalize/correlation tests
- Test: alpha-robotics / adapter integration tests or at least focused smoke checks across repo boundaries

**Step 1: Run targeted tests**

Run:

```bash
pytest src/openpi/models/pi05_test.py -q
```

Expected: PASS

**Step 2: Run correlation-related tests**

Run:

```bash
pytest src/openpi/models/pi05_test.py -k "correlation or inpainting or initial_actions" -q
```

Expected: PASS

**Step 3: Run policy-layer tests**

Run:

```bash
pytest src/openpi/policies -k "initial_actions or infer" -q
```

Expected: PASS

**Step 4: Run the action-expert inspection script**

Run:

```bash
python scripts/inspect_action_inpainting.py \
  --config-name <config> \
  --checkpoint-dir <checkpoint> \
  --tail-actions-to-keep 20
```

If correlation-aware inpainting is enabled in the implementation, also run:

```bash
python scripts/inspect_action_inpainting.py \
  --config-name <config> \
  --checkpoint-dir <checkpoint> \
  --tail-actions-to-keep 20 \
  --use-action-correlation
```

Expected:

- the script completes
- it reports how the outputs change between baseline, forwarded-actions, and correlation-aware runs
- the recorded notes make the behavior legible to a human reviewer

**Step 5: Run cross-repo smoke checks**

Run a tiny smoke path that exercises:

```python
policy.infer(obs)
policy.infer(obs, initial_actions=previous_tail)
```

Also smoke-check the deployment/runtime boundary if practical:

- broker receives a chunk
- broker caches the intended tail segment
- adapter accepts forwarded `initial_actions`
- unsupported CLI combinations fail early

Expected: all critical paths behave as designed.

**Step 6: Check for regressions**

Run:

```bash
pytest src/openpi/models/pi05_test.py -k "cfg or flow_prefix" -q
```

Expected: existing non-inpainting tests still pass.

**Step 7: Final verification review**

If all verification passes, stop and summarize:

- what changed in `openpi`
- what changed in `alpha-robotics`
- what changed in `missiontracker` / adapter interfaces
- what the action-expert inspection script showed
- what tests and smoke checks passed

Do not create a commit unless explicitly requested.

### Task 12: Nice-to-have follow-ups

**Files:**
- Optional later modifications in `src/openpi/training/config.py`, policy wrappers, docs, and the alpha-robotics deployment runtime

**Step 1: Add config presets**

Add a training/inference config that enables:

- `time_threshold_inpaint=0.3`
- `use_correlation_inpainting=True`
- `correlation_beta=0.5`
- runtime defaults for `openpi_forward_tail_actions`, `openpi_tail_actions_to_keep`, and `openpi_use_action_correlation`

**Step 2: Add a rollout helper wrapper**

Create a reusable wrapper class that stores:

- `actions_to_execute`
- `actions_to_keep`
- `next_initial_actions`
- `forward_tail_actions`
- `use_action_correlation`

**Step 3: Add benchmarks**

Measure:

- latency overhead
- chunk smoothness
- effect on grasp-sensitive tasks

**Step 4: Keep out of first implementation**

Do not add:

- action-prefix transformer tokens
- retraining-only changes
- stage tracking coupling

**Step 5: Follow-up review**

Only if these follow-ups are actually implemented, summarize them.

---

## Implementation status and deviations from original plan

> Added after implementation by the executing agent. Read this section
> first to understand what was built and what differs from the tasks above.

### Completed work (Tasks 1–12)

All core tasks are implemented and tested (54 unit tests passing).

**Key files created/modified in `openpi`:**

| File | What |
|---|---|
| `src/openpi/models/action_inpainting.py` | **New.** Shared helpers: `build_inpainting_indices`, `pad_initial_actions`, `apply_hard_inpainting`, `apply_correlated_inpainting`, `precompute_correction_matrix`, `prepare_inpainting_state`, `should_apply_inpainting`, `load_correlation_cholesky` |
| `src/openpi/models/action_inpainting_test.py` | **New.** 23 tests for the above helpers |
| `src/openpi/shared/normalize.py` | Added `action_correlation_cholesky` field to `NormStats`; added `compute_action_correlation_cholesky()` utility |
| `src/openpi/shared/normalize_test.py` | 4 new tests for Cholesky serialization and computation |
| `src/openpi/models/model.py` | Added `supports_initial_actions: ClassVar[bool] = False` to `BaseModel` |
| `src/openpi/models/pi0.py` | `supports_initial_actions = True`; `_denoise_actions` accepts `initial_actions`; wired hard + correlated inpainting into Euler loop; stores `use_correlation_inpainting`, `correlation_beta`, `_correlation_cholesky` from config |
| `src/openpi/models/pi0_config.py` | Added `time_threshold_inpaint`, `use_correlation_inpainting`, `correlation_beta` |
| `src/openpi/models/pi0_test.py` | 5 new tests for initial_actions plumbing |
| `src/openpi/models/pi05.py` | Same changes as Pi0 (supports flag, denoising loop, config fields) |
| `src/openpi/models/pi05_config.py` | Same config additions as Pi0Config |
| `src/openpi/models/pi05_test.py` | 5 new tests for initial_actions plumbing |
| `src/openpi/policies/policy.py` | `Policy.infer` accepts `initial_actions`; normalizes them to model space via extracted `_action_normalizer` before forwarding to `model.sample_actions` |
| `src/openpi/policies/policy_test.py` | 6 new tests for forwarding and rejection |
| `src/openpi/policies/policy_config.py` | Loads Cholesky from norm_stats and attaches to model when `use_correlation_inpainting` is enabled |
| `scripts/compute_norm_stats.py` | Correlation Cholesky and per-timestep stats now computed by default (both `True`); single-pass computation |
| `docs/remote_inference.md` | Added "Action inpainting (chunk overlap)" section with usage example |

### Deviations from the original plan

1. **Pi0 support added (not just Pi05).** The plan only mentioned Pi05, but the
   user requested Pi0 support too. All inpainting logic lives in the shared
   `action_inpainting.py` module; both models call the same helpers.

2. **`Policy.infer` normalizes `initial_actions` automatically.** The plan did
   not specify where normalization happens. The implementation extracts the
   `Normalize` transform from the input pipeline at init time and applies it to
   `initial_actions` before passing to the model. Callers pass actions in
   output (real) space — the same space as the policy output.

3. **Runtime integration implemented in alpha-robotics.** The plan listed this
   under Task 10 but left the approach open. The implementation:
   - `openpi_utils.py` — `AdapterChunkBroker` gained an `action_inpainting`
     flag. When set, it saves `chunk[actions_to_execute:]` and forwards it as
     `initial_actions` on the next predict call.
   - `openpi_adapter.py` — `OpenPiAdapter.predict` accepts
     `initial_actions: np.ndarray | None` and forwards to `policy.infer`.
   - `deploy_policy.py` — Added `--openpi-action-inpainting` CLI flag. Guards
     against CFG + inpainting combination at argument parsing time.

4. **`compute_norm_stats.py` now computes everything by default.** The original
   plan had correlation as opt-in (`--compute-action-correlation`). Per-timestep
   stats lived in a separate script. Both are now default (`True`), computed in
   one pass. Opt out with `--no-compute-action-correlation` or
   `--no-compute-per-timestep`.

5. **Task 8.5 deferred.** `scripts/inspect_action_inpainting.py` (the visual
   comparison script) was not created. This should be done when a trained
   checkpoint is available for evaluation.

### Remaining work

- **`scripts/inspect_action_inpainting.py`** — Create the inspection script
  for visual comparison of baseline vs inpainted vs correlation-aware outputs.
  Requires a trained checkpoint.
- **Config presets** — Add a training config that sets
  `time_threshold_inpaint=0.3`, `use_correlation_inpainting=True`,
  `correlation_beta=0.5` for quick experimentation.
- **Benchmarks** — Measure latency overhead, chunk smoothness, effect on
  grasp-sensitive tasks.
- **`compute_norm_stats_per_timestep.py`** — Consider deprecating now that the
  main script handles per-timestep stats.

---

## Post-implementation review (2026-04-10)

Systematic review of all changes found two critical bugs and one coverage gap.
All have been fixed and verified with passing tests.

### Bug 1 (critical): JAX tracing crash in denoising loops

**`pi0.py` and `pi05.py`**: Inside the `jax.lax.while_loop` body, the code
used a Python `if` on the result of `should_apply_inpainting(time_new, ...)`.
Since `time` is a traced loop-carry value, `time_new > threshold` returns a
traced boolean — and Python `if` on a tracer raises `ConcretizationTypeError`
at JIT trace time.

**Fix**: Replaced with `jnp.where(should_apply, x_t_inpainted, x_t_new)` which
always computes both branches and selects with a traced boolean. The inpainting
ops are negligible vs. the transformer forward pass, so no performance impact.

### Bug 2 (high): Missing delta-action conversion for initial_actions

**`policy.py`**: `Policy.infer()` only applied `Normalize` to initial_actions,
but many configs (bin pick pack, block tower, Aloha, Libero) use delta actions.
The model expects normalized deltas, but received normalized absolutes —
producing silently wrong inpainting targets.

**Fix**: `Policy.__init__` now also extracts `DeltaActions`/`DeltaActionsFromState`
from the input transform list. `infer()` applies delta conversion (using the
raw observation state) before normalization:
```
initial_actions → DeltaActions(state) → Normalize → model
```
This matches the full input pipeline: `Repack → Prompt → DeltaActions →
Normalize → ModelTransforms`, of which only the action-relevant subset is needed.

### Coverage gap: end-to-end denoising loop test

Added `test_pi0_denoise_actions_end_to_end_with_initial_actions` which runs
the actual `while_loop` with `initial_actions`. This would have caught Bug 1.
Also added `test_policy_infer_applies_delta_and_normalize_to_initial_actions`
which verifies the delta+normalize conversion with concrete expected values.

### alpha-robotics glue code verified

The `AdapterChunkBroker` in `openpi_utils.py` correctly:
- Saves the un-executed chunk tail as `next_initial_actions` (absolute positions)
- Passes it to `adapter.predict()` → `policy.infer()`
- The new delta+normalize conversion handles the output→model space conversion

The `deploy_policy.py` changes (CLI flags, CFG support) are compatible.
No issues found in the glue code.
