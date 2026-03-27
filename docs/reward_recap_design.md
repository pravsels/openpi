# `reward_recap`: Design And Implementation Notes

This document records our current understanding of RECAP, why we think a reward-model-based
variant makes sense, and how we intend to implement it.

The intended audience is an engineer who was not in the original discussion but now needs to
build the system. The document therefore tries to be:

- concise about motivation
- concrete about data, labels, and code paths
- explicit about what is decided vs still open

## 1. Goal

Our north star is the RECAP idea from the paper:

- use a signal about action quality
- convert that signal into a binary improvement indicator `I_t`
- train a policy that can learn from both better and worse data

However, we are making one deliberate architectural change:

- we are **not** training a separate value function
- the long-term target uses Robometer as the source of quality / progress signal instead

So this document should be read as:

- paper RECAP as the conceptual reference
- `reward_recap` as our intentionally modified implementation
- the long-term target design uses Robometer-based labeling
- the first bootstrap experiment for `bin_pick_pack` uses control-mode metadata to assign labels

Instead of using the paper’s value-function stage, we want to use an existing reward model from
`../robometer` that already produces:

- a progress signal in `[0, 1]`
- a success / failure signal

from short windows of images.

We are calling this variant:

- `reward_recap`

The intended picture is:

- paper RECAP:
  - define reward
  - train value function on returns
  - compute advantages
  - threshold into binary improvement indicator `I_t`
  - train policy conditioned on `I_t`
- `reward_recap`:
  - run Robometer on trajectory chunks
  - derive future-gain scores from the coarse Robometer progress trajectory
  - threshold those scores into binary improvement indicator `I_t`
  - train policy conditioned on `I_t`
- `reward_recap` bootstrap for `bin_pick_pack`:
  - read per-episode control-mode annotations
  - assign `I_t = 0` only for chunks explicitly marked `mode: policy`
  - assign `I_t = 1` for everything else
  - train policy conditioned on `I_t`

So the main change is:

- keep the policy-conditioning idea
- replace the value-function stage with a reward-model stage

## 2. What RECAP Is Doing Conceptually

The reference paper is:

- [π0.6*: a VLA That Learns From Experience](https://www.pi.website/download/pistar06.pdf)

At a high level, RECAP is an offline-RL-style policy improvement recipe:

1. collect data
2. train a value function
3. compute an advantage for each example
4. threshold that advantage into a binary indicator
5. train a policy conditioned on that indicator

In the paper, the binary indicator answers:

- "is this action/chunk better than expected from this state?"

The paper’s exact indicator rule is:

- `I_t = 1` when `A^{π_ref}(o_t, a_t, ℓ) > ε_ℓ`

where:

- `A^{π_ref}(o_t, a_t, ℓ)` is the advantage under the reference policy
- `ε_ℓ` is a task-dependent threshold

In plain English, advantage means:

- what actually happened after taking action `a_t`
- minus what the value function expected would happen from state / observation `o_t`

So advantage is not just:

- "is this state good?"

It is:

- "did this action lead to a better or worse future than expected from this state?"

Positive advantage means:

- the future after this action was better than the value-function baseline

Negative advantage means:

- the future after this action was worse than the value-function baseline

More specifically, the paper says in Section V-C that for each episode they obtain a **success / failure label**, and then derive the reward from that episode-level outcome:

In words:

- every non-terminal step gets a reward of `-1`
- if the episode ends in success, the terminal step gets `0`
- if the episode ends in failure, the terminal step gets `-C_fail`

So successful episodes accumulate something like "negative remaining steps until success," while failed episodes get an additional large negative terminal penalty.

Equivalent reward definition:

- `r_t = -1` for every non-terminal step
- `r_T = 0` if the episode terminates in success
- `r_T = -C_fail` if the episode terminates in failure

So the paper’s setup is **not** "purely terminal success only" in the strict mathematical sense,
because it adds a shaped `-1` step cost. But it also is **not** a rich dense supervision signal in
the way Robometer is. The underlying human-provided label is still episode-level success/failure,
and the intermediate reward is a generic construction derived from that label, not a learned
per-frame notion of semantic task progress.

That is why the value function is still important in RECAP:

- it turns this generic return signal into a state-local estimate
- in the paper, that estimate is the expected time-to-completion / return
- that estimate is then used to compute advantage and the binary improvement indicator

That is exactly where `reward_recap` departs from the paper:

- paper RECAP learns a value function and derives `I_t` from advantage
- `reward_recap` derives `I_t` directly from Robometer score trajectories

## 3. Why A Reward-Model Variant Makes Sense

The reward model in `../robometer` gives us a richer task-progress signal than the paper’s generic
reward construction.

Our working assumption is:

- if Robometer already gives a sufficiently informative progress / success signal
- then we may not need a separate learned value model in order to decide whether a chunk is
  positive or negative

So in our design, Robometer is not just a reward helper for another model. It is the primary
source of chunk-quality supervision.

What we know from `../robometer`:

- default config is in `../robometer/robometer/configs/config.yaml`
- default `max_frames` is `16`
- default `use_multi_image` is `true`
- default `use_per_frame_progress_token` is `true`
- default progress loss is discrete with `10` bins

The main model classes are:

- `RBM` in `../robometer/robometer/models/rbm.py`
- prediction heads in `../robometer/robometer/models/heads.py`

Important outputs:

- progress predictions
- success predictions

In `../robometer/robometer/models/rbm.py`, the relevant internal path produces:

- `progress_output`
- `success`

per temporal step / frame group.

In `../robometer/robometer/evals/eval_utils.py`, helper functions like
`extract_rewards_from_output(...)` and `extract_success_probs_from_output(...)` currently use
the **last** value of each subsequence as the reward-like / success-like summary.

For `reward_recap`, we do **not** want to rely only on the last scalar if we can avoid it.
The more useful object is the **16-frame progress evolution**, because it lets us decide whether
the chunk is:

- improving
- stalling
- regressing
- recovering

That is a better substrate for direct chunk labeling than a single endpoint score.

## 4. Data Sources We Expect To Use

We expect to have several types of trajectories:

1. expert demonstration data
2. DAgger-style trajectories with autonomous behavior followed by human correction
3. autonomous successful rollouts
4. autonomous failed rollouts
5. external failure data

This matters because the relabeling pipeline should work across mixed-quality data from multiple
sources.

Even though the long-term design does **not** force labels from source metadata, we still expect
these sources to contain very different behavior patterns:

- expert data is often cleaner and more goal-directed
- autonomous rollouts can include both progress and regressions
- DAgger-style data may contain both failure drift and recovery behavior
- external failure data may look very different from in-distribution success data

So `reward_recap` has two layers:

- long-term target: reward-model driven, with labels derived from Robometer outputs
- first `bin_pick_pack` bootstrap: metadata driven, with labels derived from explicit control-mode
  spans
- with optional human curation to exclude data where the Robometer reward curves look untrustworthy

## 5. Core Decision: Chunk-Level Labels, Not Episode-Level Labels

We do **not** want to label entire episodes as simply positive or negative.

That would throw away the most useful structure in DAgger-style and mixed-quality data.

Instead, the unit of labeling should be:

- a short chunk / window of frames

Matching Robometer’s natural inference granularity, the default coarse trajectory length is:

- `16` sampled temporal positions

For each Robometer run, we will compute:

- a coarse progress trajectory over `16` sampled temporal positions
- a coarse success-classifier trajectory over those same sampled positions
- metadata about the chunk source

Then we will derive:

- binary conditioning labels `I_t` for coarse intervals within that trajectory

This is the key design principle of `reward_recap`.

### 5.1 Exact alignment rule

Robometer takes a longer underlying frame sequence and compresses it into a **16-step coarse
trajectory** by uniformly subsampling the original frames.

So there are two time axes:

- the original dense episode frames
- the `16` coarse Robometer steps

For version 1, we treat each adjacent pair of coarse Robometer steps as one labelable interval.

That means:

- one Robometer run yields a coarse score trajectory `r_0, r_1, ..., r_T`
- each coarse interval `t -> t+1` corresponds to a chunk of the original episode
- each such coarse interval gets its own label `I_t`
- valid labeled indices are therefore `t = 0, ..., T-1`
- the terminal coarse point `T` has no associated interval label

If we save the subsampled frame indices from Robometer preprocessing, then the interval label `I_t`
is attached to the underlying episode span between those adjacent sampled indices.

## 6. Raw Signals Versus Derived Labels

We should separate two things cleanly:

### Raw signals from Robometer

These are model outputs:

- progress trace / progress bins
- progress score in `[0, 1]`
- per-frame binary success-classifier output
- in the current inference helpers, this is exposed as a per-frame success probability after applying sigmoid to the classifier logits

### Derived policy-training labels

These are our constructed labels:

- `I_t = 1` means positive condition
- `I_t = 0` means negative condition

Here:

- `r_t` means the Robometer progress score at coarse step `t`
- `g_t^(H)` means the future gain computed from the coarse Robometer trajectory

In our implementation, the dependency is:

```text
Robometer coarse score trajectory -> future gain g_t^(H) -> quantile-derived threshold -> I_t
```

So:

- Robometer is the measurement system
- `reward_recap` is the label-construction and policy-training system

## 7. Proposed Label Construction

We currently recommend a direct two-stage construction:

1. compute future-gain scores from the coarse Robometer trajectory
2. compute a quantile-derived threshold from those scores
3. assign binary labels `I_t` based on that quantile-derived threshold

### 7.1 Robometer coarse trajectory

Robometer gives a coarse score trajectory over the downsampled temporal positions in the input
sequence.

For version 1, define:

- `r_t`: the Robometer progress score at coarse step `t`
- `T`: the final coarse step in the Robometer trajectory
- `H`: the future horizon in coarse steps

Use the convention:

- if `H = -1`, then `end(t, H)` means the final coarse step `T`
- otherwise, `end(t, H) = min(t + H, T)`

Default:

- `H = -1`

meaning:

- look all the way to the end of the coarse Robometer trajectory

### 7.2 Future-gain score

Define the score for coarse step `t` as:

```text
g_t^(H) = r_{end(t, H)} - r_t
```

This score is only defined for labeled interval starts:

- `t = 0, ..., T-1`

This means:

- if the Robometer score increases from step `t` to the chosen future horizon, `g_t^(H)` is positive
- if it decreases, `g_t^(H)` is negative

With the default `H = -1`, the score becomes:

```text
g_t = r_T - r_t
```

So version 1 uses the **global future gain** on the 16-step coarse Robometer trajectory.

Why this choice:

- it is closer to the RECAP intuition of "did things get better after this point ?"
- it is more stable than a one-step delta
- it lets failure episodes pull later-bad behavior negative
- it accepts that some locally good early moves in failed episodes may still be weighted down

### 7.3 Threshold labeling

The labeler should use this exact rule:

1. compute `g_t^(H)` from the Robometer coarse trajectory
2. choose a quantile-derived threshold `epsilon_task`
3. assign:
   - `I_t = 1` if `g_t^(H) > epsilon_task`
   - `I_t = 0` if `g_t^(H) <= epsilon_task`

So in version 1:

- above threshold -> positive
- below threshold -> negative

There is no ignored middle bucket in this design.

### 7.4 Bootstrap simplification for `bin_pick_pack`

For the first implementation on `bin_pick_pack`, we are intentionally using a simpler labeler
before integrating Robometer.

In this bootstrap experiment, `I_t` is derived from control-mode metadata rather than from
Robometer scores.

Preferred source of control-mode labels:

- first, read the per-sample `control_mode` field exposed by the `robocandywrapper`
  `ControlModePlugin`
- if that field is unavailable or unreliable for a given dataset version, fall back to raw
  per-episode mode-span metadata

This is preferable because it lets the starter implementation consume the same wrapped dataset path
already used by training, instead of duplicating control-mode parsing immediately.

Fallback raw annotation format:

```json
{
  "0": [
    {"start_index": 0, "end_index": 190, "mode": "policy"},
    {"start_index": 191, "end_index": 242, "mode": "human"},
    {"start_index": 243, "end_index": 318, "mode": "policy"}
  ]
}
```

Bootstrap labeling rule:

- if `control_mode == "policy"`, assign `I_t = 0` and use `"Advantage: negative"`
- if `control_mode == "human"`, assign `I_t = 1` and use `"Advantage: positive"`
- if `control_mode` is missing, unknown, or there is no matching fallback mode span, assign
  `I_t = 1`
- equivalently: anything not explicitly marked `"policy"` is treated as advantage-positive

Bootstrap experiment split:

- run A: positive-only conditioning
  - force every training example to `"Advantage: positive"`
  - do not use negative conditioning at all
- run B: mixed positive / negative conditioning
  - use the bootstrap labeling rule above
  - `policy` becomes negative, everything else becomes positive

For a clean comparison between the two runs:

- use the same dataset
- use the same starting checkpoint
- change only the label-assignment rule

Recommended alignment rule for the starter implementation:

- preferred path: use the `control_mode` value already attached to the training sample by the
  plugin
- fallback path: determine the label from the chunk's start index and look up the raw mode span
  covering that start index
- default to positive when there is no explicit `policy` signal

This reflects the initial working assumption for the bootstrap experiment:

- human teleoperation is advantage-positive
- unlabeled data is assumed to be teleoperated and therefore advantage-positive
- only behavior explicitly marked as policy-generated is advantage-negative

## 8. Role Of Metadata

For the long-term Robometer design, source metadata is not part of the labeling rule.

However, for the first `bin_pick_pack` bootstrap experiment, control-mode metadata **is** part of
the labeling rule, preferably via the plugin-exposed `control_mode` field and otherwise via
explicit control-mode spans.

That means:

- the policy still does not see raw metadata directly
- the policy only sees the binary prompt condition
- the bootstrap labeler does force labels from control-mode metadata
- specifically, explicit `mode: policy` means negative and everything else means positive

We still keep source metadata because it is useful for:

- offline analysis
- debugging label quality
- slicing evaluation metrics by data source

## 9. Proposed Training Recipe

The training recipe should preserve the useful parts of RECAP while replacing the value-function
stage with offline labeling. The long-term target uses Robometer; the first `bin_pick_pack`
bootstrap uses control-mode metadata.

### Stage 0: Offline relabeling

1. collect all trajectory sources
2. for the first `bin_pick_pack` experiment, load control mode from the dataset/plugin path
3. if the plugin field is unavailable, fall back to per-episode control-mode annotations
4. assign `I_t` with the bootstrap rule:
   - explicit `control_mode == "policy"` -> `I_t = 0`
   - anything else -> `I_t = 1`
5. write out:
   - `I_t`
   - source metadata
   - plugin-derived or fallback mode metadata used to derive the label
6. later, replace this bootstrap labeler with Robometer-based relabeling

This stage should be a preprocessing step, not on-the-fly training logic.

### Stage 1: Initialize from an existing task-trained checkpoint

Purpose:

- start from a `pi05` checkpoint that was already trained normally for the target task
- examples include a task specialist such as `bin_pick_pack`
- use that existing task policy as the initialization for `reward_recap`

Version-1 assumption:

- do **not** plan around a separate warmup / SFT phase
- the existing task-trained checkpoint already plays that role

### Stage 2: Advantage-conditioned fine-tuning

Train the policy with the binary label `I_t` on all relabeled coarse intervals, starting from the
existing task-trained checkpoint from Stage 1.

At this stage:

- prepend the binary condition text
- keep the same supervised action loss as the base policy
- train on both `I_t = 1` and `I_t = 0` examples with the same loss form

At inference time:

- condition on positive

Version-1 simplifications:

- no unconditional branch
- no conditioning dropout
- no classifier-free guidance

So the policy should be sampled in "good behavior" mode.

Paper note:

- this differs from the paper, where `I_t` is derived from value-based advantage
- here we are using Robometer-derived offline labels instead

## 10. Policy Interface We Intend To Use

We should keep the policy-side interface simple and RECAP-like:

- one binary conditioning variable `I_t`

We are **not** currently proposing to condition on:

- raw progress scalar
- raw success score
- multi-bin reward classes

Reasons:

- binary conditioning is closer to RECAP
- simpler to implement
- simpler to reason about at inference time
- easier to compare against the original RECAP idea

If needed later, we can experiment with richer conditioning:

- progress bucket
- signed progress delta bucket
- success score bucket

But binary should be the first version.

### Policy conditioning

In the paper, the improvement indicator is represented as text input:

- `"Advantage: positive"`
- `"Advantage: negative"`

and is inserted into the VLA token sequence before action prediction.

We want to preserve the simple binary conditioning interface from the paper, even though the label
source is different.

For `reward_recap`, use the same wording:

- `"Advantage: positive"`
- `"Advantage: negative"`

`../exla_openpi/src/fla/recap/pi0_recap.py` can still be consulted as a reference point, but it is
not the spec.

### Decision: use the original `pi05` prompt path

For version 1, we are explicitly targeting the original open-source `pi05` conditioning path in
upstream `openpi`, not the later subtask / high-low prompt / FAST-token training extensions that
were added in this fork.

That means:

- use the standard `prompt` field
- use the standard `ModelTransformFactory` path for `ModelType.PI05`
- use `_transforms.TokenizePrompt(...)`
- do **not** use `SubtaskModelTransformFactory`
- do **not** use `TokenizeHighLowPrompt(...)`
- do **not** use FAST-token training machinery as part of `reward_recap` conditioning

The practical implication is:

- version 1 does **not** need a new reward-conditioned `pi05` model architecture
- version 1 should inject the condition as plain text into the existing `pi05` prompt path

Concrete prompt form for version 1:

- positive: `<task prompt>. Advantage: positive`
- negative: `<task prompt>. Advantage: negative`

So the condition is still represented with the paper-aligned strings:

- `"Advantage: positive"`
- `"Advantage: negative"`

but it is carried through the ordinary `pi05` prompt-tokenization path rather than any later
hierarchical prompt path.

## 11. Concrete Implementation Plan

This section is intentionally low-level enough that an engineer should be able to turn it
into code.

### 11.1 New relabeling pipeline

Create an offline script in this repo, likely under:

- `scripts/`

Candidate name:

- `scripts/label_reward_recap.py`

Responsibilities for the first implementation:

1. load trajectories from `bin_pick_pack`
2. read `control_mode` from the `robocandywrapper` plugin output when available
3. fall back to per-episode control-mode annotations only when the plugin signal is unavailable
4. assign `I_t` from control mode
5. save relabeled dataset or per-interval metadata

Later extension:

6. replace or augment the metadata labeler with Robometer inference

### 11.2 Robometer adapter (later, not first pass)

Likely wrap or reuse:

- `../robometer/scripts/example_libero_robometer_wrapper.py`
- `../robometer/robometer/evals/eval_utils.py`

We should isolate this behind a small adapter in this repo, e.g.:

- `src/openpi/reward_recap/robometer_adapter.py`

Responsibilities:

- format chunk inputs for Robometer
- batch inference
- return raw structured outputs:
  - coarse progress trajectory
  - coarse success trajectory
  - sampled frame indices

This adapter is part of the long-term target design, but it is **not** required for the first
`bin_pick_pack` bootstrap implementation.

### 11.3 Labeling logic

Create:

- `src/openpi/reward_recap/labeling.py`

Responsibilities:

- first pass: map mode spans to `I_t`
- implement the bootstrap rule:
  - explicit `mode: policy` -> negative
  - anything else -> positive
- later: compute `g_t^(H)` from the Robometer coarse trajectory
- later: implement the threshold rule for Robometer labels
- export `I_t`

This module should contain the exact labeling rules in one place.

### 11.4 Policy conditioning

Version 1 should reuse the existing `pi05` model and prompt pipeline rather than introduce a new
reward-conditioned model file.

Do **not** do the following in version 1:

- do **not** create `src/openpi/reward_recap/pi0_reward_recap.py`
- do **not** modify `src/openpi/models/pi05.py` to add a new conditioning branch
- do **not** route reward conditioning through `TokenizeHighLowPrompt(...)`
- do **not** use the subtask / FAST-token training path for `reward_recap`

Instead, make the condition flow through the existing prompt path.

#### Exact change we will make

1. In `src/openpi/transforms.py`, add a small input transform, e.g. `InjectAdvantagePrompt`, that:
   - reads the raw task `prompt`
   - reads the relabeled binary indicator `I_t`
   - rewrites the prompt text to append exactly one of:
     - `"Advantage: positive"`
     - `"Advantage: negative"`
2. The recommended rewrite rule is:
   - base prompt `pick up the fork` -> `pick up the fork. Advantage: positive`
   - base prompt `pick up the fork` -> `pick up the fork. Advantage: negative`
3. In `src/openpi/training/config.py`, add a reward-recap-specific transform path for
   `ModelType.PI05` that inserts `InjectAdvantagePrompt` immediately before
   `_transforms.TokenizePrompt(...)` in the standard `ModelTransformFactory` prompt pipeline.
4. In `scripts/label_reward_recap.py`, for the starter `bin_pick_pack` experiment:
   - first read `control_mode` from the `robocandywrapper` plugin field already present on the
     dataset sample
   - if that field is missing or unusable, fall back to the raw mode-span JSON
   - support two modes:
     - positive-only: assign `I_t = 1` for every example
     - mixed: assign `I_t = 0` only for examples explicitly marked `policy`, otherwise `I_t = 1`
5. In `scripts/train_reward_recap.py`, ensure each relabeled training example carries that binary
   label and feeds it into the prompt-rewrite transform.
6. In the relabeled dataset writer / loader, store the binary label as a dataset field, but do not
   expose raw mode metadata or Robometer scalars directly to the policy.

Resulting data flow:

```text
prompt + I_t -> InjectAdvantagePrompt -> TokenizePrompt -> tokenized_prompt -> existing pi05 model
```

Important non-changes for version 1:

- `src/openpi/models/tokenizer.py`: no new high/low prompt path for `reward_recap`
- `src/openpi/models/pi05.py`: no model-architecture change for `reward_recap`
- `src/openpi/models/pi05.py`: keep the existing flow-matching action head
- `src/openpi/training/config.py`: continue to use the standard `ModelTransformFactory`, not the
  subtask-specific factory, for `reward_recap`

### 11.5 Training entrypoint

Add a dedicated training script rather than overloading `scripts/train.py` immediately.

Candidate:

- `scripts/train_reward_recap.py`

Phases:

1. load `bin_pick_pack` data relabeled with `I_t`
2. load an existing task-trained `pi05` checkpoint for the target task
3. run the positive-only conditioning experiment
4. run the mixed positive / negative conditioning experiment
5. save checkpoints

## 12. Expected Dataset Semantics

The implementer should think of the training data after relabeling as a coarse-interval dataset
derived from Robometer trajectories.

Each row should contain something like:

- source episode / clip identifier
- coarse step index `t`
- underlying start / end frame indices for that coarse interval
- observation fields needed by the policy for the start of that interval
- target action chunk aligned to that interval
- source type:
  - demo
  - correction
  - autonomous_success
  - autonomous_failure
  - external_failure
- optional raw reward-model outputs for later Robometer integration:
  - coarse Robometer progress trajectory
  - optional coarse success trajectory
- optional control-mode metadata:
  - plugin-derived `control_mode` when available
  - per-episode spans with `start_index`, `end_index`, and `mode`
- derived fields:
  - optional `g_t^(H)` when using the Robometer labeler
  - `I_t`

This makes the system reproducible and auditable.

## 13. Open Questions

These are follow-up questions for the longer-term Robometer-based design. They should not block the
current metadata-bootstrap implementation for `bin_pick_pack`.

### Q1. Exactly how should the score be defined?

For version 1, this is now decided: `g_t^(H) = r_{end(t, H)} - r_t`, with default `H = -1`.

Open follow-up:

- should shorter horizons perform better than the default global horizon?

### Q2. What threshold should we use?

Current recommendation:

- use a single threshold `epsilon_task`
- derive `epsilon_task` from a quantile of the `g_t^(H)` distribution
- label `g_t^(H) > epsilon_task` as positive
- label `g_t^(H) <= epsilon_task` as negative

For our usual one-task-at-a-time training setup, `epsilon_task` is just the fixed threshold for
that task run.

### Q3. Should we ever exclude uncertain examples?

Current recommendation:

- no, not in the core thresholded design
- if Robometer curves look untrustworthy for some data, exclude that data during human curation

### Q4. Should labels depend on dataset source?

Current answer:

- long-term Robometer design: no
- first `bin_pick_pack` bootstrap: yes, intentionally
- explicit `mode: policy` is negative
- anything else is positive

## 14. Recommended First Experiment

The first implementable version is the bootstrap version described above:

1. start from an existing task-trained `pi05` checkpoint for `bin_pick_pack`
2. load the per-episode control-mode annotations
3. run experiment A with every sample forced to `"Advantage: positive"`
4. run experiment B with the default-positive bootstrap rule:
   - explicit `mode: policy` -> negative
   - anything else -> positive
5. inject `"Advantage: positive"` / `"Advantage: negative"` into the ordinary `pi05` prompt path
6. fine-tune the same starting checkpoint for both runs
7. keep the underlying `pi05` architecture unchanged
8. compare positive-only vs mixed conditioning
9. after this bootstrap works, replace the metadata labeler with the Robometer labeler

## 15. Bottom Line

`reward_recap` is a RECAP-inspired conditioning method that replaces the paper’s value-function
stage with offline labels used to drive binary policy conditioning. The long-term target design uses
Robometer-based labels. The first `bin_pick_pack` bootstrap experiment instead uses control-mode
metadata, treating anything not explicitly marked `mode: policy` as advantage-positive. For `pi05`,
the bootstrap version starts from an existing task-trained checkpoint, reuses the original upstream
prompt path, injects `"Advantage: positive"` / `"Advantage: negative"` into the prompt via a data
transform, and keeps the underlying model architecture unchanged.

## References

- [π0.6*: a VLA That Learns From Experience](https://www.pi.website/download/pistar06.pdf)
- `../robometer/robometer/configs/config.yaml`
- `../robometer/robometer/models/rbm.py`
- `../robometer/robometer/models/heads.py`
- `../robometer/robometer/evals/eval_utils.py`
- `../robometer/scripts/example_libero_robometer_wrapper.py`
- `../exla_openpi/src/fla/recap/pi0_recap.py`
- `../exla_openpi/scripts/train_recap_full.py`
- [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
