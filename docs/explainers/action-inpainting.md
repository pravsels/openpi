# Action Inpainting in Flow Matching

## The vanilla flow

Flow matching generates actions by starting from pure noise (**t = 1**) and
walking toward clean actions (**t = 0**) along a straight line.  The model
predicts a velocity at each step, and you simulate with Euler integration:

```
x_{t-dt} = x_t + dt · v_t      (dt is negative)
```

## What inpainting adds

Sometimes you already know the first **k** action timesteps in a chunk of,
say, 50 (e.g. from a previous prediction that partially overlaps).  Inpainting
says: *pin those known coordinates to the correct trajectory while the model
freely generates the rest.*

Flatten the `(action_horizon, action_dim)` grid into a 1D vector.  Split its
indices into two sets:

| Set | Name | Meaning |
|-----|------|---------|
| **O** | Observed / constrained | First k timesteps × provided action dims |
| **U** | Unobserved / free | Everything else |

After every Euler step the code overwrites the **O** entries and leaves the
**U** entries untouched.

### Common confusion: "what happens to the other 50 − k coordinates?"

Nothing special.  The model's Euler step stands for those — they are freely
denoised from noise.  The model paints the whole canvas, and then you paste the
known region back on top.  Only the O indices get overwritten.

## Why not just paste in the clean values?

When you first see the overwrite formula:

```
x_desired_O = (1 - t) · x₀ + t · z
```

a natural reaction is: *"Why not just set them to x₀?  We know what they should
be."*

### Common confusion: "this looks like denoising the k actions, not constraining them"

It looks like a denoising step, but it's doing the opposite — it *replaces*
whatever the model produced for those coordinates.  The Euler step already
updated all coordinates; this line then overwrites the constrained ones with the
value they *should* have at time t.

### Why the interpolation instead of the clean value

At intermediate time **t**, the free coordinates are still partly noisy.  If
you set the constrained coordinates to their fully clean values while
everything else is half-noise, the model sees a state it was never trained on —
some coordinates clean, others noisy.

Instead, constrained coordinates follow the **same interpolation schedule** as
the rest of the flow:

```
x_constrained(t) = (1 - t) · x₀ + t · z
```

- At **t = 1**: pure noise (`z`), same as the free coordinates.
- At **t = 0.5**: a 50/50 mix of clean and noise — appropriately noisy for that
  point in the flow.
- At **t = 0**: fully clean (`x₀`).

The constrained coordinates arrive at the known clean values at the end, but
they get there gradually, at the same pace as the denoising.  The model never
sees anything out of distribution.

The noise `z` is fixed once at the start (the same noise tensor used to
initialize the whole flow) and reused at every step — that's why
`prepare_inpainting_state` stores `z_flat` up front.

### Common confusion: "why does the model need to see the constrained coordinates at all?"

If we already know their values and just overwrite the model's output, why
not strip them from the input entirely?  Because the model's prediction for
the **free** coordinates depends on what it sees in the **constrained** ones.
The constrained coordinates provide context — they tell the model "here's where
the trajectory has been so far, now continue it."  If you hid them, the model
would be generating the continuation blind.

## The threshold: healing the seam

There is a `time_threshold_inpaint` parameter.  Below this threshold, the
constraint is released and the model runs freely on *all* coordinates.

### Common confusion: "so the first k actions don't need to be exact?"

It's less about accepting approximate values and more about the **boundary**
between constrained and free regions.  If you enforce the constraint all the
way to t = 0, the model never gets a chance to smooth out the junction between
the last constrained action and the first free one — you can get a jarring
discontinuity.

Releasing the constraint in the final few steps (when everything is already
nearly clean) lets the model nudge all coordinates into something globally
coherent.  The constrained values barely move because they're already close to
x₀, but the model gets to soften the transition.

Think of it like pasting a photo into a scene: you place it precisely, then
feather the edges so it blends.

## One-sentence summary

Flow matching walks all coordinates from noise to clean along straight lines.
Inpainting pins the known coordinates to their correct line at each step (so
the model sees a consistent noise level), then releases them near the end so
the model can heal the seam.

## Usage notes

- `initial_actions` is passed in the same (unnormalized) action space as the
  policy output. Delta conversion and normalization are handled internally by
  `Policy.infer`.
- This is sampler-level constrained generation — it overrides coordinates in
  the denoising loop, not prompt conditioning.
- CFG + inpainting is not yet supported; enabling both will raise
  `NotImplementedError`.

## Code map

| File | Role |
|------|------|
| `src/openpi/models/action_inpainting.py` | Shared helpers: index building, padding, the overwrite formula |
| `src/openpi/models/pi0.py` (`_denoise_actions`) | Euler loop that calls inpainting after each step |
| `src/openpi/models/pi0_config.py` | Holds `time_threshold_inpaint` |
