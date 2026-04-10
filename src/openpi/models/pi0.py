import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import action_inpainting as _inpaint
from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def _validate_initial_actions(initial_actions: jax.Array, action_dim: int, action_horizon: int) -> None:
    """Validate initial_actions shape against model dimensions."""
    if initial_actions.ndim != 3:
        raise ValueError(
            f"initial_actions must be 3D (batch, timesteps, action_dim), got ndim={initial_actions.ndim}"
        )
    if initial_actions.shape[2] > action_dim:
        raise ValueError(
            f"initial_actions action_dim {initial_actions.shape[2]} exceeds model action_dim {action_dim}"
        )
    if initial_actions.shape[1] > action_horizon:
        raise ValueError(
            f"initial_actions timesteps {initial_actions.shape[1]} exceeds model action_horizon {action_horizon}"
        )


class Pi0(_model.BaseModel):
    supports_initial_actions = True

    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.time_threshold_inpaint = config.time_threshold_inpaint
        self.use_correlation_inpainting = config.use_correlation_inpainting
        self.correlation_beta = config.correlation_beta
        self._correlation_cholesky: jax.Array | None = None
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        per_step_sq = jnp.square(v_t - u_t)  # [*b, ah, ad]

        # Mask padded action dimensions (e.g. right arm for single-arm data)
        if observation.action_dim_mask is not None:
            dim_mask = observation.action_dim_mask  # [*b, ad] or [ad]
            if dim_mask.ndim < per_step_sq.ndim:
                dim_mask = dim_mask[..., None, :]  # broadcast over ah
            num_real_dims = jnp.clip(dim_mask.sum(axis=-1, keepdims=True), 1)
            per_step_sq = per_step_sq * dim_mask
            per_step_loss = per_step_sq.sum(axis=-1) / num_real_dims.squeeze(-1)
        else:
            per_step_loss = jnp.mean(per_step_sq, axis=-1)

        # Mask out padded actions at episode boundaries so the model doesn't
        # learn to predict the repeated last action (which causes freezing).
        if observation.action_is_pad is not None:
            mask = ~observation.action_is_pad
            per_step_loss = per_step_loss * mask
            num_real = jnp.clip(mask.sum(axis=-1, keepdims=True), 1)
            per_step_loss = per_step_loss * (mask.shape[-1] / num_real)

        return per_step_loss

    def build_prefix_cache(
        self, observation: _model.Observation
    ) -> tuple[object, jax.Array]:
        """Cache the prefix (images + language) so denoising steps only run the
        shorter suffix through the LLM."""
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        # Forward pass over prefix only; suffix slot is None so we just fill the cache
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions,
        )
        return kv_cache, prefix_mask

    def _denoise_actions(
        self,
        observation: _model.Observation,
        prefix_mask: jax.Array,
        kv_cache: object,
        *,
        num_steps: int | jax.Array,
        noise: jax.Array,
        velocity_fn=None,
        initial_actions: jax.Array | None = None,
    ) -> _model.Actions:
        """Euler denoising loop from t=1 (noise) to t=0 (clean actions).

        ``velocity_fn(v_t, x_t, time)`` can override the velocity at each step,
        used by CFG to combine conditional + unconditional branches.
        """
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]

        # Pre-compute inpainting state if initial_actions are provided.
        # O_indices: flat indices of constrained coordinates (first k timesteps × provided dims)
        # x0_flat: target clean actions (padded), flattened — what O should equal at t=0
        # z_flat: initial noise (fixed), flattened — what O equals at t=1
        if initial_actions is not None:
            inp = _inpaint.prepare_inpainting_state(
                initial_actions, noise, self.action_horizon, self.action_dim,
            )
            O_indices = inp["O_indices"]
            U_indices = inp["U_indices"]
            x0_flat = inp["x0_flat"]
            z_flat = inp["z_flat"]
        else:
            O_indices = U_indices = x0_flat = z_flat = None

        # Pre-compute the correction matrix for correlation-aware inpainting.
        correction_matrix = None
        if O_indices is not None and self.use_correlation_inpainting and self._correlation_cholesky is not None:
            correction_matrix = _inpaint.precompute_correction_matrix(
                self._correlation_cholesky, O_indices, U_indices,
            )

        def step(carry):
            x_t, time = carry
            # Embed the suffix: state + noisy actions + timestep
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # Suffix self-attention + cross-attention to cached prefix
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            cross_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([cross_mask, suffix_attn_mask], axis=-1)
            # Suffix positions continue from end of prefix
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            # Project action expert output to action space
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            # CFG hook: let caller combine with an unconditional velocity
            if velocity_fn is not None:
                v_t = velocity_fn(v_t, x_t, time)

            # Euler step: x_{t-dt} = x_t + dt * v_t  (dt is negative)
            x_t_new = x_t + dt * v_t

            # Apply inpainting constraint after each Euler step, only while
            # t > threshold.  Below the threshold the model runs freely.
            # NOTE: `time` is a traced loop-carry value, so we must use
            # jnp.where instead of a Python `if` for the threshold check.
            if O_indices is not None:
                time_new = time + dt
                x_t_flat = x_t_new.reshape(batch_size, -1)
                if correction_matrix is not None:
                    x_t_inpainted = _inpaint.apply_correlated_inpainting(
                        x_t_flat, x0_flat, z_flat, O_indices, U_indices,
                        correction_matrix, time_new, self.correlation_beta,
                    )
                else:
                    x_t_inpainted = _inpaint.apply_hard_inpainting(x_t_flat, x0_flat, z_flat, O_indices, time_new)
                x_t_inpainted = x_t_inpainted.reshape(batch_size, self.action_horizon, self.action_dim)
                should_apply = time_new > self.time_threshold_inpaint
                x_t_new = jnp.where(should_apply, x_t_inpainted, x_t_new)

            return x_t_new, time + dt

        def cond(carry):
            _x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        initial_actions: jax.Array | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        if initial_actions is not None:
            _validate_initial_actions(initial_actions, self.action_dim, self.action_horizon)

        kv_cache, prefix_mask = self.build_prefix_cache(observation)
        return self._denoise_actions(
            observation, prefix_mask, kv_cache,
            num_steps=num_steps, noise=noise,
            initial_actions=initial_actions,
        )

    def sample_actions_cfg(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        uncond_observation: _model.Observation,
        *,
        guidance_scale: float = 1.0,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        initial_actions: jax.Array | None = None,
    ) -> _model.Actions:
        """Classifier-Free Guidance: v = v_uncond + scale * (v_cond - v_uncond).

        No early return for scale=1.0 here because this method is JIT-compiled
        and guidance_scale is a traced value.  The formula degenerates to v_cond
        when scale=1.0 anyway.  The adapter-level code avoids calling this path
        entirely when advantage_mode is unconditional.
        """
        if initial_actions is not None:
            raise NotImplementedError(
                "initial_actions is not yet supported with sample_actions_cfg (CFG)"
            )
        observation = _model.preprocess_observation(None, observation, train=False)
        uncond_observation = _model.preprocess_observation(None, uncond_observation, train=False)

        batch_size = observation.state.shape[0]
        if noise is None:
            # Both branches share the same noise for a fair comparison
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Two prefix caches: same images/text, different action_tokenized_prompt
        # (advantage tokens present in cond, absent in uncond)
        cond_kv_cache, cond_prefix_mask = self.build_prefix_cache(observation)
        uncond_kv_cache, uncond_prefix_mask = self.build_prefix_cache(uncond_observation)

        def cfg_velocity(v_cond, x_t, time):
            # Run the unconditional suffix pass against uncond_kv_cache
            uncond_suffix_tokens, uncond_suffix_mask, uncond_suffix_ar_mask, uncond_adarms_cond = self.embed_suffix(
                uncond_observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            uncond_suffix_attn_mask = make_attn_mask(uncond_suffix_mask, uncond_suffix_ar_mask)
            uncond_cross_mask = einops.repeat(
                uncond_prefix_mask, "b p -> b s p", s=uncond_suffix_tokens.shape[1],
            )
            uncond_full_mask = jnp.concatenate([uncond_cross_mask, uncond_suffix_attn_mask], axis=-1)
            uncond_pos = (
                jnp.sum(uncond_prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(uncond_suffix_mask, axis=-1) - 1
            )
            (_, uncond_suffix_out), _ = self.PaliGemma.llm(
                [None, uncond_suffix_tokens],
                mask=uncond_full_mask,
                positions=uncond_pos,
                kv_cache=uncond_kv_cache,
                adarms_cond=[None, uncond_adarms_cond],
            )
            v_uncond = self.action_out_proj(uncond_suffix_out[:, -self.action_horizon :])
            # CFG formula: amplify the conditional direction away from unconditional
            return v_uncond + guidance_scale * (v_cond - v_uncond)

        # _denoise_actions runs the conditional branch; cfg_velocity hooks in
        # to run unconditional and combine at each denoising step
        return self._denoise_actions(
            observation, cond_prefix_mask, cond_kv_cache,
            num_steps=num_steps, noise=noise,
            velocity_fn=cfg_velocity,
        )
