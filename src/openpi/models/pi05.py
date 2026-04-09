import logging
from typing import Literal, TypeAlias

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi05_config
import openpi.models.gemma_05 as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")
PrefixAttentionSchedule: TypeAlias = Literal["linear", "exp", "ones", "zeros"]


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


@jax.vmap
def left_to_right_align(x, input_mask, attn_mask):
    """Converts input from left-align to right-aligned."""
    # Due to vmap, this is operating in a single example (not batch level).
    assert x.ndim == 2
    assert input_mask.ndim == 1
    assert attn_mask.ndim == 2
    assert x.shape[0] == input_mask.shape[0]
    assert attn_mask.shape[0] == attn_mask.shape[1], attn_mask.shape
    seqlen = jnp.max(input_mask * jnp.arange(input_mask.shape[0])) + 1
    x = jnp.roll(x, -seqlen, axis=0)
    input_mask = jnp.roll(input_mask, -seqlen, axis=0)
    attn_mask = jnp.roll(attn_mask, -seqlen, axis=(0, 1))
    return x, input_mask, attn_mask


def put_along_last_axis(arr, indices, values):
    """Like np.put_along_axis(..., axis=-1), since jax is missing it."""
    assert arr.ndim == indices.ndim == values.ndim, (arr.ndim, indices.ndim, values.ndim)
    onehot = jax.nn.one_hot(indices, arr.shape[-1], dtype=values.dtype)
    put_mask = jnp.einsum("...i,...in->...n", jnp.ones(values.shape, jnp.int32), onehot)
    put_values = jnp.einsum("...i,...in->...n", values, onehot)
    return jnp.where(put_mask, put_values, arr)


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


class Pi05(_model.BaseModel):
    def __init__(self, config: pi05_config.Pi05Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # ⭐ Save loss weights for flexible training modes
        self.subtask_loss_weight = config.subtask_loss_weight
        self.fast_token_loss_weight = config.fast_token_loss_weight
        self.flow_matching_loss_weight = config.flow_matching_loss_weight
        self.stop_gradient_flow_to_prefix = config.stop_gradient_flow_to_prefix

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
        self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
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
            ### TODO: pi05 -> AR attention for subtask generation
            ar_mask += [True] * tokenized_inputs.shape[1]
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

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)

        # time MLP (for adaRMS)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)
        action_expert_tokens = action_tokens
        adarms_cond = time_emb
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
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        real_action_dim: int = 32,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        # ⭐ Support flexible training with multiple loss types
        observation = _model.preprocess_observation(
            rng, observation, train=train, image_keys=list(observation.images.keys())
        )

        total_loss = 0.0

        # ⭐ 1. Token Generation Loss (Subtask and/or FAST action tokens - computed separately)
        if self.subtask_loss_weight > 0 or self.fast_token_loss_weight > 0:
            prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

            # Compute one-hot targets: we predict *next* token, so shift the input tokens by one.
            targets = jax.nn.one_hot(
                observation.tokenized_prompt[:, 1:],
                self.PaliGemma.llm.module.vocab_size,
            )

            # Use prefix tokens to perform token generation
            prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
            (prefix_out, _), kv_cache = self.PaliGemma.llm(
                [prefix_token_embeddings, None],
                mask=prefix_attn_mask,
                positions=prefix_positions,
                adarms_cond=[None, None],
            )
            prefix_out = prefix_out[:, :-1]

            # Decode from embedding to logits
            logits = self.PaliGemma.llm(prefix_out[:, -targets.shape[1] :], method="deembed")
            logp = jax.nn.log_softmax(logits, axis=-1)
            token_pplx = jnp.sum(targets * logp, axis=-1)

            # ⭐ 1a. Subtask Token Loss (separately weighted)
            if self.subtask_loss_weight > 0:
                # Use subtask_region_mask to compute loss only on subtask tokens
                subtask_mask = getattr(observation, "subtask_region_mask", None)
                if subtask_mask is not None:
                    subtask_mask = subtask_mask[:, 1:]  # Shift for next-token prediction
                    subtask_loss = -jnp.sum(token_pplx * subtask_mask, axis=-1) / jnp.clip(jnp.sum(subtask_mask, -1), 1)
                    total_loss = total_loss + self.subtask_loss_weight * subtask_loss
                else:
                    # Fallback: use overall loss_mask if region masks not available
                    loss_mask = observation.token_loss_mask[:, 1:]
                    subtask_loss = -jnp.sum(token_pplx * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, -1), 1)
                    total_loss = total_loss + self.subtask_loss_weight * subtask_loss

            # ⭐ 1b. FAST Action Token Loss (separately weighted)
            if self.fast_token_loss_weight > 0:
                # Use action_region_mask to compute loss only on action tokens
                action_mask = getattr(observation, "action_region_mask", None)
                if action_mask is not None:
                    action_mask = action_mask[:, 1:]  # Shift for next-token prediction
                    action_loss = -jnp.sum(token_pplx * action_mask, axis=-1) / jnp.clip(jnp.sum(action_mask, -1), 1)
                    total_loss = total_loss + self.fast_token_loss_weight * action_loss
                else:
                    # Fallback: use overall loss_mask if region masks are not available.
                    loss_mask = observation.token_loss_mask[:, 1:]
                    action_loss = -jnp.sum(token_pplx * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, -1), 1)
                    total_loss = total_loss + self.fast_token_loss_weight * action_loss

        # ⭐ 2. Flow Matching Loss (MSE Loss for continuous actions)
        if self.flow_matching_loss_weight > 0:
            # During hybrid training (FAST token loss + flow matching loss active simultaneously),
            # the tokenized prompt contains ground-truth FAST action tokens appended after the
            # subtask text (e.g. "Task: ...; Subtask: pick up cup;\nAction: <fast_tokens>|").
            #
            # These GT action tokens must NOT leak into the flow matching prefix, because:
            #   1. At inference time, the flow matching branch never sees GT action tokens —
            #      it only sees the subtask text generated by autoregressive decoding.
            #   2. If the flow matching branch conditions on GT action tokens during training,
            #      it creates a train/inference distribution mismatch.
            #
            # The fix: when FAST token training is active, zero out the action token region
            # from the tokenized prompt before building the flow matching prefix. This ensures
            # the flow matching branch only conditions on images + task + subtask text, which
            # matches what it will see at inference time.
            flow_observation = observation
            action_region_mask = getattr(observation, "action_region_mask", None)
            if (
                self.fast_token_loss_weight > 0
                and action_region_mask is not None
                and observation.tokenized_prompt is not None
                and observation.tokenized_prompt_mask is not None
            ):
                # Replace action token IDs with 0 (padding) and mark them as invalid
                flow_tokenized_prompt = jnp.where(action_region_mask, 0, observation.tokenized_prompt)
                flow_tokenized_prompt_mask = jnp.where(action_region_mask, False, observation.tokenized_prompt_mask)
                flow_observation = _model.Observation(
                    images=observation.images,
                    image_masks=observation.image_masks,
                    state=observation.state,
                    tokenized_prompt=flow_tokenized_prompt,
                    tokenized_prompt_mask=flow_tokenized_prompt_mask,
                    action_tokenized_prompt=observation.action_tokenized_prompt,
                    action_tokenized_prompt_mask=observation.action_tokenized_prompt_mask,
                    token_ar_mask=observation.token_ar_mask,
                    token_loss_mask=observation.token_loss_mask,
                    subtask_region_mask=observation.subtask_region_mask,
                    action_region_mask=observation.action_region_mask,
                )

            # Always build a dedicated prefix cache for flow matching.
            prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_prefix(flow_observation)
            prefix_token_embeddings, prefix_mask, prefix_ar_mask = self._append_action_prompt_to_prefix(
                flow_observation, prefix_token_embeddings, prefix_mask, prefix_ar_mask
            )
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
            (_, _), kv_cache = self.PaliGemma.llm(
                [prefix_token_embeddings, None],
                mask=prefix_attn_mask,
                positions=prefix_positions,
                adarms_cond=[None, None],
            )
            if self.stop_gradient_flow_to_prefix:
                kv_cache = jax.tree.map(jax.lax.stop_gradient, kv_cache)

            noise_rng, time_rng = jax.random.split(rng, 2)
            batch_shape = actions.shape[:-2]
            noise = jax.random.normal(noise_rng, actions.shape)
            time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
            time_expanded = time[..., None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(flow_observation, x_t, time)
            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            attn_mask = make_attn_mask(input_mask, ar_mask)
            attn_mask = attn_mask[:, -suffix_tokens.shape[1] :, :]  # Q is [B, action_dim, ...], KV is full length
            positions = jnp.cumsum(input_mask, axis=1) - 1
            positions = positions[:, -suffix_tokens.shape[1] :]
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                kv_cache=kv_cache,
                mask=attn_mask,
                positions=positions,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            # Calculate flow loss with true actions (Real Action Dim <= Action Dim (Padding))
            flow_loss = jnp.mean(jnp.square(v_t[:, :, :real_action_dim] - u_t[:, :, :real_action_dim]), axis=-1)
            total_loss = total_loss + self.flow_matching_loss_weight * jnp.mean(flow_loss, axis=-1)

        return total_loss

    @override
    def sample_low_level_task(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        max_decoding_steps: int = 200,
        paligemma_eos_token: int = 1,
        temperature: float = 0.0,
    ) -> str:
        batch_size = observation.tokenized_prompt.shape[0]
        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

        # Right-align the real prefix tokens inside the padded window so the
        # decode loop can append new tokens immediately after the prefix.
        prefix_token_embeddings, prefix_mask, prefix_attn_mask = left_to_right_align(
            prefix_token_embeddings, prefix_mask, prefix_attn_mask
        )
        prefill_size = prefix_token_embeddings.shape[1]
        prefill_len = jnp.sum(prefix_mask, axis=-1)
        prefix_start = prefill_size - prefill_len

        # Run one prefill pass over the existing prefix to initialize the KV
        # cache, then decode the next token from the final prefix position.
        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_token_embeddings, None], mask=prefix_attn_mask, positions=prefix_positions, adarms_cond=[None, None]
        )
        last_token_embedding = prefix_out[:, -1:]
        last_logits = self.PaliGemma.llm(last_token_embedding, method="deembed")
        last_logits = jax.nn.log_softmax(last_logits, axis=-1)
        subtask_tokens = jnp.zeros((batch_size, max_decoding_steps))

        def step(carry):
            rng, last_logit, subtask_tokens, cache, _, step = carry

            # Sample token from last logit
            # Split RNG for this step
            rng, rng_step = jax.random.split(rng)
            token = jax.lax.cond(
                temperature > 0.0,
                lambda _: jax.random.categorical(rng_step, last_logit / temperature, axis=-1),
                lambda _: jnp.argmax(last_logit, axis=-1),
                operand=None,
            )
            subtask_tokens = put_along_last_axis(subtask_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token)

            # Check for early stopping --> stop if all batch elements have EOS token
            ### TODO: erase extra decoded token due to mismatch
            has_eos = jnp.any(token == paligemma_eos_token, axis=-1)
            all_eos = jnp.all(has_eos)

            # Feed the sampled token back through the model using the cached
            # prefix. The mask grows by one visible position per decode step.
            token_embedding = self.PaliGemma.llm(token, method="embed")
            positions = prefill_len[:, None] + step
            mask = jnp.logical_and(
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
                < (jnp.broadcast_to(prefill_size + step + 1, (prefix_start.shape[0], 1, 1))),
            )

            (prefix_out, _), kv_cache = self.PaliGemma.llm(
                [token_embedding, None], mask=mask, positions=positions, adarms_cond=[None, None], kv_cache=cache
            )
            last_token_embedding = prefix_out[:, -1:]
            last_logits = self.PaliGemma.llm(last_token_embedding, method="deembed")
            last_logits = jax.nn.log_softmax(last_logits, axis=-1)

            return rng, last_logits, subtask_tokens, kv_cache, all_eos, step + 1

        def cond(carry):
            _, _, _, _, all_eos, step = carry
            return (~all_eos) & (step < max_decoding_steps)
            # return step < max_decoding_steps

        # Use lax.while_loop so we can jit the full decoding loop.
        _, _, subtask_tokens, kv_cache, _, _ = jax.lax.while_loop(
            cond, step, (rng, last_logits, subtask_tokens, kv_cache, False, 0)
        )

        mask = jnp.concatenate([prefix_mask, (subtask_tokens != 0).astype(jnp.bool_)], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, jnp.ones(max_decoding_steps, dtype=jnp.bool_)], axis=0)
        # Notice:
        #  subtask_tokens [B, max_decoding_steps]
        #  kv_cache [B, prefix_len+max_decoding_steps, ...]
        #  mask [B, prefix_len+max_decoding_steps]
        #  ar_mask [prefix_len+max_decoding_steps]
        return subtask_tokens, kv_cache, mask, ar_mask

    @staticmethod
    def _combine_cfg_velocity(
        v_cond: jax.Array,
        v_uncond: jax.Array,
        guidance_scale: float | jax.Array,
    ) -> jax.Array:
        return v_uncond + guidance_scale * (v_cond - v_uncond)

    def _validate_cfg_observations(
        self,
        observation: _model.Observation,
        uncond_observation: _model.Observation,
    ) -> None:
        if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
            raise ValueError("sample_actions_cfg requires tokenized_prompt and tokenized_prompt_mask on observation.")
        if uncond_observation.tokenized_prompt is None or uncond_observation.tokenized_prompt_mask is None:
            raise ValueError(
                "sample_actions_cfg requires tokenized_prompt and tokenized_prompt_mask on uncond_observation."
            )
        if observation.action_tokenized_prompt is None or observation.action_tokenized_prompt_mask is None:
            raise ValueError(
                "sample_actions_cfg requires action_tokenized_prompt/action_tokenized_prompt_mask on observation "
                "so conditioning is appended after sample_low_level_task."
            )
        if (
            uncond_observation.action_tokenized_prompt is None
            or uncond_observation.action_tokenized_prompt_mask is None
        ):
            raise ValueError(
                "sample_actions_cfg requires action_tokenized_prompt/action_tokenized_prompt_mask on "
                "uncond_observation so conditioning is appended after sample_low_level_task."
            )

        same_prompt = bool(jnp.array_equal(observation.tokenized_prompt, uncond_observation.tokenized_prompt))
        same_prompt_mask = bool(
            jnp.array_equal(observation.tokenized_prompt_mask, uncond_observation.tokenized_prompt_mask)
        )
        if not same_prompt or not same_prompt_mask:
            raise ValueError(
                "sample_actions_cfg expects conditional and unconditional observations to share the same "
                "tokenized_prompt/tokenized_prompt_mask and differ only in action_tokenized_prompt."
            )

    def _append_action_prompt_to_prefix(
        self,
        observation: _model.Observation,
        prefix_tokens: jax.Array,
        prefix_mask: jax.Array,
        prefix_ar_mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if observation.action_tokenized_prompt is None or observation.action_tokenized_prompt_mask is None:
            return prefix_tokens, prefix_mask, prefix_ar_mask

        # Embed observation.action_tokenized_prompt (for example
        # "\nAdvantage: positive;\nAction: ") and concatenate it after the current
        # prefix, which already contains the generated subtask tokens.
        action_prompt_tokens = self.PaliGemma.llm(observation.action_tokenized_prompt, method="embed")
        action_prompt_mask = observation.action_tokenized_prompt_mask
        action_prompt_ar_mask = jnp.ones(action_prompt_tokens.shape[1], dtype=jnp.bool_)
        return (
            jnp.concatenate([prefix_tokens, action_prompt_tokens], axis=1),
            jnp.concatenate([prefix_mask, action_prompt_mask], axis=1),
            jnp.concatenate([prefix_ar_mask, action_prompt_ar_mask], axis=0),
        )

    def build_prefix_cache_with_generated_subtask(
        self,
        observation: _model.Observation,
        generated_subtask_tokens: jax.Array,
    ) -> tuple[object, jax.Array]:
        # Start from the original high-level prefix stored on the observation.
        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        generated_subtask_tokens = generated_subtask_tokens.astype(jnp.int32)
        generated_token_mask = (generated_subtask_tokens != 0).astype(jnp.bool_)
        # Embed the sampled subtask text so it becomes part of the prefix seen by
        # the action denoiser.
        generated_subtask_embeddings = self.PaliGemma.llm(generated_subtask_tokens, method="embed")

        full_prefix_tokens = jnp.concatenate([prefix_token_embeddings, generated_subtask_embeddings], axis=1)
        full_prefix_mask = jnp.concatenate([prefix_mask, generated_token_mask], axis=1)
        full_prefix_ar_mask = jnp.concatenate(
            [prefix_ar_mask, jnp.ones(generated_subtask_tokens.shape[1], dtype=jnp.bool_)], axis=0
        )
        # Only after the generated subtask has been appended do we add the
        # separate action-conditioning prefix (for example advantage + "Action:").
        full_prefix_tokens, full_prefix_mask, full_prefix_ar_mask = self._append_action_prompt_to_prefix(
            observation, full_prefix_tokens, full_prefix_mask, full_prefix_ar_mask
        )
        # Re-run the full prefix through the LLM to rebuild a KV cache whose
        # final position is right before action generation starts.
        full_prefix_attn_mask = make_attn_mask(full_prefix_mask, full_prefix_ar_mask)
        full_prefix_positions = jnp.cumsum(full_prefix_mask, axis=-1) - 1
        
        (_, _), kv_cache = self.PaliGemma.llm(
            [full_prefix_tokens, None],
            mask=full_prefix_attn_mask,
            positions=full_prefix_positions,
            adarms_cond=[None, None],
        )
        return kv_cache, full_prefix_mask

    def _sample_actions_with_prefix_cache(
        self,
        observation: _model.Observation,
        kv_cache: object,
        prefix_mask: jax.Array,
        *,
        num_steps: int | jax.Array,
        noise: jax.Array,
    ) -> _model.Actions:
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_mask.shape[1] + suffix_tokens.shape[1],
            )

            # No-op: full_attn_mask is already (b, S, P+S) so this slice is identity.
            # Kept for clarity — with a KV cache, only suffix tokens produce queries.
            query_attn_mask = full_attn_mask[:, -suffix_tokens.shape[1] :, :]

            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=query_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None

            # Last action_horizon tokens of the suffix are the action tokens; project to velocity.
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            _x_t, time = carry
            # Half-step tolerance: avoids an extra/missing step from float accumulation in time += dt.
            # e.g. with 10 steps, dt=-0.1, threshold is 0.05; time landing at 1e-16 instead of 0.0 won't trigger an 11th step.
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def _sample_actions_cfg_with_prefix_caches(
        self,
        observation: _model.Observation,
        cond_kv_cache: object,
        cond_prefix_mask: jax.Array,
        uncond_observation: _model.Observation,
        uncond_kv_cache: object,
        uncond_prefix_mask: jax.Array,
        *,
        guidance_scale: float | jax.Array,
        num_steps: int | jax.Array,
        noise: jax.Array,
    ) -> _model.Actions:
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            uncond_suffix_tokens, uncond_suffix_mask, uncond_suffix_ar_mask, uncond_adarms_cond = self.embed_suffix(
                uncond_observation, x_t, jnp.broadcast_to(time, batch_size)
            )

            cond_suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            uncond_suffix_attn_mask = make_attn_mask(uncond_suffix_mask, uncond_suffix_ar_mask)
            cond_prefix_attn_mask = einops.repeat(cond_prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            uncond_prefix_attn_mask = einops.repeat(
                uncond_prefix_mask, "b p -> b s p", s=uncond_suffix_tokens.shape[1]
            )

            cond_query_attn_mask = jnp.concatenate([cond_prefix_attn_mask, cond_suffix_attn_mask], axis=-1)[
                :, -suffix_tokens.shape[1] :, :
            ]
            uncond_query_attn_mask = jnp.concatenate([uncond_prefix_attn_mask, uncond_suffix_attn_mask], axis=-1)[
                :, -uncond_suffix_tokens.shape[1] :, :
            ]

            cond_positions = jnp.sum(cond_prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            uncond_positions = (
                jnp.sum(uncond_prefix_mask, axis=-1)[:, None] + jnp.cumsum(uncond_suffix_mask, axis=-1) - 1
            )

            (cond_prefix_out, cond_suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=cond_query_attn_mask,
                positions=cond_positions,
                kv_cache=cond_kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            (uncond_prefix_out, uncond_suffix_out), _ = self.PaliGemma.llm(
                [None, uncond_suffix_tokens],
                mask=uncond_query_attn_mask,
                positions=uncond_positions,
                kv_cache=uncond_kv_cache,
                adarms_cond=[None, uncond_adarms_cond],
            )
            assert cond_prefix_out is None
            assert uncond_prefix_out is None

            v_cond = self.action_out_proj(cond_suffix_out[:, -self.action_horizon :])
            v_uncond = self.action_out_proj(uncond_suffix_out[:, -self.action_horizon :])
            v_t = self._combine_cfg_velocity(v_cond, v_uncond, guidance_scale)

            return x_t + dt * v_t, time + dt

        def cond(carry):
            _x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def sample_actions_cfg(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        uncond_observation: _model.Observation,
        *,
        guidance_scale: float = 1.0,
        num_steps: int | jax.Array = 10,
        noise: jax.Array | None = None,
    ) -> _model.Actions:
        if guidance_scale == 1.0:
            return self.sample_actions(rng, observation, num_steps=num_steps, noise=noise)

        # CFG uses two observations with the same high-level prefix but different
        # action-conditioning suffixes (e.g. conditional advantage vs unconditional).
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )
        uncond_observation = _model.preprocess_observation(
            None, uncond_observation, train=False, image_keys=list(uncond_observation.images.keys())
        )
        self._validate_cfg_observations(observation, uncond_observation)

        batch_size = observation.state.shape[0]
        assert batch_size == 1, "Batch size must be 1 for sample_actions_cfg, subtask can be of different length"
        assert uncond_observation.state.shape[0] == batch_size, "Conditional and unconditional batches must match"
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Decode the low-level subtask once from the conditional branch. This is
        # still plain next-token decoding from the text prefix, just bounded to a
        # short span that we expect to contain the subtask. Both CFG branches
        # must then reuse these exact subtask tokens so guidance only changes
        # action denoising, not the generated subtask itself.
        generated_subtask_tokens, _cond_kv_cache, _cond_prefix_mask, _ = self.sample_low_level_task(
            rng, observation, max_decoding_steps=20, paligemma_eos_token=1, temperature=0.0
        )
        # Rebuild two prefix caches from the shared subtask tokens, then append
        # each branch's action_tokenized_prompt after the subtask boundary.
        cond_kv_cache, cond_prefix_mask = self.build_prefix_cache_with_generated_subtask(
            observation, generated_subtask_tokens
        )
        
        uncond_kv_cache, uncond_prefix_mask = self.build_prefix_cache_with_generated_subtask(
            uncond_observation, generated_subtask_tokens
        )
        # During denoising, run the action expert once per branch and combine the
        # two velocity predictions with the CFG guidance formula.
        x_0 = self._sample_actions_cfg_with_prefix_caches(
            observation,
            cond_kv_cache,
            cond_prefix_mask,
            uncond_observation,
            uncond_kv_cache,
            uncond_prefix_mask,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            noise=noise,
        )
        return (x_0, generated_subtask_tokens)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )
        batch_size = observation.state.shape[0]
        assert batch_size == 1, "Batch size must be 1 for sample_actions, subtask can be of different length"
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        generated_subtask_tokens, _kv_cache, _prefix_mask, _prefix_ar_mask = self.sample_low_level_task(
            rng, observation, max_decoding_steps=20, paligemma_eos_token=1, temperature=0.0
        )
        kv_cache, prefix_mask = self.build_prefix_cache_with_generated_subtask(
            observation, generated_subtask_tokens
        )
        x_0 = self._sample_actions_with_prefix_cache(
            observation, kv_cache, prefix_mask, num_steps=num_steps, noise=noise
        )
        return (x_0, generated_subtask_tokens)
