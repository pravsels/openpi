"""RL token reconstruction ablation.

Tests whether the RL token learned during Stage 1 RLT training carries
meaningful information by comparing three conditions:

  1. Real   — use the actual RL token produced by the encoder for each example.
  2. Zero   — replace the RL token with a zero vector (no information baseline).
  3. Shuffle — give each example its batch-neighbour's RL token (tests whether
               per-example identity matters, or if the token is near-constant).

The decoder tries to reconstruct the VLA's prefix embeddings from the RL token.
If the token is informative, real < shuffled < zero in reconstruction loss.

Also reports pairwise cosine similarity and L2 norm statistics across RL tokens
within each batch, to characterise how much the token varies across examples.
"""

import dataclasses
import json
import pathlib
from collections.abc import Mapping
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import tyro

import openpi.models.model as _model
import openpi.models.pi0 as _pi0
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.dataset_split as _dataset_split


@dataclasses.dataclass(frozen=True)
class Args:
    """CLI arguments (parsed by tyro)."""

    config_name: str = "pi05_rl_token_build_block_tower"
    checkpoint_path: str = tyro.MISSING
    assets_dir: str | None = None
    # Which episode split to draw examples from ("all" ignores the split).
    split: Literal["all", "train", "val"] = "train"
    batch_size: int = 16
    num_batches: int = 4
    # Fixed seed so the same indices are sampled across checkpoint comparisons.
    seed: int = 0
    output_path: str | None = None


def _resolve_checkpoint_path(checkpoint_path: pathlib.Path) -> pathlib.Path:
    """Accept either `.../9999/params` or `.../9999` and resolve to the params dir."""
    if checkpoint_path.name == "params":
        return checkpoint_path.resolve()
    candidate = checkpoint_path / "params"
    if candidate.exists():
        return candidate.resolve()
    raise ValueError(f"Checkpoint path must point to Orbax params dir or its parent. Got: {checkpoint_path}")


def _resolve_assets_dir(args: Args, checkpoint_path: pathlib.Path) -> pathlib.Path:
    """Find the assets dir (norm_stats, valid_indices, etc.) next to the checkpoint."""
    if args.assets_dir is not None:
        return pathlib.Path(args.assets_dir).resolve()
    if (checkpoint_path / "assets").exists():
        return (checkpoint_path / "assets").resolve()
    if checkpoint_path.name == "params":
        candidate = checkpoint_path.parent / "assets"
        if candidate.exists():
            return candidate.resolve()
    raise ValueError("Could not infer assets dir. Pass --assets-dir explicitly.")


def _stack_trees(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a list of per-example dicts into a single batched dict (like a DataLoader)."""
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _normalize_episode_id(value: Any) -> str:
    """Coerce episode IDs from various HF/numpy types to a plain string."""
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            pass
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value)


def _unwrap_dataset(dataset: Any) -> Any:
    """Peel through wrapper layers (e.g. TransformDataset) to reach the raw dataset."""
    inner = dataset
    while hasattr(inner, "_dataset"):
        inner = inner._dataset
    return inner


def _get_episode_ids_fast(dataset: Any) -> list[str]:
    """Extract the episode ID for every timestep in the dataset.

    Tries the fast path (reading the HF `episode_index` column directly)
    before falling back to per-item access.
    """
    inner = _unwrap_dataset(dataset)

    # Multi-dataset (ConcatDataset) case: iterate sub-datasets.
    wrapped = getattr(inner, "_datasets", None)
    lengths = getattr(inner, "_dataset_lengths", None)
    if wrapped is not None and lengths is not None:
        episode_ids: list[str] = []
        for subdataset, subdataset_len in zip(wrapped, lengths, strict=True):
            hf_dataset = getattr(subdataset, "hf_dataset", None)
            if hf_dataset is not None and "episode_index" in hf_dataset.column_names:
                col = hf_dataset["episode_index"]
                episode_ids.extend(_normalize_episode_id(col[i]) for i in range(subdataset_len))
            else:
                episode_ids.extend(_dataset_split.get_episode_id(subdataset[i]) for i in range(subdataset_len))
        return episode_ids

    # Single-dataset case.
    hf_dataset = getattr(inner, "hf_dataset", None)
    if hf_dataset is not None and "episode_index" in hf_dataset.column_names:
        return [_normalize_episode_id(x) for x in hf_dataset["episode_index"]]

    return [_dataset_split.get_episode_id(dataset[i]) for i in range(len(dataset))]


def _filter_indices_by_episode_split_fast(
    dataset: Any,
    split: _dataset_split.EpisodeSplit,
    *,
    split_name: Literal["train", "val"],
) -> list[int]:
    """Return dataset indices whose episode belongs to the requested split."""
    allowed_episode_ids = set(split.train_episode_ids if split_name == "train" else split.val_episode_ids)
    episode_ids = _get_episode_ids_fast(dataset)
    return [i for i, episode_id in enumerate(episode_ids) if episode_id in allowed_episode_ids]


def _resolve_episode_split(
    dataset: Any,
    data_config: _config.DataConfig,
    assets_dir: pathlib.Path,
) -> _dataset_split.EpisodeSplit:
    """Load or compute the deterministic episode-level train/val split.

    If a saved split exists in assets_dir, validate that its parameters match
    the config before using it. Otherwise compute a fresh split and save it.
    """
    if data_config.episode_split is None:
        raise ValueError("This config does not define an episode split.")

    split_path = assets_dir / _dataset_split.EPISODE_SPLIT_FILENAME
    if split_path.exists():
        split = _dataset_split.load_episode_split(assets_dir)
        if split.seed != data_config.episode_split.seed or not np.isclose(split.val_ratio, data_config.episode_split.val_ratio):
            raise ValueError(
                "Stored episode split does not match configured split: "
                f"stored(seed={split.seed}, val_ratio={split.val_ratio}) vs "
                f"configured(seed={data_config.episode_split.seed}, val_ratio={data_config.episode_split.val_ratio})"
            )
        return split

    episode_ids = _get_episode_ids_fast(dataset)
    split = _dataset_split.compute_episode_split(
        episode_ids,
        val_ratio=data_config.episode_split.val_ratio,
        seed=data_config.episode_split.seed,
    )
    _dataset_split.save_episode_split(assets_dir, split)
    return split


def _reconstruction_loss(
    predictions: jax.Array,
    target_embeddings: jax.Array,
    mask: jax.Array | None = None,
) -> jax.Array:
    """Mean per-example L2 reconstruction error (sum over embedding dim, mean over tokens).

    This mirrors the training objective in pi0_rl.py (Eq. 2 from the paper):
    squared difference summed over the embedding dimension gives per-token L2²,
    then averaged over valid tokens per example, then averaged over the batch.
    """
    recon_sq = jnp.square(predictions - target_embeddings)  # (b, M, dim)
    per_token_l2 = recon_sq.sum(axis=-1)  # (b, M) — L2² per token

    if mask is not None:
        per_token_l2 = per_token_l2 * mask
        num_valid = jnp.clip(mask.sum(axis=1), 1)
        per_example = per_token_l2.sum(axis=1) / num_valid  # (b,)
    else:
        per_example = jnp.mean(per_token_l2, axis=1)

    return jnp.mean(per_example)  # scalar


def _compute_reconstruction_ablation_metrics(
    decoder: Any,
    rl_token: jax.Array,
    target_embeddings: jax.Array,
    mask: jax.Array | None = None,
) -> dict[str, float]:
    """Run the three-condition ablation and return losses + gaps.

    The decoder takes (rl_token, target_embeddings, mask) and autoregressively
    reconstructs the target embeddings. We test:
      - real:     actual RL token → measures how well the encoder/decoder work together.
      - zero:     zero vector     → how much does the token contribute vs the decoder alone?
      - shuffled: neighbour's token (cyclic shift along batch dim) → does per-example
                  identity matter, or would any token from the same task work equally well?

    Gaps are (condition_loss - real_loss): larger means the real token is more valuable.
    """
    real_predictions = decoder(rl_token, target_embeddings, mask)
    real_loss = _reconstruction_loss(real_predictions, target_embeddings, mask)

    zero_token = jnp.zeros_like(rl_token)
    zero_predictions = decoder(zero_token, target_embeddings, mask)
    zero_loss = _reconstruction_loss(zero_predictions, target_embeddings, mask)

    # Cyclic shift: example i gets example (i-1)'s RL token.
    shuffled_token = jnp.roll(rl_token, shift=1, axis=0)
    shuffled_predictions = decoder(shuffled_token, target_embeddings, mask)
    shuffled_loss = _reconstruction_loss(shuffled_predictions, target_embeddings, mask)

    return {
        "real_recon_loss": float(real_loss),
        "zero_recon_loss": float(zero_loss),
        "shuffled_recon_loss": float(shuffled_loss),
        "zero_recon_gap": float(zero_loss - real_loss),
        "shuffled_recon_gap": float(shuffled_loss - real_loss),
    }


def _pairwise_cosine_stats(tokens: np.ndarray) -> dict[str, float]:
    """Cosine similarity between all pairs of RL tokens in the batch.

    High similarity (→1.0) means tokens are nearly identical across examples.
    For a single-task dataset this is expected; across tasks it would signal collapse.
    Uses upper-triangle of the similarity matrix to avoid double-counting pairs.
    """
    if len(tokens) < 2:
        return {"rl_token_pairwise_cosine_mean": 0.0, "rl_token_pairwise_cosine_std": 0.0}
    normed = tokens / (np.linalg.norm(tokens, axis=-1, keepdims=True) + 1e-12)
    sims = normed @ normed.T
    values = sims[np.triu_indices_from(sims, k=1)]
    return {
        "rl_token_pairwise_cosine_mean": float(values.mean()),
        "rl_token_pairwise_cosine_std": float(values.std()),
    }


def _compute_batch_metrics(
    model: Any,
    transformed_dataset: Any,
    batch_indices: list[int],
) -> dict[str, float]:
    """End-to-end forward pass for one batch: VLA prefix → RL encoder → decoder ablation.

    Steps:
      1. Fetch and collate individual examples from the dataset.
      2. Run the VLA backbone (PaliGemma) on image+language to get prefix embeddings.
         These are stop-gradiented — we never backprop into the VLA here.
      3. Pass prefix embeddings through the RL encoder to produce the RL token (b, dim).
      4. Run the three-condition ablation through the RL decoder.
      5. Compute token statistics (L2 norm, pairwise cosine similarity).
    """
    items = [transformed_dataset[i] for i in batch_indices]
    batch = _stack_trees(items)
    batch = jax.tree.map(lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, batch)
    observation = _model.Observation.from_dict(batch)
    observation = _model.preprocess_observation(None, observation, train=False)

    # VLA prefix forward pass (image tokens + language tokens → embeddings).
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1

    outputs, _ = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    prefix_out = jax.lax.stop_gradient(outputs[0])  # (b, M, dim) — frozen VLA embeddings

    # RL encoder: cross-attention with a learned query over the prefix embeddings → (b, dim).
    rl_token = model.rl_encoder(prefix_out, mask=prefix_mask)

    # Run the three-condition ablation through the RL decoder.
    metrics = _compute_reconstruction_ablation_metrics(model.rl_decoder, rl_token, prefix_out, prefix_mask)

    # Token statistics for diagnosing similarity/collapse.
    rl_token_np = np.asarray(jax.device_get(rl_token), dtype=np.float32)
    metrics.update(
        {
            "rl_token_norm_mean": float(np.linalg.norm(rl_token_np, axis=-1).mean()),
            "rl_token_norm_std": float(np.linalg.norm(rl_token_np, axis=-1).std()),
        }
    )
    metrics.update(_pairwise_cosine_stats(rl_token_np))
    return metrics


def aggregate_batch_metrics(batch_metrics: list[Mapping[str, float]]) -> dict[str, float]:
    """Average numeric metrics across batches (skipping per-batch metadata like indices)."""
    if not batch_metrics:
        raise ValueError("Need at least one batch of metrics to aggregate.")

    def _is_aggregate_metric(key: str, value: Any) -> bool:
        if key in {"batch_idx", "batch_size", "indices"}:
            return False
        return isinstance(value, (int, float, np.integer, np.floating))

    keys = sorted(key for key, value in batch_metrics[0].items() if _is_aggregate_metric(key, value))
    return {key: float(np.mean([metrics[key] for metrics in batch_metrics])) for key in keys}


def _sample_indices(
    dataset: Any,
    data_config: _config.DataConfig,
    assets_dir: pathlib.Path,
    *,
    split_name: Literal["all", "train", "val"],
    batch_size: int,
    num_batches: int,
    seed: int,
) -> list[list[int]]:
    """Deterministically sample timestep indices, grouped into batches.

    Uses a fixed seed so that running the same ablation on different checkpoints
    evaluates exactly the same examples, making the comparison fair.
    """
    if split_name == "all":
        candidate_indices = list(range(len(dataset)))
    else:
        split = _resolve_episode_split(dataset, data_config, assets_dir)
        candidate_indices = _filter_indices_by_episode_split_fast(dataset, split, split_name=split_name)

    if not candidate_indices:
        raise ValueError(f"No indices found for split={split_name}.")

    sample_count = min(len(candidate_indices), batch_size * num_batches)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(np.asarray(candidate_indices), size=sample_count, replace=False).tolist()
    return [sampled[i : i + batch_size] for i in range(0, len(sampled), batch_size)]


def main(args: Args) -> None:
    checkpoint_path = _resolve_checkpoint_path(pathlib.Path(args.checkpoint_path))
    assets_dir = _resolve_assets_dir(args, checkpoint_path)

    # --- Step 1: Load the frozen Pi0RL model from a Stage 1 checkpoint. ---
    print("[1/4] Loading config and model...")
    config = _config.get_config(args.config_name)
    config = dataclasses.replace(config, assets_dir=str(assets_dir))
    data_config = config.data.create(config.assets_dirs, config.model)
    params = _model.restore_params(checkpoint_path, restore_type=np.ndarray)
    model = config.model.load(params)
    model.eval()

    # --- Step 2: Build the dataset (downloads from HF if not cached). ---
    print("[2/4] Creating dataset...")
    raw_dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    transformed_dataset = _data_loader.transform_dataset(raw_dataset, data_config)

    # --- Step 3: Sample deterministic batches of timestep indices. ---
    print("[3/4] Sampling batches...")
    sampled_batches = _sample_indices(
        raw_dataset,
        data_config,
        assets_dir,
        split_name=args.split,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        seed=args.seed,
    )
    print(f"  split: {args.split}")
    print(f"  sampled_batches: {len(sampled_batches)}")
    print(f"  sampled_examples: {sum(len(batch) for batch in sampled_batches)}")

    # --- Step 4: For each batch, run the full forward pass and ablation. ---
    print("[4/4] Running ablation...")
    per_batch = []
    for batch_idx, batch_indices in enumerate(sampled_batches, start=1):
        metrics = _compute_batch_metrics(model, transformed_dataset, batch_indices)
        per_batch.append({"batch_idx": batch_idx, "batch_size": len(batch_indices), "indices": batch_indices, **metrics})
        print(
            f"  batch {batch_idx}/{len(sampled_batches)} "
            f"real={metrics['real_recon_loss']:.3f} "
            f"zero_gap={metrics['zero_recon_gap']:.3f} "
            f"shuffle_gap={metrics['shuffled_recon_gap']:.3f}"
        )

    # --- Output: aggregate metrics + per-batch detail as JSON. ---
    summary = {
        "config_name": args.config_name,
        "checkpoint_path": str(checkpoint_path),
        "assets_dir": str(assets_dir),
        "split": args.split,
        "batch_size": args.batch_size,
        "num_batches": len(sampled_batches),
        "aggregate": aggregate_batch_metrics([metrics for metrics in per_batch]),
        "per_batch": per_batch,
    }

    payload = json.dumps(summary, indent=2)
    print(payload)
    if args.output_path is not None:
        output_path = pathlib.Path(args.output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n")
        print(f"Saved ablation report to {output_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
