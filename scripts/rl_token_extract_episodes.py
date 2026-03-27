"""Extract per-frame RL token embeddings from a Pi0RL model for specified datasets.

Saves raw embeddings as .npz for downstream analysis (e.g. cosine similarity).
Designed to run on HPC with GPU; analysis is done locally.
"""

import dataclasses
import json
import logging
import pathlib
from collections import defaultdict
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import tyro

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.dataset_split as _dataset_split

LOGGER = logging.getLogger("openpi.rl_token_extract")


@dataclasses.dataclass(frozen=True)
class Args:
    config_name: str = "pi05_rl_token_build_block_tower"
    checkpoint_path: str = tyro.MISSING
    assets_dir: str | None = None
    id_dataset: str = "villekuosmanen/build_block_tower"
    ood_dataset: str = "villekuosmanen/eval_dAgger_drop_footbag_into_dice_tower_1.7.0"
    episodes_per_dataset: int = 1
    batch_size: int = 8
    num_denoising_steps: int = 10
    output_dir: str = tyro.MISSING
    seed: int = 42


def _init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        # Force handler creation even if root logger was already configured by
        # an import (e.g. JAX, lerobot). Without force=True, basicConfig is a
        # no-op if any handler already exists and all LOGGER output is lost.
        force=True,
    )


def _resolve_checkpoint_path(checkpoint_path: pathlib.Path) -> pathlib.Path:
    if checkpoint_path.name == "params":
        return checkpoint_path
    candidate = checkpoint_path / "params"
    if candidate.exists():
        return candidate.resolve()
    raise ValueError(f"Checkpoint path must point to Orbax params dir or its parent. Got: {checkpoint_path}")


def _resolve_assets_dir(args: Args, checkpoint_path: pathlib.Path) -> pathlib.Path:
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
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _unwrap_dataset(dataset: Any) -> Any:
    """Peel off wrapper layers to reach the dataset with hf_dataset / _datasets."""
    inner = dataset
    while hasattr(inner, "_dataset"):
        inner = inner._dataset
    return inner


def _get_episode_ids_fast(dataset: Any) -> list[str]:
    """Per-frame episode IDs via HF column access (no image I/O)."""
    inner = _unwrap_dataset(dataset)

    # Multi-dataset (ConcatDataset with _datasets list)
    wrapped = getattr(inner, "_datasets", None)
    lengths = getattr(inner, "_dataset_lengths", None)
    if wrapped is not None and lengths is not None:
        ids: list[str] = []
        for sub, length in zip(wrapped, lengths):
            hf = getattr(sub, "hf_dataset", None)
            if hf is not None and "episode_index" in hf.column_names:
                # Column access reads only the episode_index column, no images.
                col = hf["episode_index"]
                ids.extend(str(col[i]) for i in range(length))
            else:
                LOGGER.warning("Falling back to slow episode enumeration for subdataset")
                ids.extend(_dataset_split.get_episode_id(sub[i]) for i in range(length))
        return ids

    # Single dataset with hf_dataset
    hf = getattr(inner, "hf_dataset", None)
    if hf is not None and "episode_index" in hf.column_names:
        col = hf["episode_index"]
        return [str(x) for x in col]

    # Last resort: slow path (loads every item including images)
    LOGGER.warning(
        "No hf_dataset with episode_index column found — falling back to slow "
        "episode enumeration. This will load every frame including images."
    )
    return [_dataset_split.get_episode_id(dataset[i]) for i in range(len(dataset))]


def _pick_episodes(dataset: Any, n: int) -> list[str]:
    """First n unique episode IDs."""
    all_ids = _get_episode_ids_fast(dataset)
    seen: list[str] = []
    seen_set: set[str] = set()
    for eid in all_ids:
        if eid not in seen_set:
            seen.append(eid)
            seen_set.add(eid)
            if len(seen) >= n:
                break
    return seen


def _get_episode_indices(dataset: Any, episode_ids: set[str]) -> dict[str, list[int]]:
    """Map episode_id -> list of frame indices."""
    all_ids = _get_episode_ids_fast(dataset)
    result: dict[str, list[int]] = defaultdict(list)
    for i, eid in enumerate(all_ids):
        if eid in episode_ids:
            result[eid].append(i)
    return dict(result)


def _extract_rl_tokens(
    model: Any,
    transformed_dataset: Any,
    indices: list[int],
    *,
    batch_size: int,
    base_rng: jax.Array,
    num_denoising_steps: int,
) -> np.ndarray:
    """Extract RL token embeddings for given frame indices. Returns [N, emb]."""
    all_tokens: list[np.ndarray] = []
    total_batches = (len(indices) + batch_size - 1) // batch_size

    for batch_idx, start in enumerate(range(0, len(indices), batch_size)):
        batch_indices = indices[start : start + batch_size]
        # transformed_dataset[i] loads and transforms a single frame (including
        # images). This is the only place where full data I/O happens — keep
        # batch_size small to avoid OOM.
        items = [transformed_dataset[i] for i in batch_indices]

        batch = _stack_trees(items)
        # gemma.Module.embed expects JAX arrays, not numpy.
        batch = jax.tree.map(lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, batch)
        observation = _model.Observation.from_dict(batch)
        rng = jax.random.fold_in(base_rng, batch_idx)

        _, batch_rl_tokens = model.sample_actions_with_rl_token(
            rng, observation, num_steps=num_denoising_steps
        )
        all_tokens.append(np.asarray(jax.device_get(batch_rl_tokens)))
        print(f"    batch {batch_idx + 1}/{total_batches} done")

    return np.concatenate(all_tokens, axis=0).astype(np.float32)


def _create_dataset_for_repo(
    data_config: _config.DataConfig,
    repo_id: str,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
) -> tuple[Any, Any]:
    """Create raw + transformed datasets for a repo_id, reusing ID norm stats."""
    override_config = dataclasses.replace(data_config, repo_id=repo_id)
    raw = _data_loader.create_torch_dataset(override_config, action_horizon, model_config)
    transformed = _data_loader.transform_dataset(raw, data_config)
    return raw, transformed


def main(args: Args) -> None:
    _init_logging()
    print("[1/6] Resolving paths...")
    checkpoint_path = _resolve_checkpoint_path(pathlib.Path(args.checkpoint_path).resolve())
    assets_dir = _resolve_assets_dir(args, checkpoint_path)
    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  assets:     {assets_dir}")
    print(f"  output:     {output_dir}")

    print("[2/6] Loading config and building data pipeline...")
    config = _config.get_config(args.config_name)
    config = dataclasses.replace(config, assets_dir=str(assets_dir))
    data_config = config.data.create(config.assets_dirs, config.model)
    print(f"  config: {args.config_name}")

    print("[3/6] Loading model checkpoint (this allocates GPU memory)...")
    params = _model.restore_params(checkpoint_path, restore_type=np.ndarray)
    model = config.model.load(params)
    model.eval()
    print("  model loaded and set to eval mode")

    base_rng = jax.random.key(args.seed)
    results: dict[str, Any] = {}

    for ds_idx, (label, repo_id) in enumerate([("id", args.id_dataset), ("ood", args.ood_dataset)]):
        step = 4 + ds_idx
        print(f"[{step}/6] Processing {label.upper()} dataset: {repo_id}")

        print(f"  [{step}a] Creating dataset (may download from HuggingFace)...")
        raw_ds, transformed_ds = _create_dataset_for_repo(
            data_config, repo_id, config.model.action_horizon, config.model
        )
        print(f"  [{step}a] Dataset ready: {len(raw_ds)} frames")

        print(f"  [{step}b] Enumerating episodes (fast path via HF column access)...")
        episode_ids = _pick_episodes(raw_ds, args.episodes_per_dataset)
        print(f"  [{step}b] Selected episodes: {episode_ids}")

        ep_index_map = _get_episode_indices(raw_ds, set(episode_ids))

        for ep_id in episode_ids:
            indices = ep_index_map[ep_id]
            total_batches = (len(indices) + args.batch_size - 1) // args.batch_size
            print(f"  [{step}c] Extracting RL tokens: episode {ep_id}, {len(indices)} frames, {total_batches} batches")
            tokens = _extract_rl_tokens(
                model,
                transformed_ds,
                indices,
                batch_size=args.batch_size,
                base_rng=jax.random.fold_in(base_rng, hash(ep_id) % (2**31)),
                num_denoising_steps=args.num_denoising_steps,
            )
            key = f"{label}_ep{ep_id}"
            results[key] = tokens
            print(f"  [{step}c] Done: {key} -> {tokens.shape}")

    print("[6/6] Saving outputs...")
    npz_path = output_dir / "rl_token_embeddings.npz"
    np.savez_compressed(npz_path, **results)
    print(f"  embeddings: {npz_path}")

    meta = {
        "config_name": args.config_name,
        "checkpoint_path": str(checkpoint_path),
        "id_dataset": args.id_dataset,
        "ood_dataset": args.ood_dataset,
        "episodes_per_dataset": args.episodes_per_dataset,
        "num_denoising_steps": args.num_denoising_steps,
        "embedding_keys": list(results.keys()),
        "embedding_shapes": {k: list(v.shape) for k, v in results.items()},
    }
    meta_path = output_dir / "extraction_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  metadata:   {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main(tyro.cli(Args))
