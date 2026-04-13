"""RLT-on-retain ablation.

Tests whether the RLT encoder trained on the BASE policy (pi05-build-block-tower-6mix
step 49999) still produces meaningful RL tokens when the backbone weights have been
replaced by a RETAIN'd checkpoint.

Procedure:
  1. Load the trained RLT checkpoint (Pi0RL params: PaliGemma + action_expert +
     rl_encoder + rl_decoder).
  2. Load the retain'd backbone (Pi0 params: PaliGemma + action_expert only).
  3. Graft: replace PaliGemma and action_expert keys from the retain'd backbone,
     keep rl_encoder and rl_decoder from the trained RLT checkpoint.
  4. Run the reconstruction ablation (real / zero / shuffled token) in memory.

Usage:
    cd external/openpi
    python scripts/rlt_retain_ablation.py \\
        --rlt-checkpoint   /path/to/pi05-build-block-tower-rlt-6mix/checkpoints/10000 \\
        --retain-backbone  /path/to/checkpoints/pi05-build-block-tower-6mix/retain/step_49999/alpha_0.5 \\
        --assets-dir       /path/to/pi05-build-block-tower-rlt-6mix/assets \\
        [--config-name     pi05_rlt_build_block_tower_6mix] \\
        [--batch-size 16]  [--num-batches 4] \\
        [--output-path     results/rlt_retain_ablation.json]
"""

import dataclasses
import json
import pathlib
import re
from typing import Any

import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import tyro

import openpi.models.model as _model
import openpi.models.pi0 as _pi0
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

# Reuse helpers from the ablation script.
from scripts.rl_token_recon_ablation import (
    _compute_reconstruction_ablation_metrics,
    _pairwise_cosine_stats,
    _reconstruction_loss,  # noqa: F401 — imported so the module is usable stand-alone
    _resolve_episode_split,
    _sample_indices,
    _stack_trees,
    aggregate_batch_metrics,
)


@dataclasses.dataclass(frozen=True)
class Args:
    """CLI arguments."""

    rlt_checkpoint: str = tyro.MISSING
    """Path to the trained RLT checkpoint directory (the step dir, e.g. .../10000).
    Must contain a 'params/' sub-directory (Orbax format)."""

    retain_backbone: str = tyro.MISSING
    """Path to the retain'd Pi0 checkpoint directory (e.g. .../alpha_0.5).
    Must contain a 'params/' sub-directory."""

    assets_dir: str = tyro.MISSING
    """Path to the assets directory (norm_stats.json, etc.) from the RLT repo."""

    config_name: str = "pi05_rlt_build_block_tower_6mix"
    """Training config name for dataset / model architecture."""

    split: str = "train"
    """Which episode split to draw from: 'all', 'train', or 'val'."""

    batch_size: int = 16
    num_batches: int = 4
    seed: int = 0
    output_path: str | None = None


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _params_dir(directory: str | pathlib.Path) -> pathlib.Path:
    """Accept either '.../step/params' or '.../step' and resolve to params dir."""
    p = pathlib.Path(directory)
    if p.name == "params":
        return p.resolve()
    candidate = p / "params"
    if candidate.exists():
        return candidate.resolve()
    raise ValueError(
        f"Could not find an Orbax 'params' directory at or inside: {p}"
    )


def _graft_retain_backbone(
    rlt_params: dict[str, Any],
    retain_params: dict[str, Any],
) -> dict[str, Any]:
    """Replace backbone keys (PaliGemma, action_expert) in rlt_params with retain_params.

    The RLT encoder/decoder keys (rl_encoder/*, rl_decoder/*) are kept from rlt_params.
    """
    flat_rlt = flax.traverse_util.flatten_dict(rlt_params, sep="/")
    flat_retain = flax.traverse_util.flatten_dict(retain_params, sep="/")

    rlt_only = {k for k in flat_rlt if re.match(r"^(rl_encoder|rl_decoder)/", k)}
    backbone_keys_from_rlt = {k for k in flat_rlt if k not in rlt_only}

    backbone_keys_in_retain = set(flat_retain.keys())
    missing = backbone_keys_from_rlt - backbone_keys_in_retain
    if missing:
        raise ValueError(
            f"Retain checkpoint is missing {len(missing)} backbone keys that the "
            f"RLT checkpoint expects. First few: {sorted(missing)[:5]}"
        )

    result = {}
    # Take backbone from retain.
    for k in backbone_keys_from_rlt:
        v_rlt = flat_rlt[k]
        v_retain = flat_retain[k]
        if v_retain.shape != v_rlt.shape:
            raise ValueError(
                f"Shape mismatch for '{k}': RLT={v_rlt.shape}, retain={v_retain.shape}"
            )
        result[k] = v_retain.astype(v_rlt.dtype) if v_retain.dtype != v_rlt.dtype else v_retain
    # Keep encoder/decoder from RLT.
    for k in rlt_only:
        result[k] = flat_rlt[k]

    n_backbone = len(backbone_keys_from_rlt)
    n_rlt = len(rlt_only)
    print(f"  Grafted {n_backbone} backbone params from retain, kept {n_rlt} RLT encoder/decoder params.")
    return flax.traverse_util.unflatten_dict(result, sep="/")


# ---------------------------------------------------------------------------
# Batch metrics (mirrors rl_token_recon_ablation._compute_batch_metrics)
# ---------------------------------------------------------------------------

def _compute_batch_metrics(
    model: Any,
    transformed_dataset: Any,
    batch_indices: list[int],
) -> dict[str, float]:
    items = [transformed_dataset[i] for i in batch_indices]
    batch = _stack_trees(items)
    batch = jax.tree.map(
        lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number) else x,
        batch,
    )
    observation = _model.Observation.from_dict(batch)
    observation = _model.preprocess_observation(None, observation, train=False)

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1

    outputs, _ = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    prefix_out = jax.lax.stop_gradient(outputs[0])

    rl_token = model.rl_encoder(prefix_out, mask=prefix_mask)
    metrics = _compute_reconstruction_ablation_metrics(model.rl_decoder, rl_token, prefix_out, prefix_mask)

    rl_token_np = np.asarray(jax.device_get(rl_token), dtype=np.float32)
    metrics.update({
        "rl_token_norm_mean": float(np.linalg.norm(rl_token_np, axis=-1).mean()),
        "rl_token_norm_std": float(np.linalg.norm(rl_token_np, axis=-1).std()),
    })
    metrics.update(_pairwise_cosine_stats(rl_token_np))
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: Args) -> None:
    rlt_params_dir = _params_dir(args.rlt_checkpoint)
    retain_params_dir = _params_dir(args.retain_backbone)
    assets_dir = pathlib.Path(args.assets_dir).resolve()

    print(f"RLT checkpoint:   {rlt_params_dir}")
    print(f"Retain backbone:  {retain_params_dir}")
    print(f"Assets dir:       {assets_dir}")

    # --- Load config ---
    print("\n[1/4] Loading config...")
    config = _config.get_config(args.config_name)
    config = dataclasses.replace(config, assets_dir=str(assets_dir))
    data_config = config.data.create(config.assets_dirs, config.model)

    # --- Load and graft params ---
    print("[2/4] Loading and grafting params...")
    rlt_params = _model.restore_params(rlt_params_dir, restore_type=np.ndarray)
    retain_params = _model.restore_params(retain_params_dir, restore_type=np.ndarray)
    grafted_params = _graft_retain_backbone(rlt_params, retain_params)

    model = config.model.load(grafted_params)
    model.eval()

    # --- Build dataset ---
    print("[3/4] Creating dataset...")
    raw_dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    transformed_dataset = _data_loader.transform_dataset(raw_dataset, data_config)

    # --- Sample batches ---
    print("[4/4] Running ablation...")
    sampled_batches = _sample_indices(
        raw_dataset,
        data_config,
        assets_dir,
        split_name=args.split,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        seed=args.seed,
    )
    print(f"  split={args.split}, batches={len(sampled_batches)}, "
          f"examples={sum(len(b) for b in sampled_batches)}")

    per_batch = []
    for batch_idx, batch_indices in enumerate(sampled_batches, start=1):
        metrics = _compute_batch_metrics(model, transformed_dataset, batch_indices)
        per_batch.append({
            "batch_idx": batch_idx,
            "batch_size": len(batch_indices),
            "indices": batch_indices,
            **metrics,
        })
        print(
            f"  batch {batch_idx}/{len(sampled_batches)} "
            f"real={metrics['real_recon_loss']:.3f} "
            f"zero_gap={metrics['zero_recon_gap']:.3f} "
            f"shuffle_gap={metrics['shuffled_recon_gap']:.3f} "
            f"cosine={metrics['rl_token_pairwise_cosine_mean']:.3f}"
        )

    summary = {
        "config_name": args.config_name,
        "rlt_checkpoint": str(rlt_params_dir),
        "retain_backbone": str(retain_params_dir),
        "assets_dir": str(assets_dir),
        "split": args.split,
        "batch_size": args.batch_size,
        "num_batches": len(sampled_batches),
        "aggregate": aggregate_batch_metrics([m for m in per_batch]),
        "per_batch": per_batch,
    }

    payload = json.dumps(summary, indent=2)
    print("\n=== Aggregate results ===")
    print(json.dumps(summary["aggregate"], indent=2))

    if args.output_path is not None:
        out = pathlib.Path(args.output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n")
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
