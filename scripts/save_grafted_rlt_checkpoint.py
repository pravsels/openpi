"""Save a grafted RLT checkpoint.

Grafts a trained RLT encoder/decoder onto a RETAIN'd (or any Pi0) backbone
and saves the result as a standard Orbax checkpoint that can be loaded with
`_model.restore_params` — no graft script needed by the recipient.

The output directory has the same layout as any Pi0RL training checkpoint:

    <output_dir>/
        params/          # Full Pi0RL params (backbone + rl_encoder + rl_decoder)
        assets/          # Copied from the RLT checkpoint (norm_stats, episode_split, etc.)

Quickstart:
    cd external/openpi && \\
    LD_LIBRARY_PATH=/home/praveen/miniconda3/envs/alpha-robotics/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH \\
    python scripts/save_grafted_rlt_checkpoint.py \\
        --rlt-checkpoint  ../../checkpoints/pi05-build-block-tower-rlt-6mix/checkpoints/10000 \\
        --backbone        ../../checkpoints/pi05-build-block-tower-6mix/retain/step_49999/alpha_0.5 \\
        --assets-dir      ../../checkpoints/pi05-build-block-tower-rlt-6mix/assets \\
        --config-name     pi05_rlt_build_block_tower_6mix \\
        --output-dir      ../../checkpoints/pi05-build-block-tower-rlt-6mix-retain-alpha0.5
"""

import pathlib
import re
import shutil

import flax.traverse_util
import numpy as np
import orbax.checkpoint as ocp
import tyro
import dataclasses

import openpi.models.model as _model


@dataclasses.dataclass(frozen=True)
class Args:
    rlt_checkpoint: str = tyro.MISSING
    """Path to the trained RLT checkpoint directory (contains params/ sub-dir)."""

    backbone: str = tyro.MISSING
    """Path to the backbone checkpoint to graft in (Pi0, contains params/ sub-dir)."""

    assets_dir: str = tyro.MISSING
    """Path to assets directory (norm_stats.json etc.) to copy into output."""

    config_name: str = tyro.MISSING
    """Training config name used to load this checkpoint, e.g. pi05_rlt_build_block_tower_6mix.
    Written to output_dir/CONFIG so the recipient knows which config to use."""

    output_dir: str = tyro.MISSING
    """Output directory for the grafted checkpoint."""


def _params_dir(directory: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(directory)
    if p.name == "params":
        return p.resolve()
    candidate = p / "params"
    if candidate.exists():
        return candidate.resolve()
    raise ValueError(f"No Orbax 'params' directory found at or inside: {p}")


def _graft(rlt_params, backbone_params):
    flat_rlt = flax.traverse_util.flatten_dict(rlt_params, sep="/")
    flat_bb = flax.traverse_util.flatten_dict(backbone_params, sep="/")

    rlt_only = {k for k in flat_rlt if re.match(r"^(rl_encoder|rl_decoder)/", k)}
    backbone_keys = set(flat_rlt) - rlt_only

    missing = backbone_keys - set(flat_bb)
    if missing:
        raise ValueError(f"Backbone checkpoint missing {len(missing)} keys. First few: {sorted(missing)[:5]}")

    result = {}
    for k in backbone_keys:
        v_rlt = flat_rlt[k]
        v_bb = flat_bb[k]
        if v_bb.shape != v_rlt.shape:
            raise ValueError(f"Shape mismatch for '{k}': RLT={v_rlt.shape}, backbone={v_bb.shape}")
        result[k] = v_bb.astype(v_rlt.dtype) if v_bb.dtype != v_rlt.dtype else v_bb
    for k in rlt_only:
        result[k] = flat_rlt[k]

    print(f"  {len(backbone_keys)} backbone params from backbone checkpoint")
    print(f"  {len(rlt_only)} RLT encoder/decoder params from RLT checkpoint")
    return flax.traverse_util.unflatten_dict(result, sep="/")


def main(args: Args) -> None:
    rlt_params_dir = _params_dir(args.rlt_checkpoint)
    backbone_params_dir = _params_dir(args.backbone)
    assets_dir = pathlib.Path(args.assets_dir).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()

    print(f"RLT checkpoint:  {rlt_params_dir}")
    print(f"Backbone:        {backbone_params_dir}")
    print(f"Assets:          {assets_dir}")
    print(f"Output:          {output_dir}")

    print("\nLoading params...")
    rlt_params = _model.restore_params(rlt_params_dir, restore_type=np.ndarray)
    backbone_params = _model.restore_params(backbone_params_dir, restore_type=np.ndarray)

    print("Grafting...")
    grafted = _graft(rlt_params, backbone_params)

    print("Saving params...")
    params_out = output_dir / "params"
    params_out.mkdir(parents=True, exist_ok=True)
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(params_out, {"params": grafted}, force=True)

    print("Copying assets...")
    assets_out = output_dir / "assets"
    if assets_out.exists():
        shutil.rmtree(assets_out)
    shutil.copytree(assets_dir, assets_out)

    (output_dir / "CONFIG").write_text(args.config_name + "\n")
    print(f"Config name written to {output_dir}/CONFIG")

    print(f"\nDone. Grafted checkpoint saved to: {output_dir}")
    print(f"Config: {args.config_name}")
    print("Load with:")
    print(f"  config = _config.get_config('{args.config_name}')")
    print(f"  params = _model.restore_params('{output_dir}/params')")
    print( "  model  = config.model.load(params)")

    print("Verifying saved checkpoint loads correctly...")
    loaded = _model.restore_params(params_out, restore_type=np.ndarray)
    flat_saved = flax.traverse_util.flatten_dict(grafted, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded, sep="/")

    assert set(flat_saved) == set(flat_loaded), "Key mismatch after save/load!"
    mismatched = [
        k for k in flat_saved
        if not np.allclose(flat_saved[k].astype(np.float32), flat_loaded[k].astype(np.float32), atol=1e-4)
    ]
    if mismatched:
        print(f"  WARNING: {len(mismatched)} arrays differ after save/load: {mismatched[:3]}")
    else:
        print(f"  OK — all {len(flat_saved)} arrays match after save/load.")


if __name__ == "__main__":
    main(tyro.cli(Args))
