"""Quick sanity check: load the grafted checkpoint from /shared and print key stats.

Usage:
    python scripts/verify_grafted_checkpoint.py \
        --checkpoint-dir /shared/pi05-build-block-tower-rlt-6mix-retain-alpha0.5
"""

import dataclasses
import pathlib

import flax.traverse_util
import numpy as np
import tyro

import openpi.models.model as _model
import openpi.training.config as _config


@dataclasses.dataclass(frozen=True)
class Args:
    checkpoint_dir: str = "/shared/pi05-build-block-tower-rlt-6mix-retain-alpha0.5"


def main(args: Args) -> None:
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    params_dir = checkpoint_dir / "params"
    config_file = checkpoint_dir / "CONFIG"
    assets_dir = checkpoint_dir / "assets"

    print(f"Checkpoint dir: {checkpoint_dir}")

    assert params_dir.exists(), f"params/ not found at {params_dir}"
    assert config_file.exists(), f"CONFIG not found at {config_file}"
    assert assets_dir.exists(), f"assets/ not found at {assets_dir}"

    config_name = config_file.read_text().strip()
    print(f"Config:         {config_name}")
    print(f"Assets:         {sorted(p.name for p in assets_dir.iterdir())}")

    print("\nLoading params...")
    params = _model.restore_params(params_dir, restore_type=np.ndarray)
    flat = flax.traverse_util.flatten_dict(params, sep="/")

    backbone_keys = [k for k in flat if not k.startswith(("rl_encoder/", "rl_decoder/"))]
    encoder_keys  = [k for k in flat if k.startswith("rl_encoder/")]
    decoder_keys  = [k for k in flat if k.startswith("rl_decoder/")]

    print(f"\nParam counts:")
    print(f"  backbone (PaliGemma + action_expert): {len(backbone_keys)}")
    print(f"  rl_encoder:                           {len(encoder_keys)}")
    print(f"  rl_decoder:                           {len(decoder_keys)}")
    print(f"  total:                                {len(flat)}")

    sample_key = backbone_keys[0]
    print(f"\nSample backbone param: '{sample_key}'")
    print(f"  dtype: {flat[sample_key].dtype}  shape: {flat[sample_key].shape}")

    print(f"\nLoading model with config '{config_name}'...")
    config = _config.get_config(config_name)
    model = config.model.load(params)
    model.eval()
    print("  Model loaded successfully.")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main(tyro.cli(Args))
