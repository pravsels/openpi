"""Compute valid sample indices for a config and save to a text file.

Instead of iterating over every sample (slow — requires decoding images),
this script reads episode-level metadata directly from the plugins:

  - EpisodeOutcomePlugin: outcomes.json  → filter to successful episodes
  - ControlModePlugin:    episode_modes.json → filter out autonomous frames

The result is a comma-separated list of global indices written to
``config.assets_dirs / valid_indices.txt``.
"""

import dataclasses
import json
import logging
import pathlib

import tqdm_loggable.auto as tqdm
import tyro

from robocandywrapper.factory import make_dataset_without_config
from robocandywrapper.plugins import EpisodeOutcomePlugin, ControlModePlugin

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.valid_indices as _valid_indices


def compute_valid_indices(dataset, policy: _valid_indices.ValidIndicesPolicy) -> list[int]:
    return _valid_indices.compute_valid_indices(dataset, policy)


def main(
    config_name: str, assets_base_dir: str = "./assets",
) -> None:
    config = _config.get_config(config_name)
    config = dataclasses.replace(config, assets_base_dir=assets_base_dir)
    data_config = config.data.create(config.assets_dirs, config.model)
    policy = _valid_indices.policy_from_train_config(config)

    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id to compute valid indices.")

    repo_id = data_config.repo_id
    if repo_id.endswith(".json") and pathlib.Path(repo_id).exists():
        with open(repo_id) as f:
            repo_id = json.load(f)
        logging.info("Loaded %d repo_ids from JSON file %s", len(repo_id), data_config.repo_id)
    else:
        logging.info("Loading dataset for repo_id=%s", repo_id)

    dataset = make_dataset_without_config(
        repo_id,
        plugins=[
            EpisodeOutcomePlugin(),
            ControlModePlugin(),
        ],
        load_videos=False,
    )

    n = len(dataset)
    print(f"\nComputing valid indices over {n} items from {len(dataset._datasets)} datasets...\n")
    valid = compute_valid_indices(dataset, policy)
    print(f"\n=> {len(valid)}/{n} valid indices ({100 * len(valid) / max(n, 1):.1f}%)")

    output_dir = pathlib.Path(config.assets_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _data_loader.VALID_INDICES_FILENAME
    output_path.write_text(",".join(str(i) for i in valid))
    print(f"=> Wrote to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
