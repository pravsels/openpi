"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def _save_checkpoint(stats: dict, checkpoint_path: str, batch_idx: int):
    """Save running stats checkpoint so progress isn't lost on crash."""
    import pickle
    state = {
        "batch_idx": batch_idx,
        "stats": {
            key: {
                "_count": s._count,
                "_mean": s._mean,
                "_mean_of_squares": s._mean_of_squares,
                "_min": s._min,
                "_max": s._max,
                "_histograms": s._histograms,
                "_bin_edges": s._bin_edges,
            }
            for key, s in stats.items()
        },
    }
    with open(checkpoint_path, "wb") as f:
        pickle.dump(state, f)


def _load_checkpoint(checkpoint_path: str, keys: list[str]) -> tuple[dict, int] | None:
    """Load running stats from checkpoint. Returns (stats_dict, last_batch_idx) or None."""
    import pickle
    import os
    if not os.path.exists(checkpoint_path):
        return None
    try:
        with open(checkpoint_path, "rb") as f:
            state = pickle.load(f)
        stats = {}
        for key in keys:
            s = normalize.RunningStats()
            saved = state["stats"][key]
            s._count = saved["_count"]
            s._mean = saved["_mean"]
            s._mean_of_squares = saved["_mean_of_squares"]
            s._min = saved["_min"]
            s._max = saved["_max"]
            s._histograms = saved["_histograms"]
            s._bin_edges = saved["_bin_edges"]
            stats[key] = s
        print(f"Resumed from checkpoint at batch {state['batch_idx']} "
              f"({stats[keys[0]]._count} samples processed)")
        return stats, state["batch_idx"]
    except Exception as e:
        print(f"Could not load checkpoint: {e}, starting fresh")
        return None


def main(
    config_name: str,
    max_frames: int | None = None,
    checkpoint_interval: int = 5000,
):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    checkpoint_path = f"/tmp/norm_stats_checkpoint_{config_name}.pkl"

    resumed = _load_checkpoint(checkpoint_path, keys)
    if resumed is not None:
        stats, start_batch = resumed
    else:
        stats = {key: normalize.RunningStats() for key in keys}
        start_batch = 0

    for batch_idx, batch in enumerate(
        tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats",
                  initial=start_batch)
    ):
        if batch_idx < start_batch:
            continue
        for key in keys:
            stats[key].update(np.asarray(batch[key]))
        if (batch_idx + 1) % checkpoint_interval == 0:
            _save_checkpoint(stats, checkpoint_path, batch_idx + 1)
            tqdm.tqdm.write(f"  Checkpoint saved at batch {batch_idx + 1}")

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)

    # Clean up checkpoint
    import os
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Checkpoint cleaned up")


if __name__ == "__main__":
    tyro.cli(main)
