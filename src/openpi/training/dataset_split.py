from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import random
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import Literal


EPISODE_SPLIT_FILENAME = "episode_split.json"


@dataclasses.dataclass(frozen=True)
class EpisodeSplit:
    train_episode_ids: tuple[str, ...]
    val_episode_ids: tuple[str, ...]
    val_ratio: float
    seed: int


def compute_episode_split(episode_ids: Sequence[str], *, val_ratio: float, seed: int) -> EpisodeSplit:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    unique_episode_ids = sorted(set(episode_ids))
    if len(unique_episode_ids) < 2:
        raise ValueError("Need at least two unique episode IDs to create a train/val split.")

    num_val = max(1, math.ceil(len(unique_episode_ids) * val_ratio))
    if num_val >= len(unique_episode_ids):
        raise ValueError("Validation split would consume all episodes.")

    shuffled = unique_episode_ids.copy()
    random.Random(seed).shuffle(shuffled)

    val_episode_ids = tuple(sorted(shuffled[:num_val]))
    train_episode_ids = tuple(sorted(shuffled[num_val:]))
    return EpisodeSplit(
        train_episode_ids=train_episode_ids,
        val_episode_ids=val_episode_ids,
        val_ratio=val_ratio,
        seed=seed,
    )


def save_episode_split(assets_dir: pathlib.Path | str, split: EpisodeSplit) -> pathlib.Path:
    assets_path = pathlib.Path(assets_dir)
    assets_path.mkdir(parents=True, exist_ok=True)
    split_path = assets_path / EPISODE_SPLIT_FILENAME
    split_path.write_text(
        json.dumps(
            {
                "train_episode_ids": list(split.train_episode_ids),
                "val_episode_ids": list(split.val_episode_ids),
                "val_ratio": split.val_ratio,
                "seed": split.seed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return split_path


def load_episode_split(assets_dir: pathlib.Path | str) -> EpisodeSplit:
    payload = json.loads((pathlib.Path(assets_dir) / EPISODE_SPLIT_FILENAME).read_text())
    return EpisodeSplit(
        train_episode_ids=tuple(payload["train_episode_ids"]),
        val_episode_ids=tuple(payload["val_episode_ids"]),
        val_ratio=float(payload["val_ratio"]),
        seed=int(payload["seed"]),
    )


def get_episode_id(item: Mapping[str, Any]) -> str:
    for key in ("episode_id", "episode_index"):
        if key in item:
            value = item[key]
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            return str(value)

    recording_folderpath = item.get("recording_folderpath")
    file_path = item.get("file_path")
    if recording_folderpath is not None and file_path is not None:
        return f"{recording_folderpath}--{file_path}"

    raise KeyError("Could not determine episode ID from dataset item.")


def get_episode_ids_from_dataset(dataset: Any) -> list[str]:
    wrapped_datasets = getattr(dataset, "_datasets", None)
    dataset_lengths = getattr(dataset, "_dataset_lengths", None)
    if (
        isinstance(wrapped_datasets, Sequence)
        and isinstance(dataset_lengths, Sequence)
        and len(wrapped_datasets) == len(dataset_lengths)
    ):
        episode_ids: list[str] = []
        for subdataset, subdataset_len in zip(wrapped_datasets, dataset_lengths, strict=True):
            hf_dataset = getattr(subdataset, "hf_dataset", None)
            if hf_dataset is None:
                break
            episode_ids.extend(get_episode_id(hf_dataset[i]) for i in range(subdataset_len))
        else:
            return episode_ids

    return [get_episode_id(dataset[i]) for i in range(len(dataset))]


def filter_indices_by_episode_split(
    items: Sequence[Mapping[str, Any]],
    split: EpisodeSplit,
    *,
    split_name: Literal["train", "val"],
) -> list[int]:
    allowed_episode_ids = set(split.train_episode_ids if split_name == "train" else split.val_episode_ids)
    return [i for i, item in enumerate(items) if get_episode_id(item) in allowed_episode_ids]


def filter_dataset_indices_by_episode_split(
    dataset: Any,
    split: EpisodeSplit,
    *,
    split_name: Literal["train", "val"],
) -> list[int]:
    allowed_episode_ids = set(split.train_episode_ids if split_name == "train" else split.val_episode_ids)
    episode_ids = get_episode_ids_from_dataset(dataset)
    return [i for i, episode_id in enumerate(episode_ids) if episode_id in allowed_episode_ids]
