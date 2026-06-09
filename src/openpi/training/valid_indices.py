import dataclasses
import json
import logging
import pathlib

from openpi.training import config as _config


logger = logging.getLogger("openpi")


@dataclasses.dataclass(frozen=True)
class ValidIndicesPolicy:
    mode: str = "positive_only"
    require_outcomes: bool = True


def _unwrap_dataset(dataset):
    while hasattr(dataset, "_dataset"):
        dataset = dataset._dataset
    return dataset


def indices_to_ranges(indices: list[int]) -> list[list[int]]:
    """Collapse a sorted list of ints into [start, end] inclusive ranges."""
    if not indices:
        return []
    ranges: list[list[int]] = []
    start = indices[0]
    end = start
    for idx in indices[1:]:
        if idx == end + 1:
            end = idx
        else:
            ranges.append([start, end])
            start = idx
            end = idx
    ranges.append([start, end])
    return ranges


def ranges_to_indices(ranges: list[list[int]]) -> list[int]:
    """Expand [start, end] inclusive ranges into a flat sorted list."""
    indices: list[int] = []
    for start, end in ranges:
        indices.extend(range(start, end + 1))
    return indices


def save_valid_indices(indices: list[int], path: pathlib.Path | str) -> None:
    """Save valid indices as JSON ranges."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ranges = indices_to_ranges(sorted(indices))
    path.write_text(json.dumps(ranges))
    logger.info("Wrote %d valid indices (%d ranges) to %s", len(indices), len(ranges), path)


def load_valid_indices(path: pathlib.Path | str) -> list[int]:
    """Load valid indices from a JSON ranges file."""
    path = pathlib.Path(path)
    ranges = json.loads(path.read_text())
    return ranges_to_indices(ranges)


def policy_from_train_config(config: _config.TrainConfig) -> ValidIndicesPolicy:
    mode = "positive_only"
    if getattr(config.data, "use_control_mode_advantage_prompt", False):
        mode = getattr(config.data, "advantage_prompt_mode", "positive_only")
    return ValidIndicesPolicy(mode=mode, require_outcomes=True)


def _get_plugin_instance(instances: list[object], attr_name: str) -> object | None:
    for instance in instances:
        if hasattr(instance, attr_name):
            return instance
    return None


def _episode_bounds(ds) -> tuple[list[int], list[int]]:
    episode_data_index = getattr(ds, "episode_data_index", None)
    if episode_data_index is not None:
        return list(episode_data_index["from"]), list(episode_data_index["to"])

    hf_dataset = getattr(ds, "hf_dataset", None)
    if hf_dataset is not None:
        episode_index = hf_dataset["episode_index"]
        ep_from: list[int] = []
        ep_to: list[int] = []
        current_episode = object()
        for idx, episode_idx in enumerate(episode_index):
            if episode_idx != current_episode:
                ep_from.append(idx)
                if len(ep_from) > 1:
                    ep_to.append(idx)
                current_episode = episode_idx
        if ep_from:
            ep_to.append(len(episode_index))
            return ep_from, ep_to

    raise ValueError(f"Dataset {getattr(ds, 'repo_id', '<unknown>')} is missing episode_data_index")


def _local_to_global_indices(index_map) -> dict[int, int] | None:
    if index_map is None:
        return None
    return {int(local_idx): virtual_idx for virtual_idx, local_idx in enumerate(index_map)}


def _control_mode_frame_sets(segments, episode_length: int) -> tuple[set[int], set[int]]:
    human_frames = set(range(episode_length))
    policy_frames: set[int] = set()
    if segments is None:
        return human_frames, policy_frames

    for segment in segments:
        if getattr(segment, "mode", None) != "policy":
            continue
        start = int(segment.start_index)
        end = int(segment.end_index)
        segment_frames = set(range(start, end + 1))
        policy_frames |= segment_frames
        human_frames -= segment_frames
    return human_frames, policy_frames


def compute_valid_indices(dataset, policy: ValidIndicesPolicy) -> list[int]:
    dataset = _unwrap_dataset(dataset)
    valid: list[int] = []

    for ds_idx, ds in enumerate(dataset._datasets):
        plugin_instances = dataset._plugin_instances[ds_idx]
        outcome_instance = _get_plugin_instance(plugin_instances, "outcomes")
        control_instance = _get_plugin_instance(plugin_instances, "episode_modes")
        outcomes = {} if outcome_instance is None else dict(outcome_instance.outcomes)
        episode_modes = {} if control_instance is None else dict(control_instance.episode_modes)

        ep_from, ep_to = _episode_bounds(ds)
        num_episodes = len(ep_from)
        if policy.require_outcomes:
            if not outcomes:
                raise ValueError(f"Missing outcome metadata for {getattr(ds, 'repo_id', '<unknown>')}")
            missing = [ep_idx for ep_idx in range(num_episodes) if outcomes.get(ep_idx) not in {"success", "failure"}]
            if missing:
                raise ValueError(
                    f"Missing outcome metadata for {getattr(ds, 'repo_id', '<unknown>')} episodes {missing[:5]}"
                )

        local_to_virtual = _local_to_global_indices(dataset._index_maps[ds_idx])
        global_offset = int(dataset._cumulative_lengths[ds_idx])

        for ep_idx in range(num_episodes):
            outcome = outcomes.get(ep_idx)
            if outcome not in {"success", "failure"}:
                continue

            episode_length = int(ep_to[ep_idx]) - int(ep_from[ep_idx])
            human_frames, policy_frames = _control_mode_frame_sets(episode_modes.get(ep_idx), episode_length)

            if policy.mode == "mixed":
                kept_frames = sorted((human_frames if outcome == "success" else set()) | policy_frames)
            elif policy.mode == "positive_only":
                kept_frames = sorted(human_frames) if outcome == "success" else []
            else:
                raise ValueError(f"Unsupported valid indices mode: {policy.mode}")

            for frame_in_episode in kept_frames:
                local_idx = int(ep_from[ep_idx]) + frame_in_episode
                if local_to_virtual is None:
                    global_idx = global_offset + local_idx
                else:
                    virtual_idx = local_to_virtual.get(local_idx)
                    if virtual_idx is None:
                        continue
                    global_idx = global_offset + virtual_idx
                valid.append(global_idx)

    return valid


def ensure_valid_indices_file(
    dataset,
    output_path: pathlib.Path | str,
    policy: ValidIndicesPolicy,
) -> pathlib.Path:
    output_path = pathlib.Path(output_path)
    if output_path.exists():
        logger.info("Using existing valid indices file at %s", output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        valid = compute_valid_indices(dataset, policy)
        save_valid_indices(valid, output_path)
    except ValueError:
        n = len(dataset)
        logger.info("No outcome metadata found; treating all %d indices as valid", n)
        ranges = [[0, n - 1]]
        output_path.write_text(json.dumps(ranges))
        logger.info("Wrote %d valid indices (1 range) to %s", n, output_path)

    return output_path
