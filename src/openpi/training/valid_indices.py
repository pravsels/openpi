import dataclasses
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
    if episode_data_index is None:
        raise ValueError(f"Dataset {getattr(ds, 'repo_id', '<unknown>')} is missing episode_data_index")
    return list(episode_data_index["from"]), list(episode_data_index["to"])


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
    valid = compute_valid_indices(dataset, policy)
    output_path.write_text(",".join(str(i) for i in valid))
    logger.info("Wrote %d valid indices to %s", len(valid), output_path)
    return output_path
