import json
import logging
import pathlib

import numpy as np
import numpydantic
import pydantic


@pydantic.dataclasses.dataclass
class NormStats:
    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    q01: numpydantic.NDArray | None = None  # 1st quantile
    q99: numpydantic.NDArray | None = None  # 99th quantile


class RunningStats:
    """Compute running statistics of a batch of vectors."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # for computing quantiles on the fly

    def update(self, batch: np.ndarray) -> None:
        """
        Update the running statistics with a batch of vectors.

        Args:
            vectors (np.ndarray): An array where all dimensions except the last are batch dimensions.
        """
        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares.
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)

        self._update_histograms(batch)

    def get_statistics(self, *, use_min_max: bool = False) -> NormStats:
        """
        Compute and return the statistics of the vectors processed so far.

        Args:
            use_min_max: Store exact observed minima/maxima in the historical
                q01/q99 fields instead of the default histogram-based 1st/99th
                percentiles.

        Returns:
            dict: A dictionary containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        if use_min_max:
            q01, q99 = self._min.copy(), self._max.copy()
        else:
            q01, q99 = self._compute_quantiles([0.01, 0.99])
        return NormStats(mean=self._mean, std=stddev, q01=q01, q99=q99)

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)

            # Redistribute the existing histogram counts to the new bins
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        """Compute quantiles based on histograms."""
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results


class _NormStatsDict(pydantic.BaseModel):
    norm_stats: dict[str, NormStats]


def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    """Serialize the running statistics to a JSON string."""
    return _NormStatsDict(norm_stats=norm_stats).model_dump_json(indent=2)


def deserialize_json(data: str) -> dict[str, NormStats]:
    """Deserialize the running statistics from a JSON string."""
    return _NormStatsDict(**json.loads(data)).norm_stats


def save(directory: pathlib.Path | str, norm_stats: dict[str, NormStats]) -> None:
    """Save the normalization stats to a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats))


def load(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """Load the normalization stats from a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())


_ACTIONS_PER_TIMESTEP_FILENAME = "norm_stats_actions_per_timestep.json"


def save_actions_per_timestep(directory: pathlib.Path | str, actions_stats: NormStats) -> None:
    """Save per-timestep action normalization stats to a directory."""
    path = pathlib.Path(directory) / _ACTIONS_PER_TIMESTEP_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_NormStatsDict(norm_stats={"actions": actions_stats}).model_dump_json(indent=2))


def load_actions_per_timestep(directory: pathlib.Path | str) -> NormStats:
    """Load per-timestep action normalization stats from a directory."""
    path = pathlib.Path(directory) / _ACTIONS_PER_TIMESTEP_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Per-timestep action stats file not found at: {path}")
    return deserialize_json(path.read_text())["actions"]


# A per-timestep spread below this is treated as collapsed. Same order as
# float32's machine epsilon (~1.2e-7), so nothing narrower carries information.
_MIN_SPREAD = 1e-6


def _spread(stats: NormStats) -> np.ndarray:
    """The width of the range each normalization mode divides by."""
    if stats.q01 is not None and stats.q99 is not None:
        return np.asarray(stats.q99) - np.asarray(stats.q01)
    return np.asarray(stats.std)


def backfill_collapsed_timesteps(per_timestep: NormStats, global_stats: NormStats) -> NormStats:
    """Give a channel its global stats when its per-timestep stats collapse.

    Per-timestep stats let each step of the horizon use its own scale, which suits a
    channel whose delta grows with the prediction distance. It behaves badly for a
    *sparse* channel -- a gripper that only opens at the end of an episode. Such a
    channel barely moves in the early steps, so their percentiles collapse onto a
    single value and the resulting scale is far narrower than the channel's true
    range. Normalizing with it maps the values the channel really does take to
    enormous targets, and the loss, summed over channels, ends up tracking that one
    dimension instead of the arm.

    The channel's own global stats, pooled over the whole horizon, do see the motion,
    so falling back to them is enough to bring the targets back into range. The
    verdict is taken per channel rather than per step: once a channel collapses
    anywhere it is normalized globally throughout, because the steps either side of a
    collapse are near-collapsed too and patching only the exact zeros leaves the
    channel almost as badly scaled as before.

    Channels that never collapse keep their per-timestep stats untouched, and so does
    a channel with no spread at all even globally -- one that is genuinely constant
    has no better scale to offer, and the degenerate-scale guard in ``transforms``
    keeps it harmless.
    """
    collapsed = np.abs(_spread(per_timestep)).min(axis=0) < _MIN_SPREAD
    recoverable = collapsed & (np.abs(_spread(global_stats)) >= _MIN_SPREAD)
    if not recoverable.any():
        return per_timestep

    logging.info(
        "Per-timestep action stats collapse for channel(s) %s; using their global stats instead.",
        np.flatnonzero(recoverable).tolist(),
    )
    if (constant := collapsed & ~recoverable).any():
        logging.info(
            "Channel(s) %s have no spread even globally; leaving them to the degenerate-scale guard.",
            np.flatnonzero(constant).tolist(),
        )

    def pick(per_step: np.ndarray | None, overall: np.ndarray | None) -> np.ndarray | None:
        if per_step is None or overall is None:
            return per_step
        per_step = np.asarray(per_step)
        return np.where(recoverable, np.broadcast_to(np.asarray(overall), per_step.shape), per_step).astype(
            per_step.dtype
        )

    return NormStats(
        mean=pick(per_timestep.mean, global_stats.mean),
        std=pick(per_timestep.std, global_stats.std),
        q01=pick(per_timestep.q01, global_stats.q01),
        q99=pick(per_timestep.q99, global_stats.q99),
    )


def merge_action_norm_stats(
    norm_stats: dict[str, NormStats],
    *,
    per_timestep_action_stats: NormStats | None,
    use_per_timestep_action_norm: bool | None,
) -> dict[str, NormStats]:
    """Return normalization stats with actions overridden by per-timestep stats if enabled."""
    if not use_per_timestep_action_norm:
        return norm_stats
    if per_timestep_action_stats is None:
        logging.info("Per-timestep action normalization enabled, but stats not found. Using global stats.")
        return norm_stats
    merged = dict(norm_stats)
    merged["actions"] = per_timestep_action_stats
    return merged
