"""Analyse RL token embeddings extracted by rl_token_extract_episodes.py.

Runs locally (no GPU needed). Computes cosine similarity matrices, within-
and cross-episode statistics, and saves heatmaps + summary JSON.
"""

import dataclasses
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro


@dataclasses.dataclass(frozen=True)
class Args:
    embeddings_path: str = tyro.MISSING
    output_dir: str | None = None


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between every pair of rows in a and b. Returns [N, M]."""
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


def _upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    """Extract upper-triangle (excluding diagonal) from a square matrix."""
    return matrix[np.triu_indices_from(matrix, k=1)]


def _plot_heatmap(
    sim_matrix: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    path: pathlib.Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim_matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_distributions(
    distributions: dict[str, np.ndarray],
    title: str,
    path: pathlib.Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, values in distributions.items():
        if len(values) > 0:
            ax.hist(values, bins=50, alpha=0.5, label=f"{label} (μ={values.mean():.3f})", density=True)
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _distribution_stats(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(np.median(values)),
    }


def main(args: Args) -> None:
    embeddings_path = pathlib.Path(args.embeddings_path).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve() if args.output_dir else embeddings_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    data = dict(np.load(embeddings_path))
    print(f"Loaded embeddings from {embeddings_path}")
    for k, v in data.items():
        print(f"  {k}: {v.shape}")

    id_keys = sorted(k for k in data if k.startswith("id_"))
    ood_keys = sorted(k for k in data if k.startswith("ood_"))

    if not id_keys or not ood_keys:
        raise ValueError(f"Expected keys starting with 'id_' and 'ood_', got: {list(data.keys())}")

    summary: dict = {"embeddings_path": str(embeddings_path), "analyses": []}

    for id_key in id_keys:
        for ood_key in ood_keys:
            id_emb = data[id_key]
            ood_emb = data[ood_key]
            pair_label = f"{id_key}_vs_{ood_key}"
            print(f"\n=== {pair_label} ===")

            # Cross-episode: ID frames vs OOD frames
            cross_sim = _cosine_sim_matrix(id_emb, ood_emb)
            _plot_heatmap(
                cross_sim,
                title=f"Cross-episode cosine sim: {id_key} vs {ood_key}",
                xlabel=f"{ood_key} frames",
                ylabel=f"{id_key} frames",
                path=output_dir / f"heatmap_{pair_label}.png",
            )

            # Within-episode baselines
            id_self_sim = _cosine_sim_matrix(id_emb, id_emb)
            ood_self_sim = _cosine_sim_matrix(ood_emb, ood_emb)

            id_self_values = _upper_triangle_values(id_self_sim)
            ood_self_values = _upper_triangle_values(ood_self_sim)
            cross_values = cross_sim.flatten()

            _plot_distributions(
                {
                    f"{id_key} self": id_self_values,
                    f"{ood_key} self": ood_self_values,
                    "cross": cross_values,
                },
                title=f"Cosine similarity distributions: {pair_label}",
                path=output_dir / f"distributions_{pair_label}.png",
            )

            # Mean-pooled episode-level similarity
            id_mean = id_emb.mean(axis=0)
            ood_mean = ood_emb.mean(axis=0)
            episode_sim = float(
                np.dot(id_mean, ood_mean)
                / (np.linalg.norm(id_mean) * np.linalg.norm(ood_mean) + 1e-12)
            )

            analysis = {
                "pair": pair_label,
                "id_frames": id_emb.shape[0],
                "ood_frames": ood_emb.shape[0],
                "embedding_dim": id_emb.shape[1],
                "episode_level_cosine_sim": episode_sim,
                "cross_frame_sim": _distribution_stats(cross_values),
                "id_self_sim": _distribution_stats(id_self_values),
                "ood_self_sim": _distribution_stats(ood_self_values),
            }
            summary["analyses"].append(analysis)

            print(f"  Episode-level cosine sim (mean-pooled): {episode_sim:.4f}")
            print(f"  Cross-frame sim:  mean={cross_values.mean():.4f}, std={cross_values.std():.4f}")
            print(f"  ID self-sim:      mean={id_self_values.mean():.4f}, std={id_self_values.std():.4f}")
            print(f"  OOD self-sim:     mean={ood_self_values.mean():.4f}, std={ood_self_values.std():.4f}")

    summary_path = output_dir / "cosine_analysis.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {summary_path}")
    print(f"Plots saved to {output_dir}/")


if __name__ == "__main__":
    main(tyro.cli(Args))
