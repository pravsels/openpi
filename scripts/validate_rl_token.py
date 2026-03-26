import dataclasses
import json
import logging
import pathlib
import random
from typing import Any

import jax
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import tyro

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.dataset_split as _dataset_split


LOGGER = logging.getLogger("openpi.rlt_validate")


@dataclasses.dataclass(frozen=True)
class Args:
    config_name: str = "pi05_rl_token_build_block_tower"
    checkpoint_path: str = tyro.MISSING
    assets_dir: str | None = None
    output_dir: str | None = None
    batch_size: int = 32
    probe_batch_size: int = 256
    probe_epochs: int = 40
    hidden_dim: int = 256
    lr: float = 1e-3
    seed: int = 42
    num_denoising_steps: int = 10
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    device: str = "cuda"


def init_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _decode_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    if hasattr(value, "item"):
        try:
            return _decode_text(value.item())
        except ValueError:
            pass
    return str(value)


def _stack_trees(items: list[dict[str, Any]]) -> dict[str, Any]:
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _resolve_assets_dir(args: Args, checkpoint_path: pathlib.Path) -> pathlib.Path:
    if args.assets_dir is not None:
        return pathlib.Path(args.assets_dir).resolve()
    if (checkpoint_path / "assets").exists():
        return (checkpoint_path / "assets").resolve()
    if checkpoint_path.name == "params":
        candidate = checkpoint_path.parent / "assets"
        if candidate.exists():
            return candidate.resolve()
    raise ValueError(
        "Could not infer assets dir from checkpoint path. Pass --assets-dir explicitly "
        "(for example the sibling assets directory next to the checkpoint params)."
    )


def _resolve_output_dir(args: Args, checkpoint_path: pathlib.Path) -> pathlib.Path:
    if args.output_dir is not None:
        return pathlib.Path(args.output_dir).resolve()
    return (checkpoint_path.parent / "rlt_stage2_validation").resolve()


def _resolve_checkpoint_path(checkpoint_path: pathlib.Path) -> pathlib.Path:
    if checkpoint_path.name == "params":
        return checkpoint_path
    candidate = checkpoint_path / "params"
    if candidate.exists():
        return candidate.resolve()
    raise ValueError(
        "Checkpoint path must point to the Orbax params directory or its parent step directory. "
        f"Got: {checkpoint_path}"
    )


def _resolve_episode_split(
    dataset: Any,
    data_config: _config.DataConfig,
    assets_dir: pathlib.Path,
) -> _dataset_split.EpisodeSplit:
    if data_config.episode_split is None:
        raise ValueError("Stage 2 validation requires data_config.episode_split to be configured.")

    split_path = assets_dir / _dataset_split.EPISODE_SPLIT_FILENAME
    if split_path.exists():
        split = _dataset_split.load_episode_split(assets_dir)
        if split.seed != data_config.episode_split.seed or not np.isclose(split.val_ratio, data_config.episode_split.val_ratio):
            raise ValueError(
                "Stored episode split does not match configured split: "
                f"stored(seed={split.seed}, val_ratio={split.val_ratio}) vs "
                f"configured(seed={data_config.episode_split.seed}, val_ratio={data_config.episode_split.val_ratio})"
            )
        return split

    episode_ids = _dataset_split.get_episode_ids_from_dataset(dataset)
    split = _dataset_split.compute_episode_split(
        episode_ids,
        val_ratio=data_config.episode_split.val_ratio,
        seed=data_config.episode_split.seed,
    )
    _dataset_split.save_episode_split(assets_dir, split)
    LOGGER.info("Saved episode split to %s", split_path)
    return split


def _maybe_limit(indices: list[int], limit: int | None) -> list[int]:
    if limit is None or len(indices) <= limit:
        return indices
    return indices[:limit]


class ActionMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ClassifierMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _l2_per_example(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(pred - target, axis=-1).mean())


def _mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.square(pred - target)))


def _accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    preds = np.argmax(logits, axis=-1)
    return float(np.mean(preds == labels))


def _make_torch_loader(x: np.ndarray, y: np.ndarray, *, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _train_regressor(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    batch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]]]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_loader = _make_torch_loader(train_x, train_y, batch_size=batch_size, shuffle=True)

    history: list[dict[str, float]] = []
    best_state = None
    best_val_loss = float("inf")

    val_x_t = torch.from_numpy(val_x).float().to(device)
    val_y_t = torch.from_numpy(val_y).float().to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * batch_x.shape[0]
            count += batch_x.shape[0]

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x_t)
            val_loss = float(loss_fn(val_pred, val_y_t).item())

        train_loss = running_loss / max(1, count)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu()
    return model, history


def _train_classifier(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    batch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]]]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = _make_torch_loader(train_x, train_y.astype(np.int64), batch_size=batch_size, shuffle=True)

    history: list[dict[str, float]] = []
    best_state = None
    best_val_loss = float("inf")

    val_x_t = torch.from_numpy(val_x).float().to(device)
    val_y_t = torch.from_numpy(val_y).long().to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.long().to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * batch_x.shape[0]
            count += batch_x.shape[0]

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x_t)
            val_loss = float(loss_fn(val_logits, val_y_t).item())
            val_acc = float((val_logits.argmax(dim=-1) == val_y_t).float().mean().item())

        train_loss = running_loss / max(1, count)
        history.append(
            {"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc}
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu()
    return model, history


def _predict_regressor(model: nn.Module, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).float()
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds, axis=0)


def _predict_classifier_logits(model: nn.Module, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).float()
            logits.append(model(batch).cpu().numpy())
    return np.concatenate(logits, axis=0)


def _extract_split_features(
    model: Any,
    transformed_dataset: Any,
    raw_dataset: Any,
    indices: list[int],
    *,
    batch_size: int,
    base_rng: jax.Array,
    num_denoising_steps: int,
) -> dict[str, np.ndarray | list[str]]:
    if not indices:
        raise ValueError("Cannot extract features for an empty split.")

    rl_tokens: list[np.ndarray] = []
    states: list[np.ndarray] = []
    vla_actions: list[np.ndarray] = []
    gt_actions: list[np.ndarray] = []
    subtasks: list[str] = []

    for batch_idx, start in enumerate(range(0, len(indices), batch_size)):
        batch_indices = indices[start : start + batch_size]
        transformed_items = [transformed_dataset[i] for i in batch_indices]
        raw_items = [raw_dataset[i] for i in batch_indices]

        batch = _stack_trees(transformed_items)
        observation = _model.Observation.from_dict(batch)
        rng = jax.random.fold_in(base_rng, batch_idx)

        pred_actions, batch_rl_tokens = model.sample_actions_with_rl_token(
            rng, observation, num_steps=num_denoising_steps
        )
        batch_rl_tokens = np.asarray(jax.device_get(batch_rl_tokens))
        pred_actions = np.asarray(jax.device_get(pred_actions))

        rl_tokens.append(batch_rl_tokens)
        states.append(np.asarray(batch["state"], dtype=np.float32))
        vla_actions.append(pred_actions.reshape(pred_actions.shape[0], -1).astype(np.float32))
        gt_actions.append(np.asarray(batch["actions"], dtype=np.float32).reshape(len(batch_indices), -1))
        subtasks.extend(_decode_text(item.get("subtask", "")).strip() for item in raw_items)

    return {
        "rl_token": np.concatenate(rl_tokens, axis=0).astype(np.float32),
        "state": np.concatenate(states, axis=0).astype(np.float32),
        "vla_action": np.concatenate(vla_actions, axis=0).astype(np.float32),
        "gt_action": np.concatenate(gt_actions, axis=0).astype(np.float32),
        "subtask": subtasks,
    }


def _encode_labels(train_labels: list[str], val_labels: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    non_empty_train = [label for label in train_labels if label]
    non_empty_val = [label for label in val_labels if label]
    all_labels = sorted(set(non_empty_train) | set(non_empty_val))
    if len(all_labels) < 2:
        raise ValueError("Need at least two distinct non-empty subtask labels for phase classification.")
    label_to_id = {label: idx for idx, label in enumerate(all_labels)}
    train_ids = np.asarray([label_to_id[label] for label in train_labels], dtype=np.int64)
    val_ids = np.asarray([label_to_id[label] for label in val_labels], dtype=np.int64)
    return train_ids, val_ids, all_labels


def main(args: Args) -> None:
    init_logging()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_path = _resolve_checkpoint_path(pathlib.Path(args.checkpoint_path).resolve())
    assets_dir = _resolve_assets_dir(args, checkpoint_path)
    output_dir = _resolve_output_dir(args, checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _config.get_config(args.config_name)
    config = dataclasses.replace(config, assets_dir=str(assets_dir))
    data_config = config.data.create(config.assets_dirs, config.model)

    LOGGER.info("Loading model checkpoint from %s", checkpoint_path)
    params = _model.restore_params(checkpoint_path, restore_type=np.ndarray)
    model = config.model.load(params)
    model.eval()

    LOGGER.info("Creating raw and transformed datasets")
    raw_dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    transformed_dataset = _data_loader.transform_dataset(raw_dataset, data_config)

    split = _resolve_episode_split(raw_dataset, data_config, assets_dir)
    train_indices = _dataset_split.filter_dataset_indices_by_episode_split(raw_dataset, split, split_name="train")
    val_indices = _dataset_split.filter_dataset_indices_by_episode_split(raw_dataset, split, split_name="val")
    train_indices = _maybe_limit(train_indices, args.max_train_samples)
    val_indices = _maybe_limit(val_indices, args.max_val_samples)

    if not train_indices:
        raise ValueError("Train split resolved to zero samples after filtering/limits.")
    if not val_indices:
        raise ValueError("Val split resolved to zero samples after filtering/limits.")

    LOGGER.info("Resolved %d train samples and %d val samples", len(train_indices), len(val_indices))

    base_rng = jax.random.key(args.seed)
    train_features = _extract_split_features(
        model,
        transformed_dataset,
        raw_dataset,
        train_indices,
        batch_size=args.batch_size,
        base_rng=base_rng,
        num_denoising_steps=args.num_denoising_steps,
    )
    val_features = _extract_split_features(
        model,
        transformed_dataset,
        raw_dataset,
        val_indices,
        batch_size=args.batch_size,
        base_rng=jax.random.fold_in(base_rng, 1),
        num_denoising_steps=args.num_denoising_steps,
    )

    train_x = np.concatenate([train_features["rl_token"], train_features["state"]], axis=-1)
    val_x = np.concatenate([val_features["rl_token"], val_features["state"]], axis=-1)
    train_vla = train_features["vla_action"]
    val_vla = val_features["vla_action"]
    train_gt = train_features["gt_action"]
    val_gt = val_features["gt_action"]
    train_state = train_features["state"]
    val_state = val_features["state"]

    device = torch.device(args.device)

    LOGGER.info("Training action prediction MLP")
    action_model, action_history = _train_regressor(
        ActionMLP(train_x.shape[-1], train_vla.shape[-1], args.hidden_dim),
        train_x,
        train_vla,
        val_x,
        val_vla,
        batch_size=args.probe_batch_size,
        epochs=args.probe_epochs,
        lr=args.lr,
        device=device,
    )
    val_action_pred = _predict_regressor(action_model, val_x)

    LOGGER.info("Training linear state probe")
    linear_model, linear_history = _train_regressor(
        LinearProbe(train_features["rl_token"].shape[-1], train_state.shape[-1]),
        train_features["rl_token"],
        train_state,
        val_features["rl_token"],
        val_state,
        batch_size=args.probe_batch_size,
        epochs=args.probe_epochs,
        lr=args.lr,
        device=device,
    )
    val_state_pred = _predict_regressor(linear_model, val_features["rl_token"])

    LOGGER.info("Training random-feature linear baseline")
    rng = np.random.default_rng(args.seed)
    random_train_x = rng.standard_normal(train_features["rl_token"].shape).astype(np.float32)
    random_val_x = rng.standard_normal(val_features["rl_token"].shape).astype(np.float32)
    random_linear_model, random_linear_history = _train_regressor(
        LinearProbe(random_train_x.shape[-1], train_state.shape[-1]),
        random_train_x,
        train_state,
        random_val_x,
        val_state,
        batch_size=args.probe_batch_size,
        epochs=args.probe_epochs,
        lr=args.lr,
        device=device,
    )
    val_state_pred_random = _predict_regressor(random_linear_model, random_val_x)

    subtask_report: dict[str, Any]
    train_subtasks = train_features["subtask"]
    val_subtasks = val_features["subtask"]
    if all(label for label in train_subtasks) and all(label for label in val_subtasks):
        LOGGER.info("Training subtask classifier")
        train_subtask_ids, val_subtask_ids, label_vocab = _encode_labels(train_subtasks, val_subtasks)
        classifier, classifier_history = _train_classifier(
            ClassifierMLP(train_features["rl_token"].shape[-1], len(label_vocab), args.hidden_dim),
            train_features["rl_token"],
            train_subtask_ids,
            val_features["rl_token"],
            val_subtask_ids,
            batch_size=args.probe_batch_size,
            epochs=args.probe_epochs,
            lr=args.lr,
            device=device,
        )
        val_logits = _predict_classifier_logits(classifier, val_features["rl_token"])
        subtask_report = {
            "enabled": True,
            "label_vocab": label_vocab,
            "num_classes": len(label_vocab),
            "history": classifier_history,
            "val_accuracy": _accuracy(val_logits, val_subtask_ids),
        }
    else:
        missing_train = sum(not label for label in train_subtasks)
        missing_val = sum(not label for label in val_subtasks)
        subtask_report = {
            "enabled": False,
            "reason": (
                "Subtask labels missing from one or both splits; "
                f"missing_train={missing_train}, missing_val={missing_val}"
            ),
        }

    metrics = {
        "config_name": args.config_name,
        "checkpoint_path": str(checkpoint_path),
        "assets_dir": str(assets_dir),
        "num_train_samples": len(train_indices),
        "num_val_samples": len(val_indices),
        "action_probe": {
            "input": "concat(rl_token, state)",
            "target": "vla_action_chunk",
            "history": action_history,
            "val_mse_to_vla": _mse(val_action_pred, val_vla),
            "val_l2_to_vla": _l2_per_example(val_action_pred, val_vla),
            "val_mse_to_ground_truth": _mse(val_action_pred, val_gt),
            "val_l2_to_ground_truth": _l2_per_example(val_action_pred, val_gt),
        },
        "linear_probe": {
            "target": "state",
            "history": linear_history,
            "val_mse": _mse(val_state_pred, val_state),
            "random_baseline_history": random_linear_history,
            "random_baseline_val_mse": _mse(val_state_pred_random, val_state),
        },
        "subtask_classifier": subtask_report,
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    LOGGER.info("Wrote metrics to %s", metrics_path)

    np.savez_compressed(
        output_dir / "features_and_predictions.npz",
        train_rl_token=train_features["rl_token"],
        val_rl_token=val_features["rl_token"],
        train_state=train_state,
        val_state=val_state,
        val_action_pred=val_action_pred,
        val_vla_action=val_vla,
        val_gt_action=val_gt,
        val_state_pred=val_state_pred,
        val_state_pred_random=val_state_pred_random,
    )
    LOGGER.info("Wrote cached features to %s", output_dir / "features_and_predictions.npz")


if __name__ == "__main__":
    main(tyro.cli(Args))
