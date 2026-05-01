import dataclasses
import os
import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config

from . import train


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)


def test_visual_drift_loss_anchors_configured_parameter_subtree():
    class TinyModel(nnx.Module):
        def __init__(self):
            self.PaliGemma = nnx.Dict(
                img=nnx.Dict(weight=nnx.Param(jnp.array([3.0, 5.0]))),
                llm=nnx.Dict(weight=nnx.Param(jnp.array([100.0]))),
            )

    config = dataclasses.replace(
        _config.get_config("debug"),
        visual_drift_regularization_weight=2.0,
    )
    model = TinyModel()
    reference = train.create_visual_drift_reference(nnx.state(model), config)

    model.PaliGemma.img.weight.value = jnp.array([4.0, 7.0])
    model.PaliGemma.llm.weight.value = jnp.array([999.0])

    loss = train.compute_visual_drift_loss(model, reference, config)

    assert jnp.isclose(loss, 2.0 * ((1.0**2 + 2.0**2) / 2.0))
