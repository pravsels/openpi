import os

import pynvml
import pytest

from openpi.training.lerobot_rcw_compat import ensure_lerobot_04_module_aliases

ensure_lerobot_04_module_aliases()


def set_jax_cpu_backend_if_no_gpu() -> None:
    try:
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        # No GPU found.
        os.environ["JAX_PLATFORMS"] = "cpu"


def pytest_configure(config: pytest.Config) -> None:
    set_jax_cpu_backend_if_no_gpu()
