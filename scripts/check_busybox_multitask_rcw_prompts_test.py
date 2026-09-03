import importlib.util
import pathlib

SCRIPT = pathlib.Path("scripts/check_busybox_multitask_rcw_prompts.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("check_busybox_multitask_rcw_prompts", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_installed_rcw_is_git_main_lock():
    module = _load_module()
    commit = module.assert_rcw_git_main()
    assert commit.startswith("597aa9ad21176e7f7dcee4aede5dc1ffc07eacee")


def test_pypi_style_direct_url_is_rejected():
    module = _load_module()
    module.rcw_direct_url = lambda: {"url": "https://pypi.org/simple/robocandywrapper/", "vcs_info": {}}
    try:
        module.assert_rcw_git_main()
    except SystemExit as exc:
        assert "refusing to train on PyPI RCW" in str(exc)
    else:
        raise AssertionError("PyPI RCW must fail closed")
