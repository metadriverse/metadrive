import os
import subprocess
import sys
from unittest import mock

from metadrive import MetaDriveEnv
from metadrive.engine.asset_loader import initialize_asset_loader, AssetLoader
from metadrive.engine.core.engine_core import EngineCore


def test_asset_loader():
    default_config = MetaDriveEnv.default_config()
    world = EngineCore(global_config=default_config)
    try:
        world.clear_world()
        initialize_asset_loader(world)
        # print(AssetLoader.asset_path)
        # print(AssetLoader.file_path("aaa"))
    # # print(AssetLoader.get_loader())
    finally:
        world.close_engine()


def test_asset_loader_leaves_environment_untouched():
    """
    initialize_asset_loader must not modify os.environ (issue #864); it used to set
    PYTHONUTF8="on", an invalid value, so every Python subprocess started afterwards
    aborted during interpreter startup.
    """
    default_config = MetaDriveEnv.default_config()
    world = EngineCore(global_config=default_config)
    try:
        world.clear_world()
        with mock.patch.dict(os.environ):
            environ_before = dict(os.environ)
            initialize_asset_loader(world)
            child = subprocess.run([sys.executable, "-c", "pass"], capture_output=True, text=True)
            assert child.returncode == 0, child.stderr
            assert dict(os.environ) == environ_before
    finally:
        world.close_engine()


if __name__ == '__main__':
    test_asset_loader()
    test_asset_loader_leaves_environment_untouched()
