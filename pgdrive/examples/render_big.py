import sys
from pgdrive.utils import setup_logger
from pgdrive.tests.block_test.test_big import vis_big
from pgdrive.utils.asset_loader import AssetLoader
from pgdrive.world.pg_world import root_path

if __name__ == "__main__":
    setup_logger(True)
    path = AssetLoader.windows_style2unix_style(root_path) if sys.platform == "win32" else root_path
    vis_big(asset_path=path)
