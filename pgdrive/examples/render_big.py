from pgdrive.tests.block_test.test_big import vis_big
from pgdrive.world.pg_world import root_path
import sys
from pgdrive.utils.asset_loader import AssetLoader

if __name__ == "__main__":
    path = AssetLoader.windows_style2unix_style(root_path) if sys.platform == "win32" else root_path
    vis_big(asset_path=path)
