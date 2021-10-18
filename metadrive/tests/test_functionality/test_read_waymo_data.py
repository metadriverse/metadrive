from metadrive.utils.waymo_map_utils import read_waymo_data
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.engine_utils import initialize_engine

def test_read_waymo_data():
    file_path = AssetLoader.file_path("waymo", "test.pkl", linux_style=False)
    data = read_waymo_data(file_path)
    print(data)
