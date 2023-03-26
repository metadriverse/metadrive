from metadrive.engine.asset_loader import AssetLoader
from metadrive.utils.waymo_utils.utils import read_waymo_data


def test_read_waymo_data():
    for i in range(3):
        file_path = AssetLoader.file_path("waymo", "{}.pkl".format(i), return_raw_style=False)
        data = read_waymo_data(file_path)
        # print(data)
