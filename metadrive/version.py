import os

from metadrive.engine.asset_loader import AssetLoader

VERSION = "0.4.0.1"


def assert_version():
    asset_version = os.path.join(AssetLoader.asset_path, "version.txt")
    with open(asset_version, "r") as file:
        lines = file.readlines()
    return lines[0]
