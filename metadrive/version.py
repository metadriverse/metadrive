import os

VERSION = "0.4.2.1"


def asset_version():
    asset_path = os.path.join(os.path.dirname(__file__), "assets")
    asset_version = os.path.join(asset_path, "version.txt")

    if not os.path.isfile(asset_version):
        raise ValueError(
            "Asset version file version.txt does not exist! Existing files: {}".format(os.listdir(asset_path)))

    with open(asset_version, "r") as file:
        lines = file.readlines()
    return lines[0]
