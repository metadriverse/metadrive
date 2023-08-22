import os

VERSION = "0.4.1.1"


def asset_version():
    asset_path = os.path.join(os.path.dirname(__file__), "assets")
    asset_version = os.path.join(asset_path, "version.txt")
    with open(asset_version, "r") as file:
        lines = file.readlines()
    return lines[0]
