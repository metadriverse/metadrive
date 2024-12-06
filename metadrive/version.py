from pathlib import Path

VERSION = "0.4.3"


def asset_version():
    root_dir = Path(__file__).parent
    asset_path = root_dir / "assets"
    asset_version_path = asset_path / "version.txt"
    if not asset_version_path.exists():
        import os
        raise ValueError("Asset version file does not exist! Files: {}".format(os.listdir(asset_path)))
    with open(asset_version_path, "r") as file:
        lines = file.readlines()
    ret = lines[0]
    ret = ret.replace('\n', '')
    ret = ret.replace(' ', '')
    return ret
