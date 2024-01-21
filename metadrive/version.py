from pathlib import Path

VERSION = "0.4.2.1"


def asset_version():
    root_dir = Path(__file__).parent
    asset_path = root_dir / "assets"
    asset_version_path = asset_path / "version.txt"
    assert asset_version_path.exists(), "Asset version file does not exist!"
    with open(asset_version_path, "r") as file:
        lines = file.readlines()
    return lines[0]
