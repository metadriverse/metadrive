from pathlib import Path

ASSET_LOCK = 'assets.lock'
ROOT_DIR = Path(__file__).parent

VERSION = "0.4.2.1"


def asset_version():
    lock_path = ROOT_DIR / ASSET_LOCK
    asset_path = ROOT_DIR / "assets"
    asset_version_path = asset_path / "version.txt"

    # Another instance of this program is already running. Wait for the asset pulling finished from another program...
    if lock_path.exists():
        import time
        while lock_path.exists():
            # Assets not pulled yet. Waiting for 10 seconds...
            time.sleep(10)
    # Assets are now available.

    with open(asset_version_path, "r") as file:
        lines = file.readlines()
    return lines[0]
