from pathlib import Path

ASSET_LOCK = 'assets.lock'
VERSION = "0.4.2.1"


def asset_version():
    root_dir = Path(__file__).parent
    lock_path = root_dir / ASSET_LOCK
    asset_path = root_dir / "assets"
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
