import argparse
import logging
import os
import shutil
import time
import urllib.request
from pathlib import Path

import filelock
import progressbar
from filelock import Timeout

from metadrive.constants import VERSION
from metadrive.engine.logger import get_logger
from metadrive.version import asset_version

ROOT_DIR = Path(__file__).parent
ASSET_URL = "https://github.com/metadriverse/metadrive/releases/download/MetaDrive-{}/assets.zip".format(VERSION)


class MyProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def _is_asset_version_file_ready():
    asset_version_path = ROOT_DIR / 'assets' / 'version.txt'
    return asset_version_path.exists()


def wait_asset_lock():
    logger = get_logger()
    logger.info(
        "Another instance of this program is already running. "
        "Wait for the asset pulling finished from another program..."
    )
    if not _is_asset_version_file_ready():
        import time
        while not _is_asset_version_file_ready():
            logger.info("Assets not pulled yet. Waiting for 10 seconds...")
            time.sleep(10)
    logger.info("Assets are now available.")


def pull_asset(update):
    logger = get_logger()

    assets_folder = ROOT_DIR / "assets"
    zip_path = ROOT_DIR / 'assets.zip'
    lock_path = ROOT_DIR / 'assets.lock'
    temp_assets_folder = ROOT_DIR / "temp_assets"

    if _is_asset_version_file_ready() and not update:
        logger.warning(
            "Fail to update assets. Assets already exists, version: {}. Expected version: {}. "
            "To overwrite existing assets and update, add flag '--update' and rerun this script".format(
                asset_version(), VERSION
            )
        )
        return

    lock = filelock.FileLock(lock_path, timeout=1)

    # Download the file
    try:
        with lock:
            # Download assets
            logger.info("Pull assets from {} to {}".format(ASSET_URL, zip_path))
            extra_arg = [MyProgressBar()] if logger.level == logging.INFO else []
            urllib.request.urlretrieve(ASSET_URL, zip_path, *extra_arg)

            # Prepare for extraction
            if os.path.exists(assets_folder):
                logger.info("Remove existing assets. Files: {}".format(os.listdir(assets_folder)))
                shutil.rmtree(assets_folder, ignore_errors=True)
            if os.path.exists(temp_assets_folder):
                shutil.rmtree(temp_assets_folder, ignore_errors=True)

            # Extract to temporary directory
            logger.info("Extracting assets.")
            shutil.unpack_archive(filename=zip_path, extract_dir=temp_assets_folder)
            shutil.move(str(temp_assets_folder / 'assets'), str(ROOT_DIR))

    except Timeout:  # Timeout will be raised if the lock can not be acquired in 1s.
        logger.info(
            "Another instance of this program is already running. "
            "Wait for the asset pulling finished from another program..."
        )
        wait_asset_lock()
        logger.info("Assets are now available.")

    finally:
        # Cleanup
        for path in [zip_path, lock_path, temp_assets_folder]:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)

    # Final check
    if not assets_folder.exists():
        raise ValueError("Assets folder does not exist! Files: {}".format(os.listdir(ROOT_DIR)))
    if not _is_asset_version_file_ready():
        raise ValueError("Assets version misses! Files: {}".format(os.listdir(assets_folder)))

    logger.info("Successfully download assets, version: {}. MetaDrive version: {}".format(asset_version(), VERSION))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="Force overwrite the current assets")
    args = parser.parse_args()
    pull_asset(args.update)
