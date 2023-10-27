import argparse
import time
import os
import progressbar
import shutil
import urllib.request
import zipfile
import filelock
from filelock import Timeout
from metadrive.constants import VERSION
from metadrive.engine.logger import get_logger
from metadrive.version import asset_version

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


def pull_asset(update):
    logger = get_logger(propagate=False)
    TARGET_DIR = os.path.join(os.path.dirname(__file__))
    if os.path.exists(os.path.join(TARGET_DIR, "assets")):
        if not update:
            logger.warning(
                "Fail to pull. Assets already exists, version: {}. Expected version: {}. "
                "To overwrite existing assets and update, add flag '--update' and rerun this script".format(
                    asset_version(), VERSION
                )
            )
            return
        else:
            logger.info("Remove existing assets, version: {}..".format(asset_version()))
            shutil.rmtree(os.path.join(TARGET_DIR, "assets"))

    zip_path = os.path.join(TARGET_DIR, 'assets.zip')
    zip_lock = os.path.join(TARGET_DIR, 'assets.zip.lock')
    # filelock
    lock = filelock.FileLock(zip_lock, timeout=1)
    # Extract the zip file to the desired location
    try:
        with lock.acquire():
            # Fetch the zip file
            logger.info("Pull assets from {}".format(ASSET_URL))
            urllib.request.urlretrieve(ASSET_URL, zip_path, MyProgressBar())
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(TARGET_DIR)
            logger.info(
                "Successfully download assets, version: {}. MetaDrive version: {}".format(asset_version(), VERSION)
            )
            # Remove the downloaded zip file (optional)
            if os.path.exists(zip_path):
                os.remove(zip_path)
            if os.path.exists(zip_lock):
                os.remove(zip_lock)

    except Timeout:
        logger.info(
            "Another instance of this program is already running. "
            "Wait for the asset pulling finished from another program..."
        )
        while os.path.exists(zip_lock):
            logger.info("Assets not pulled yet. Waiting for 10 seconds...")
            time.sleep(10)

        logger.info("Assets are now available.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="Force overwrite the current assets")
    args = parser.parse_args()
    pull_asset(args.update)
