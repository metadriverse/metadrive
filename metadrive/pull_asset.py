import argparse
import os
import progressbar
import shutil
import urllib.request
import zipfile
from filelock import Filelock

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
    logger.handlers.pop()
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
            if asset_version() != VERSION:
                logger.info("Remove existing assets, version: {}..".format(asset_version()))
                shutil.rmtree(os.path.join(TARGET_DIR, "assets"))
            else:
                logger.warning(
                    "Fail to pull. Assets is already up-to-date, version: {}. MetaDrive version: {}".format(
                        asset_version(), VERSION
                    )
                )
                return

    zip_path = os.path.join(TARGET_DIR, 'assets.zip')
    zip_lock = os.path.join(TARGET_DIR, 'assets.zip.lock')

    # Fetch the zip file
    logger.info("Pull assets from {}".format(ASSET_URL))
    urllib.request.urlretrieve(ASSET_URL, zip_path, MyProgressBar())

    # Extract the zip file to the desired location
    lock = FileLock(zip_lock)
    with lock:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(TARGET_DIR)

    # Remove the downloaded zip file (optional)
    if os.path.exists(zip_path):
        os.remove(zip_path)
    logger.info("Successfully download assets, version: {}. MetaDrive version: {}".format(asset_version(), VERSION))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="Force overwrite the current assets")
    args = parser.parse_args()
    pull_asset(args.update)
