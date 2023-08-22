import os
import urllib.request
import zipfile
import argparse

from metadrive.version import asset_version
from metadrive.constants import VERSION
import progressbar

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


def pull_asset():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="Force overwrite the current assets")
    args = parser.parse_args()
    TARGET_DIR = os.path.join(os.path.dirname(__file__))
    if os.path.exists(os.path.join(TARGET_DIR, "assets")) and not args.update:
        print("Fail to pull. Assets already exists, version: {}. "
              "To overwrite existing assets, add flag '--update' and rerun this script".format(asset_version()))

    zip_path = os.path.join(TARGET_DIR, 'assets.zip')

    # Fetch the zip file
    print("Pull the assets from {}".format(ASSET_URL))
    urllib.request.urlretrieve(ASSET_URL, zip_path, MyProgressBar())

    # Extract the zip file to the desired location
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(TARGET_DIR)

    # Remove the downloaded zip file (optional)
    os.remove(zip_path)
    print("Successfully download assets, version: {}. MetaDrive version: {}".format(asset_version(), VERSION))


if __name__ == '__main__':
    pull_asset()
