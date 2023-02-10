# Please don't change the order of following packages!
import sys
from os import path
import os
import shutil
from setuptools import setup, find_namespace_packages  # This should be place at top!

from os.path import join as pjoin

ROOT_DIR = os.path.dirname(__file__)


def is_mac():
    return sys.platform == "darwin"


def is_win():
    return sys.platform == "win32"


assert sys.version_info.major == 3 and sys.version_info.minor >= 6, "python version >= 3.6 is required"

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
packages = find_namespace_packages(
    exclude=("docs", "docs.*", "documentation", "documentation.*", "build.*"))
print("We will install the following packages: ", packages)

""" ===== Remember to modify the PG_EDITION at first ====="""
version = "0.2.6.0"

# Can install specific branch via:
# pip install git+https://github.com/metadriverse/metadrive.git@fix-asset-copy

# Our target destniation is:
# /Users/pengzhenghao/opt/anaconda3/envs/cs260/lib/python3.8/site-packages

# PZH: We need to copy assets to destination
# Code from: https://github.com/apache/arrow/blob/master/python/setup.py
# scm_version_write_to_prefix = os.environ.get(
#     'SETUPTOOLS_SCM_VERSION_WRITE_TO_PREFIX', ROOT_DIR)
# print("Write to: ", scm_version_write_to_prefix)
# def copy_assets(dir):
#     working_dir = pjoin(os.getcwd())
#
#     print("Root directory: ", ROOT_DIR)
#
#     print("Working directory: ", working_dir)
#     for path in os.listdir(pjoin(working_dir, "metadrive", "assets")):
#         print("The files in the assets folders: ", path)
#
#     # The files you already download:
#
#
#     for path in os.listdir(pjoin(working_dir, dir)):
#
#         print("Path: ", path)
#
#         if "python" in path:
#             metadrive_path = pjoin(working_dir, "metadrive", path)
#
#             print("MetaDrive path: ", metadrive_path)
#
#             if os.path.exists(metadrive_path):
#                 os.remove(metadrive_path)
#             metadrive_asset_path = pjoin(working_dir, dir, path)
#             print(f"Copying {metadrive_asset_path} to {metadrive_path}")
#             shutil.copy(metadrive_asset_path, metadrive_path)
#
#
# # Move libraries to python/pyarrow
# # For windows builds, move DLL from bin/
# try:
#     copy_assets("bin")
# except OSError:
#     pass
# copy_assets("lib")


install_requires = [
    "gym==0.19.0",
    "numpy",
    "matplotlib",
    "pandas",
    "pygame",
    "tqdm",
    "yapf",
    "seaborn",
    "tqdm",
    "panda3d==1.10.8",
    "panda3d-gltf",
    "panda3d-simplepbr",
    "pillow",
    "protobuf==3.20.1",
    "pytest",
    "opencv-python",
    "lxml",
    "scipy",
    "psutil"
]

# if (not is_mac()) and (not is_win()):
#     install_requires.append("evdev")

setup(
    name="metadrive-simulator",
    version=version,
    description="An open-ended driving simulator with infinite scenes",
    url="https://github.com/metadriverse/metadrive",
    author="MetaDrive Team",
    author_email="liquanyi@bupt.edu.cn, pzh@cs.ucla.edu",
    packages=packages,
    install_requires=install_requires,
    include_package_data=True,
    license="Apache 2.0",
    long_description=long_description,
    long_description_content_type='text/markdown',
)

"""
How to publish to pypi?  Noted by Zhenghao in Dec 27, 2020.

1. Remove old files and ext_modules from setup() to get a clean wheel for all platforms in py3-none-any.wheel
    rm -rf dist/ build/ documentation/build/ metadrive_simulator.egg-info/ docs/build/

2. Rename current version to X.Y.Z.rcA, where A is arbitrary value represent "release candidate A". 
   This is really important since pypi do not support renaming and re-uploading. 
   Rename version in metadrive/constants.py and setup.py 

3. Get wheel
    python setup.py sdist bdist_wheel

    WARNING: when create wheels on windows, modifying MANIFEST.in to include assets by using
    recursive-include metadrive\\assets\\ *
    recursive-include metadrive\\examples\\ *

4. Upload to test channel
    twine upload --repository testpypi dist/*

5. Test as next line. If failed, change the version name and repeat 1, 2, 3, 4, 5.
    pip install --index-url https://test.pypi.org/simple/ metadrive

6. Rename current version to X.Y.Z in setup.py, rerun 1, 3 steps.

7. Upload to production channel 
    twine upload dist/*

"""
