# Please don't change the order of following packages!
import sys
from distutils.core import setup
from os import path

from setuptools import find_namespace_packages  # This should be place at top!


def is_mac():
    return sys.platform == "darwin"


def is_win():
    return sys.platform == "win32"


assert sys.version_info.major == 3 and sys.version_info.minor >= 6, "python version >= 3.6 is required"

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
packages = find_namespace_packages(
    exclude=("docs", "docs.*", "documentation", "documentation.*", "metadrive.assets.*", "build.*"))
print("We will install the following packages: ", packages)

""" ===== Remember to modify the PG_EDITION at first ====="""
version = "0.2.3"

install_requires = [
    "gym",
    "numpy<=1.19.3",
    "matplotlib",
    "pandas",
    "pygame",
    "tqdm",
    "yapf",
    "seaborn",
    "panda3d~=1.10.8",
    "panda3d-gltf",
    "panda3d-simplepbr",
    "pillow",
    "pytest",
    "opencv-python-headless",
]

# if (not is_mac()) and (not is_win()):
#     install_requires.append("evdev")

setup(
    name="metadrive-simulator",
    version=version,
    description="An open-ended driving simulator with infinite scenes",
    url="https://github.com/decisionforce/metadrive",
    author="MetaDrive Team",
    author_email="liquanyi@bupt.edu.cn, pengzh@ie.cuhk.edu.hk",
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
    rm -rf dist/ build/ documentation/build/ metadrive.egg-info/ docs/build/

2. Rename current version to X.Y.Z.rcA, where A is arbitrary value represent "release candidate A". 
   This is really important since pypi do not support renaming and re-uploading.

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
