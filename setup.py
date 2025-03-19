# Please don't change the order of following packages!
import os
import sys
from os import path

from setuptools import setup, find_namespace_packages  # This should be place at top!

ROOT_DIR = os.path.dirname(__file__)


def get_version():
    context = {}
    with open('./metadrive/version.py', 'r') as file:
        exec(file.read(), context)
    return context['VERSION']


VERSION = get_version()


def is_mac():
    return sys.platform == "darwin"


def is_win():
    return sys.platform == "win32"


this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
packages = find_namespace_packages(
    exclude=("docs", "docs.*", "documentation", "documentation.*", "build.*"))
print("We will install the following packages: ", packages)

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "python version >= 3.6 is required"
    
install_requires = [
    "requests",
    "matplotlib",
    "pygame",
    "yapf",
    "tqdm",
    "progressbar",
    "pillow",
    "lxml",
    "psutil",
    "shapely",
    "filelock",
    "Pygments",
    "mediapy"
]

test_requirement = [
    "pandas",
    "pytest",
    "scipy"
]

cuda_requirement = [
    "cuda-python==12.1.0",
    "PyOpenGL==3.1.6",
    "PyOpenGL-accelerate==3.1.6",
    "pyrr==0.10.3",
    "glfw",
]

gym_requirement = [
    "gym>=0.19.0, <=0.26.0"
]

ros_requirement = [
    "zmq"
]

setup(
    name="metadrive-simulator",
    version=VERSION,
    description="An open-ended driving simulator with infinite scenes",
    url="https://github.com/metadriverse/metadrive",
    author="MetaDrive Team",
    author_email="quanyili0057@gmail.com, pzh@cs.ucla.edu",
    packages=packages,
    install_requires=install_requires,
    extras_require={
        "cuda": cuda_requirement,
        "gym": gym_requirement,
        "ros": ros_requirement,
        ":python_version == '3.6'": ["panda3d==1.10.11", "panda3d-gltf==0.10, <1.0", "gymnasium>=0.26", "numpy", "opencv-python==4.5.5.64"],
        ":python_version == '3.7'": ["panda3d>=1.10.11", "panda3d-gltf>=0.10, <1.0", "gymnasium>=0.26", "numpy", "opencv-python"],
        ":python_version >= '3.8'": ["panda3d>=1.10.14", "panda3d-gltf>=0.13, <1.0", "gymnasium>=0.28", "numpy>=1.21.6", "opencv-python"],
    },
    include_package_data=True,
    license="Apache 2.0",
    long_description=long_description,
    long_description_content_type='text/markdown',
)

"""
How to publish to pypi and Draft github Release?  Noted by Zhenghao and Quanyi in Dec 27, 2020.

0. Checkout a new branch from main called releases/x.y.z

1. Rename VERSION in metadrive/version.py to x.y.z

2. Revise the version in metadrive/assets/version.txt to x.y.z, and compress the folder: zip -r assets.zip assets

3. Commit changes and push this branch to remote

4. Remove old files and ext_modules from setup() to get a clean wheel for all platforms in py3-none-any.wheel
    rm -rf dist/ build/ documentation/build/ metadrive_simulator.egg-info/ docs/build/

5. Get wheel
    python setup.py sdist bdist_wheel

6. Upload to production channel
    twine upload dist/*

7. Draft a release on github with new version number.
Create new tag. 
Set target to main.
Upload assets.zip and the generated .whl files to the release.

8. Publish the release

9. Merge this branch into main

!!!!!!!!!!!!! NOTE: please make sure that unzip assets.zip will generate a folder called assets instead of files

"""
