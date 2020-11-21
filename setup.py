import sys
from distutils.core import setup

from setuptools import find_packages

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, "python version >= 3.6 is required"

setup(
    name='pg-drive',
    version='0.01',
    description='None',
    url="None",
    author='Quanyi Li, Zhenghao Peng',
    author_email='liquanyi@bupt.edu.cn, pengzh@ie.cuhk.edu.hk',
    package_dir={"pg-drive": "pg-drive"},
    packages=find_packages(),
    install_requires=[
        'gym==0.17.2', 'numpy', 'matplotlib', 'pandas', "yapf==0.30.0", "panda3d==1.10.5", "ray==1.0.0",
        "ray[all]==1.0.0", "tensorflow==2.3.1", "seaborn", "tensorflow-probability==0.11.1", "panda3d-gltf",
        "panda3d-simplepbr"
    ]
)
