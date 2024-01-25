#!/usr/bin/env bash

set -e

pip install /metadrive
python -m metadrive.pull_asset

pip install mediapy nbmake pytest pytest-cov pytest-xdist