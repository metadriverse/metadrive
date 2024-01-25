#!/usr/bin/env bash

set -e

/usr/bin/Xvfb :0 -screen 0 1280x1024x24 &

sleep 2

cd metadrive
pytest --cov=./ --cov-config=.coveragerc --cov-report=xml -sv $TEST