#!/usr/bin/env bash

set -ex

DIR="$(dirname "$0")"

docker buildx build -f tests/Dockerfile -t "metadrive-base:$PYTHON_VERSION" --build-arg PYTHON_VERSION=$PYTHON_VERSION $DIR/..
docker run --rm -e TEST="$TEST" -v $(pwd):/metadrive -w /metadrive -e PYTHONPATH=/metadrive metadrive-base:$PYTHON_VERSION tests/test.sh
