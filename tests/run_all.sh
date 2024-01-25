#!/usr/bin/env bash

set -ex

for PYTHON_VERSION in 3.9 3.10 3.11; do
  for TEST in test_functionality test_env test_policy test_component test_export_record_scenario test_sensors; do
    export PYTHON_VERSION=$PYTHON_VERSION
    export TEST=$TEST
    ./tests/run.sh
  done
done