#!/usr/bin/env bash
# Usage: at the root dir >> bash format.sh
yapf --in-place --recursive -p --verbose --style .style.yapf pg_drive
