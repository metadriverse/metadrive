#!/usr/bin/env bash
# Usage: at the root dir >> bash scripts/format.sh

# Check yapf version.
ver=$(yapf --version)
if ! echo $ver | grep -q 0.30.0; then
  echo "Wrong YAPF version installed: 0.30.0 is required, not $ver. Please install via `pip install yapf==0.30.0`"
  exit 1
fi

yapf --in-place --recursive -p --verbose --style .style.yapf metadrive/

if [[ "$1" == '--test' ]]; then # Only for CI usage, user should not use --test flag.
  if ! git diff --quiet &>/dev/null; then
    echo '*** You have not formatted your code! Please run [bash format.sh] at root directory before commit! Thanks! ***'
    exit 1
  else
    echo "Code style test passed!"
  fi
fi
