#!/usr/bin/env bash

set -euo pipefail

me=`basename "$0"`
echo "Not doing anything during the `echo $me | sed 's/\([a-zA-Z]*\)\.sh/\1/'` phase"
