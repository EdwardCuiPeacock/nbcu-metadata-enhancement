#!/bin/bash -e
function cleanup {
  echo "Removing Pipeline Output"
  rm -rf pipeline_output
}

trap cleanup EXIT

python -m test.functional.generate_test_data
pytest test/functional/ --ignore=test/functional/generate_test_data -s