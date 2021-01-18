#!/usr/bin/env bash

set -eu pipefail

# Run tests for Training Pipeline
docker build -f training/Dockerfile .

# TODO: convert transformer dockerfile to multistage build and run tests
