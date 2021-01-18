#!/usr/bin/env bash

set -eu pipefail

# To be updated
readonly NAMESPACE=mlops
readonly TRAINING_MODULE=custom_tfx_image

readonly VERSION=$(git rev-parse --short HEAD)

readonly TRAINING_GCP_TAG="${GCR_EU_HOST}/${NAMESPACE}/${TRAINING_MODULE}"

docker build -f training/Dockerfile -t ${TRAINING_GCP_TAG}:${VERSION} -t ${TRAINING_GCP_TAG}:latest .
docker push ${TRAINING_GCP_TAG}:${VERSION}
docker push ${TRAINING_GCP_TAG}:latest
