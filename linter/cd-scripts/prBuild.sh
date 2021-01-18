set -eu pipefail

# Run tests for Training Pipeline
docker build -f linter/precommit.Dockerfile .
