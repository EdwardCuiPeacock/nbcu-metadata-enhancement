SHELL = /bin/bash

help: ## Display this help screen
	@grep -h -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

push-custom-tfx-image: ## Builds custom TFX image and pushes to registry
	@docker build -t eu.gcr.io/disco-mgmt-100/mlops/custom_tfx_image:local -f training/CustomTFX.Dockerfile . & \
	docker push eu.gcr.io/disco-mgmt-100/mlops/custom_tfx_image:local

push-image-to-sandbox: ## Builds custom TFX image and pushes to sandbox for manual testing
	@docker build -t gcr.io/res-nbcupea-dev-ds-sandbox-001/temp_metadata_pipeline:local -f training/Dockerfile . & \
	docker push gcr.io/res-nbcupea-dev-ds-sandbox-001/temp_metadata_pipeline:local

push-transformer-image: ## Builds KFServing transformer image and pushes to registry
	@docker build -t eu.gcr.io/disco-mgmt-100/mlops/semantic-tagging-tfx-transformer:local -f serving/Transformer.Dockerfile . & \
	docker push eu.gcr.io/disco-mgmt-100/mlops/semantic-tagging-tfx-transformer:local

pre-commit: ## Runs the pre-commit on all files in the repo
	@pre-commit run --all-files
