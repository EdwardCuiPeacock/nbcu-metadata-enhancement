GOOGLE_CLOUD_PROJECT="res-nbcupea-dev-ds-sandbox-001"
ENDPOINT=df6bc4688870067-dot-us-east1.pipelines.googleusercontent.com
CUSTOM_TFX_IMAGE=gcr.io/${GOOGLE_CLOUD_PROJECT}/tfx-pipeline
PIPELINE_NAME=metadata_dev_edc_base_0_0_4

# For local machine
# gcloud init # initialize and login if not already
# gcloud auth login # logging from webpage
# gcloud auth configure-docker # add credential to docker

alias build_pipeline_local="tfx pipeline create --pipeline-path=./local_runner.py"
alias update_pipeline_local="tfx pipeline update --pipeline-path=./local_runner.py"
alias run_pipeline_local="tfx run create --pipeline_name$PIPELINE_NAME"

alias build_pipeline="tfx pipeline create --pipeline-path=./kubeflow_dag_runner.py --endpoint=${ENDPOINT} --engine=kubeflow --build-target-image=${CUSTOM_TFX_IMAGE}"
alias update_pipeline="tfx pipeline update --pipeline-path=./kubeflow_dag_runner.py --endpoint=${ENDPOINT} --engine=kubeflow"
alias run_pipeline="tfx run create --pipeline_name=$PIPELINE_NAME --endpoint=$ENDPOINT"
