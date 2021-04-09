ENDPOINT=df6bc4688870067-dot-us-east1.pipelines.googleusercontent.com
GOOGLE_CLOUD_PROJECT=res-nbcupea-dev-ds-sandbox-001
CUSTOM_TFX_IMAGE=gcr.io/${GOOGLE_CLOUD_PROJECT}/metadata-dev-pipeline-base
PIPELINE_NAME=metadata-dev-base
python3 -c "import tfx; print('TFX version: {}'.format(tfx.__version__))"

alias run_pipeline="tfx run create --pipeline_name=$PIPELINE_NAME --endpoint=$ENDPOINT"
alias update_pipeline="tfx pipeline update --pipeline-path=src/kubeflow_dag_runner.py --endpoint=${ENDPOINT} --engine=kubeflow"
alias build_pipeline="tfx pipeline create --pipeline-path=src/kubeflow_dag_runner.py --endpoint=${ENDPOINT} --engine=kubeflow --build-target-image=${CUSTOM_TFX_IMAGE}"

