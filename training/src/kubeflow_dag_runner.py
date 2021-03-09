"""Define KubeflowDagRunner to run the pipeline using Kubeflow."""

import os
from absl import logging

from main.pipelines import configs
from main.pipelines import base_pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.proto import trainer_pb2
from tfx.utils import telemetry_utils

# TFX pipeline produces many output files and metadata. All output data will be
# stored under this OUTPUT_DIR.
OUTPUT_DIR = os.path.join("gs://", configs.GCS_BUCKET_NAME)

# TFX produces two types of outputs, files and metadata.
# - Files will be created under PIPELINE_ROOT directory.
PIPELINE_ROOT = os.path.join(
    OUTPUT_DIR, "tfx-metadata-dev-pipeline-output_3", configs.PIPELINE_NAME
)

SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, "serving_model")
OUTPUT_FILENAME = f"{configs.PIPELINE_NAME}.yaml"

# TODO: Put this in configs? 
with open('main/queries/ingest_query.sql', 'r') as input_query:
    query = input_query.read()


def run():
    metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()

    tfx_image = os.environ.get("KUBEFLOW_TFX_IMAGE", configs.IMAGE)

    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=metadata_config, tfx_image=tfx_image
    )

    kubeflow_dag_runner.KubeflowDagRunner(
        config=runner_config, output_filename=OUTPUT_FILENAME
    ).run(
        base_pipeline.create_pipeline(
            pipeline_name=configs.PIPELINE_NAME,
            pipeline_root=PIPELINE_ROOT,
            data_path=configs.DATA_PATH,
            query=query,
            preprocessing_fn=configs.PREPROCESSING_FN,
            run_fn=configs.RUN_FN,
            train_args=trainer_pb2.TrainArgs(num_steps=configs.TRAIN_NUM_STEPS, splits=['train']),
            eval_args=trainer_pb2.EvalArgs(num_steps=configs.EVAL_NUM_STEPS, splits=['train']),
            eval_accuracy_threshold=None,
            serving_model_dir=SERVING_MODEL_DIR,
            beam_pipeline_args=configs.BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS,
            ai_platform_training_args=configs.GCP_AI_PLATFORM_TRAINING_ARGS #None, #configs.GCP_AI_PLATFORM_TRAINING_ARGS
        )
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    run()
