"""Define KubeflowDagRunner to run the pipeline using Kubeflow."""

import os
from absl import logging

from main.pipelines import configs
from main.pipelines import base_pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.proto import trainer_pb2
from tfx.utils import telemetry_utils

##########################################
# Defaults
##########################################
# TFX pipeline produces many output files and metadata. All output data will be
# stored under this OUTPUT_DIR.
OUTPUT_DIR = os.path.join("gs://", configs.GCS_BUCKET_NAME)
PIPELINE_NAME = configs.PIPELINE_NAME
OUTPUT_FILENAME = f"{PIPELINE_NAME}.tar.gz"
#OUTPUT_FILENAME = f"{PIPELINE_NAME}.yaml"
IMAGE = configs.IMAGE


def run(output_dir=OUTPUT_DIR, output_filename=OUTPUT_FILENAME, pipeline_name=PIPELINE_NAME, image=IMAGE):
    # TFX produces two types of outputs, files and metadata.
    # - Files will be created under PIPELINE_ROOT directory.
    pipeline_root = os.path.join(
        output_dir, "tfx-metadata-dev-pipeline-output", pipeline_name
    )

    serving_model_dir = os.path.join(pipeline_root, "serving_model")

    metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()

    tfx_image = os.environ.get("KUBEFLOW_TFX_IMAGE", image)

    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=metadata_config,
        tfx_image=tfx_image
    )

    kubeflow_dag_runner.KubeflowDagRunner(
        config=runner_config, output_filename=output_filename
    ).run(
        base_pipeline.create_pipeline(
            pipeline_name=pipeline_name,
            pipeline_root=pipeline_root,
            query=configs.query,
            preprocessing_fn=configs.PREPROCESSING_FN,
            run_fn=configs.RUN_FN,
            train_args=trainer_pb2.TrainArgs(num_steps=configs.TRAIN_NUM_STEPS, splits=['train']),
            eval_args=trainer_pb2.EvalArgs(num_steps=configs.EVAL_NUM_STEPS, splits=['train']),
            eval_accuracy_threshold=None,
            serving_model_dir=serving_model_dir,
            custom_config=configs.custom_config,
            beam_pipeline_args=configs.BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS,
            ai_platform_training_args=configs.GCP_AI_PLATFORM_TRAINING_ARGS #None, #configs.GCP_AI_PLATFORM_TRAINING_ARGS
        )
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    run()