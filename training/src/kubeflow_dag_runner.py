"""Define KubeflowDagRunner to run the pipeline using Kubeflow."""

import os
from absl import logging

from pipeline import configs
from pipeline import pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.proto import trainer_pb2
from tfx.utils import telemetry_utils

# TFX pipeline produces many output files and metadata. All output data will be
# stored under this OUTPUT_DIR.
OUTPUT_DIR = os.path.join("gs://", configs.GCS_BUCKET_NAME)

# TFX produces two types of outputs, files and metadata.
# - Files will be created under PIPELINE_ROOT directory.
PIPELINE_ROOT = os.path.join(
    OUTPUT_DIR, "tfx-metadata-dev-pipeline-output", configs.PIPELINE_NAME
)

SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, "serving_model")


def run():
    """Define a kubeflow pipeline."""

    # Metadata config. The defaults works work with the installation of
    # KF Pipelines using Kubeflow. If installing KF Pipelines using the
    # lightweight deployment option, you may need to override the defaults.
    # If you use Kubeflow, metadata will be written to MySQL database inside
    # Kubeflow cluster.
    metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()

    # This pipeline automatically injects the Kubeflow TFX image if the
    # environment variable 'KUBEFLOW_TFX_IMAGE' is defined. Currently, the tfx
    # cli tool exports the environment variable to pass to the pipelines.
    # TODO(b/157598477) Find a better way to pass parameters from CLI handler to
    # pipeline DSL file, instead of using environment vars.
    tfx_image = os.environ.get("KUBEFLOW_TFX_IMAGE", None)

    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=metadata_config, tfx_image=tfx_image
    )
    pod_labels = kubeflow_dag_runner.get_default_pod_labels().update(
        {telemetry_utils.LABEL_KFP_SDK_ENV: "tfx-template"}
    )
    kubeflow_dag_runner.KubeflowDagRunner(
        config=runner_config, pod_labels_to_attach=pod_labels
    ).run(
        pipeline.create_pipeline(
            pipeline_name=configs.PIPELINE_NAME,
            pipeline_root=PIPELINE_ROOT,
            data_path=configs.DATA_PATH,
            query=None,
            preprocessing_fn=configs.PREPROCESSING_FN,
            run_fn=configs.RUN_FN,
            train_args=trainer_pb2.TrainArgs(num_steps=configs.TRAIN_NUM_STEPS),
            eval_args=trainer_pb2.EvalArgs(num_steps=configs.EVAL_NUM_STEPS),
            eval_accuracy_threshold=configs.EVAL_ACCURACY_THRESHOLD,
            serving_model_dir=SERVING_MODEL_DIR,
            beam_pipeline_args=configs.DATAFLOW_BEAM_PIPELINE_ARGS,
            ai_platform_training_args=configs.GCP_AI_PLATFORM_TRAINING_ARGS,
        )
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    run()
