import os
from absl import logging

from main.pipelines import configs
from main.pipelines import base_pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
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
    """Define a local pipeline."""
    pipeline_root = configs.PIPELINE_ROOT

    serving_model_dir = os.path.join(pipeline_root, "serving_model")

    tfx_image = os.environ.get("KUBEFLOW_TFX_IMAGE", image)

    LocalDagRunner().run(
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
            ai_platform_training_args=None if not configs.USE_AI_PLATFORM else configs.GCP_AI_PLATFORM_TRAINING_ARGS
        )
    )
    


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    run()
