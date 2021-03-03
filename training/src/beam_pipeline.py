import os 
from absl import logging

from main.pipelines import base_pipeline
from main.pipelines import configs 
from tfx.orchestration import metadata
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner 
from tfx.proto import trainer_pb2 
import tempfile 

OUTPUT_DIR = tempfile.mkdtemp(prefix="beam_test")

PIPELINE_ROOT = os.path.join(OUTPUT_DIR, "tfx_pipeline_output", configs.PIPELINE_NAME)
METADATA_PATH = os.path.join(OUTPUT_DIR, "tfx_metadata", configs.PIPELINE_NAME, "metadata.db")
SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, "serving_model")

# TODO: Put this in configs? 
with open('main/queries/ingest_query.sql', 'r') as input_query:
    query = input_query.read()


def run():
    BeamDagRunner().run(
        base_pipeline.create_pipeline(
            pipeline_name=configs.PIPELINE_NAME,
            pipeline_root=PIPELINE_ROOT,
            data_path=configs.DATA_PATH_TEST,
            query=query,
            preprocessing_fn=configs.PREPROCESSING_FN,
            run_fn=configs.RUN_FN, 
            train_args=trainer_pb2.TrainArgs(num_steps=configs.TRAIN_NUM_STEPS_TEST),
            eval_args=trainer_pb2.EvalArgs(num_steps=configs.EVAL_NUM_STEPS_TEST),
            eval_accuracy_threshold=None,
            serving_model_dir=SERVING_MODEL_DIR,
            beam_pipeline_args=configs.BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS, 
            metadata_connection_config=metadata.sqlite_metadata_connection_config(
                METADATA_PATH
            ),
        )
    )

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    logging.info(f"Directory at {OUTPUT_DIR}")
    run()