from absl import logging

from tfx.orchestration import metadata
from tfx.orchestration.beam.beam_dag_runner import (
    BeamDagRunner,
)  # Maybe use local runner?
from tfx.proto import trainer_pb2
import tensorflow as tf

from main.pipelines import local_pipeline
from main.pipelines import configs

from test.functional.common_test_setup import CommonTestSetup


class DataForTestingGenerator(CommonTestSetup):
    """
    This class will run the full pipeline once using the beam runner. The primary
    purpose is to generate test data used by subsequent tests, but since we are
    running the full pipeline we will get the added benefit of being able to
    test that a full pipeline run works. This is why this is being run as a
    tf.test.TestCase.
    """

    def setUp(self):
        super(DataForTestingGenerator, self).setUp()

    def test_run_pipeline_once(self):
        """
        This will generate test data that can be used later

        For now we use test data sitting in GCS, but the two parquet files
        will also be available locally.

        Need access to GCS anyway to initialize the embeddings
        """
        pipeline = local_pipeline.create_pipeline(
            pipeline_name=configs.PIPELINE_NAME,
            pipeline_root=self._pipeline_root,
            data_path=configs.DATA_PATH_TEST, 
            preprocessing_fn=configs.PREPROCESSING_FN,
            run_fn=configs.RUN_FN,
            train_args=trainer_pb2.TrainArgs(num_steps=configs.TRAIN_NUM_STEPS_TEST, splits=['train']),
            eval_args=trainer_pb2.EvalArgs(num_steps=configs.EVAL_NUM_STEPS_TEST, splits=['train']),
            eval_accuracy_threshold=None,
            serving_model_dir=self._serving_model_dir,
            custom_config=configs.custom_config,
            beam_pipeline_args=configs.BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS,
            metadata_connection_config=metadata.sqlite_metadata_connection_config(
                self._metadata_path
            ),
        )

        BeamDagRunner().run(pipeline)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    tf.test.main()
