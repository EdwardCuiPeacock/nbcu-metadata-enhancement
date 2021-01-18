import os
from absl import logging
from typing import Text

from tfx.orchestration import metadata
from tfx.orchestration.beam.beam_dag_runner import (
    BeamDagRunner,
)  # Maybe use local runner?
from tfx.proto import trainer_pb2
import tensorflow as tf

from main.pipelines import base_pipeline
from main.pipelines import config
from test.functional.common_test_setup import CommonTestSetup


class PipelineTest(CommonTestSetup):
    """
    This class will validate the execution of TFX components in the training pipeline.

    We will examine the output of the pipeline run from the generate_test_data.py file
    and then do another pipeline run here to make further assertions about expected
    pipeline output.

    Therefore, this must be run AFTER generate_test_data.py
    """

    def setUp(self):
        # Run the exact same setup as DataForTestingGenerator so we are looking in the same
        # place for pipeline artifacts
        super().setUp()

    def assertExecutedOnce(self, component: Text) -> None:
        """Check that the component is executed exactly once."""
        component_path = os.path.join(self._pipeline_root, component)
        self.assertTrue(tf.io.gfile.exists(component_path))
        outputs = tf.io.gfile.listdir(component_path)
        for output in outputs:
            execution = tf.io.gfile.listdir(os.path.join(component_path, output))
            self.assertEqual(1, len(execution))

    def assertPipelineExecution(self) -> None:
        self.assertExecutedOnce("ExampleValidator")
        self.assertExecutedOnce("FileBasedExampleGen")
        self.assertExecutedOnce("Pusher")
        # self.assertExecutedOnce("ImporterNode")
        self.assertExecutedOnce("StatisticsGen")
        self.assertExecutedOnce("ThresholdValidator")
        self.assertExecutedOnce("Trainer")
        self.assertExecutedOnce("Transform")

    def test_pipeline_execution(self):
        self.assertPipelineExecution()
        # The pipeline we ran in generate_test_data has a high evaluation threshold for pushing
        # If the validation threshold is not met, the model is not "blessed" and is therefore not pushed
        # We expect the model will not be blessed and should not have been pushed
        threshold_blessing = os.path.join(
            self._pipeline_root, "ThresholdValidator/blessing"
        )
        blessing_execution = os.path.join(
            threshold_blessing, os.listdir(threshold_blessing)[0]
        )
        self.assertEqual(os.listdir(blessing_execution)[0], "NOT_BLESSED")

        pusher_output = os.path.join(self._pipeline_root, "Pusher/pushed_model")
        execution = os.listdir(pusher_output)[0]
        self.assertEqual(len(os.listdir(os.path.join(pusher_output, execution))), 0)

        # Since we did not bless the model and did not push, there should be no
        # serving model directory
        self.assertFalse(tf.io.gfile.exists(self._serving_model_dir))

        # Now we check the artifacts from each component in the pipeline
        # by looking at the metadata
        expected_execution_count_first_run = 8  # 8 Components in pipeline
        self.assertTrue(tf.io.gfile.exists(self._metadata_path))
        metadata_config = metadata.sqlite_metadata_connection_config(
            self._metadata_path
        )

        with metadata.Metadata(metadata_config) as m:
            artifact_count = len(m.store.get_artifacts())
            execution_count = len(m.store.get_executions())
            self.assertGreaterEqual(artifact_count, execution_count)
            self.assertEqual(expected_execution_count_first_run, execution_count)

        # Run pipeline again with a lower validation threshold
        # Since validation threshold should be met now, model is "blessed" and is therefore pushed
        # Purpose is to check another possible path through the pipeline when a model is actually pushed
        pipeline_low_threshold = base_pipeline.create_pipeline(
            pipeline_name=config.PIPELINE_NAME,
            pipeline_root=self._pipeline_root,
            data_path=config.TEST_DATA_PATH,
            preprocessing_fn=config.PREPROCESSING_FN,
            run_fn=config.RUN_FN,
            train_args=trainer_pb2.TrainArgs(num_steps=config.TRAIN_NUM_STEPS_TEST),
            eval_args=trainer_pb2.EvalArgs(num_steps=config.EVAL_NUM_STEPS_TEST),
            eval_precision_threshold=config.EVAL_PRECISION_THRESHOLD_TEST,
            eval_recall_threshold=config.EVAL_RECALL_THRESHOLD_TEST,
            serving_model_dir=self._serving_model_dir,
            metadata_connection_config=metadata.sqlite_metadata_connection_config(
                self._metadata_path
            ),
        )

        BeamDagRunner().run(pipeline_low_threshold)

        # We expect the model will be blessed and should have been pushed
        blessing_execution = os.path.join(
            threshold_blessing, os.listdir(threshold_blessing)[1]
        )
        self.assertEqual(os.listdir(blessing_execution)[0], "BLESSED")

        execution = sorted(os.listdir(pusher_output), key=int)[-1]
        self.assertNotEqual(len(os.listdir(os.path.join(pusher_output, execution))), 0)

        # Since we did bless the model and did push, there should now be a
        # serving model directory
        self.assertTrue(tf.io.gfile.exists(self._serving_model_dir))

        # Check the metadata store to make sure it registered
        # executions of all components from 2nd pipeline run
        with metadata.Metadata(metadata_config) as m:
            self.assertEqual(
                expected_execution_count_first_run * 2, len(m.store.get_executions())
            )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    tf.test.main()
