import os

import tensorflow as tf
import tensorflow_transform as tft

import pandas as pd

import main.components.component_utils as component_utils


class CommonTestSetup(tf.test.TestCase):
    """
    Class for sharing a setup function between functional tests so we are always
    looking for artifacts in the same place
    """

    def setUp(self):
        super().setUp()

        # This directory will contain pipeline artifacts that we can use for later
        # functional tests
        self._pipeline_dir = "pipeline_output"

        # Pipeline artifacts written to pipeline root in pipeline_output/tfx_pipeline_output/
        # Metadata written to SQLite database in pipeline_output/tfx_metadata/
        self._pipeline_root = os.path.join(self._pipeline_dir, "tfx_pipeline_output")
        self._metadata_path = os.path.join(
            self._pipeline_dir, "tfx_metadata", "metadata.db"
        )
        self._serving_model_dir = os.path.join(self._pipeline_root, "serving_model")

    def setTransformOutputs(self):
        transform_graph_output = os.path.join(
            self._pipeline_root, "Transform/transform_graph"
        )
        self.transform_output = os.path.join(
            transform_graph_output, os.listdir(transform_graph_output)[0]
        )
        self.tf_transform_output = tft.TFTransformOutput(self.transform_output)

        self.num_tags = self.tf_transform_output.vocabulary_size_by_name("tags")
        self.tag_file = self.tf_transform_output.vocabulary_file_by_name("tags")
        self.vocab_size = self.tf_transform_output.vocabulary_size_by_name("vocab")
        self.vocab_file = self.tf_transform_output.vocabulary_file_by_name("vocab")

        transformed_examples_output = os.path.join(
            self._pipeline_root, "Transform/transformed_examples"
        )
        self.transformed_train_files = os.path.join(
            transformed_examples_output,
            os.listdir(transformed_examples_output)[0],
            "train/*",
        )
        self.transformed_eval_files = os.path.join(
            transformed_examples_output,
            os.listdir(transformed_examples_output)[0],
            "eval/*",
        )

        self.tag_df = pd.read_csv(self.tag_file, header=None)
        self.tag_lookup_table = component_utils.create_tag_lookup_table(self.tag_file)
        self.vocab_df = pd.read_csv(self.vocab_file, header=None)
        self.vocab_lookup_table = component_utils.create_tag_lookup_table(
            self.vocab_file
        )

        self.transformed_train_dataset = component_utils._input_fn(
            file_pattern=self.transformed_train_files,
            tf_transform_output=self.tf_transform_output,
            num_tags=self.num_tags,
            shuffle=False,
            table=self.tag_lookup_table,
            batch_size=10,
        )

        self.transformed_eval_dataset = component_utils._input_fn(
            file_pattern=self.transformed_eval_files,
            tf_transform_output=self.tf_transform_output,
            num_tags=self.num_tags,
            shuffle=False,
            table=self.tag_lookup_table,
            batch_size=10,
        )
