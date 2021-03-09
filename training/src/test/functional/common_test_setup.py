import os

import tensorflow as tf
import tensorflow_transform as tft

import pandas as pd

import main.components.bert_model as bert_model


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

        self.label_file = self.tf_transform_output.vocabulary_file_by_name("labels")
        self.tag_file = self.tf_transform_output.vocabulary_file_by_name("tags")

        self.label_df = pd.read_csv(self.label_file, header=None)
        self.tag_df = pd.read_csv(self.tag_file, header=None)

        transformed_examples_output = os.path.join(
            self._pipeline_root, "Transform/transformed_examples"
        )
        self.transformed_train_files = os.path.join(
            transformed_examples_output,
            os.listdir(transformed_examples_output)[0],
            "train/*",
        )
        self.transformed_train_dataset = bert_model._input_fn(
            file_pattern=self.transformed_train_files,
            tf_transform_output=self.tf_transform_output,
            shuffle=False,
            batch_size=20,
        )
