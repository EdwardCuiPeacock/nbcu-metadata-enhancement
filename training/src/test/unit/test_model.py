from mock import patch
from unittest.mock import Mock

import tensorflow as tf
import pandas as pd
import numpy as np

import main.components.bert_model as bert_model


class ModelTest(tf.test.TestCase):
    @classmethod 
    @patch("main.components.bert_model.N_TAGS", 5)
    def setUpClass(cls):
        cls.model = bert_model.get_compiled_model()

    @patch("main.components.bert_model.tf.data.TFRecordDataset", autospec=True)
    def test_gzip_reader_fn(self, mockDataset):
        filenames = tf.data.Dataset.from_tensor_slices(["file1", "file2", "file3"])

        mockDataset.return_value = "did we return dataset?"
        tfrecord = bert_model._gzip_reader_fn(filenames)

        mockDataset.assert_called_with(filenames, compression_type="GZIP")
        self.assertEqual(tfrecord, "did we return dataset?")

    @patch(
        "main.components.bert_model.tf.data.experimental.make_batched_features_dataset",
        autospec=True,
    )
    def test_input_fn_uses_defaults(self, mockBatchedDataset):
        file_pattern = "path/to/file/*"
        tf_transform_output_mock = Mock()
        feature_spec = {"program_longsynopsis_xf": tf.io.FixedLenFeature([], tf.string),
                        "tags": tf.io.FixedLenFeature([53], tf.int64)
                        }
        tf_transform_output_mock.transformed_feature_spec.return_value.copy.return_value = (
            feature_spec
        )

        test_dataset = tf.data.Dataset.from_tensor_slices((["features"], ["tags"]))
        mockBatchedDataset.return_value = test_dataset
        dataset = bert_model._input_fn(
            file_pattern, tf_transform_output_mock
        )

        mockBatchedDataset.assert_called_with(
            file_pattern=file_pattern,
            batch_size=64,
            features=feature_spec,
            reader=bert_model._gzip_reader_fn,
            shuffle=True,
            label_key="tags",
            num_epochs=None,
        )
        for elem in dataset.take(1):
            self.assertEqual(elem[0], ["features"])
            self.assertEqual(elem[1], ["tags"])

    @patch(
        "main.components.bert_model.tf.data.experimental.make_batched_features_dataset",
        autospec=True,
    )
    def test_input_fn_overrides_defaults(self, mockBatchedDataset):
        file_pattern = "path/to/file/*"
        tf_transform_output_mock = Mock()
        feature_spec = {"program_longsynopsis_xf": tf.io.FixedLenFeature([], tf.string),
                        "tags": tf.io.FixedLenFeature([53], tf.int64)
                        }
        tf_transform_output_mock.transformed_feature_spec.return_value.copy.return_value = (
            feature_spec
        )
        batch_size = 32
        shuffle = False
        epochs = 10

        test_dataset = tf.data.Dataset.from_tensor_slices((["features"], ["tags"]))
        mockBatchedDataset.return_value = test_dataset

        dataset = bert_model._input_fn(
            file_pattern,
            tf_transform_output_mock,
            batch_size=batch_size,
            shuffle=shuffle,
            epochs=epochs,
        )
        mockBatchedDataset.assert_called_with(
            file_pattern=file_pattern,
            batch_size=32,
            features=feature_spec,
            reader=bert_model._gzip_reader_fn,
            shuffle=False,
            label_key="tags",
            num_epochs=10,
        )

        for elem in dataset.take(1):
            self.assertEqual(elem[0], ["features"])
            self.assertEqual(elem[1], ["tags"])

    def test_build_bert_tagger(self):
        preprocessing_layer = self.model.layers[1]
        encoding_layer = self.model.layers[2]
        self.assertFalse(preprocessing_layer.trainable)
        self.assertTrue(encoding_layer.trainable)
        # Make sure the inputs and outputs line up
        self.assertAllEqual(encoding_layer.input.keys(), 
                            preprocessing_layer.output.keys())
        self.assertEqual(len(encoding_layer.output['pooled_output'].shape), 2)

        dense_layer = self.model.layers[3]
        self.assertEqual(dense_layer.output_shape[1], 5)

    def test_get_compiled_model(self):
        self.assertTrue(self.model._is_compiled)
        self.assertIsInstance(self.model.distribute_strategy, tf.distribute.MirroredStrategy)

        expected_optimizer_config = tf.keras.optimizers.Adam().get_config().copy()
        expected_optimizer_config.update({"learning_rate": 0.00003})
        expected_optimizer_config.update({"epsilon": 1e-8})
        self.assertEqual(self.model.optimizer.get_config(), expected_optimizer_config)

        self.assertIsInstance(self.model.loss, tf.keras.losses.BinaryCrossentropy)

if __name__ == "__main__":
    tf.test.main()
